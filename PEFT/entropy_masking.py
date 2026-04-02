import logging

import torch
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class EntropyMaskingLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, masking_prob: float = 0.0):
        super().__init__()
        self.base_linear = base_linear
        self.masking_prob = masking_prob
        self.masking = None
        self.tunable_weight = nn.Parameter(self.base_linear.weight.clone().detach())

    def apply_mask(self, grad_tensor: torch.Tensor, weight_tensor: torch.Tensor, n_tuning_params: int):
        grad_tensor = grad_tensor.to(self.base_linear.weight.device)
        weight_tensor = weight_tensor.to(self.base_linear.weight.device)

        if n_tuning_params <= 0:
            self.masking = torch.zeros_like(
                self.base_linear.weight,
                dtype=self.base_linear.weight.dtype,
                device=self.base_linear.weight.device,
            )
            return

        importance_score = grad_tensor.abs() / (weight_tensor.abs() + 1e-8)
        flat_scores = importance_score.view(-1)
        k = min(n_tuning_params, flat_scores.numel())
        threshold = torch.topk(flat_scores, k, largest=True, sorted=False).values[-1]
        self.masking = (
            (importance_score >= threshold)
            .float()
            .to(self.base_linear.weight.device)
            .detach()
            .to(self.base_linear.weight.dtype)
        )

    def forward(self, x: torch.Tensor):
        if self.masking is not None:
            masked_weight = (self.tunable_weight * self.masking).to(self.base_linear.weight.dtype)
        else:
            masked_weight = self.tunable_weight.to(self.base_linear.weight.dtype)
        return F.linear(x, self.base_linear.weight + masked_weight, self.base_linear.bias)


class EntropyBasedMasking:
    """
    Historical name kept for compatibility with existing experiment scripts.
    This implementation corresponds to GEM, exposed through the mode name
    `entropy_gradweight_masking`.
    """

    def __init__(self, model, gradients, weights, masking_prob):
        self.model = model
        self.gradients = gradients
        self.weights = weights
        self.masking_prob = masking_prob
        self.linear_module_list = ["q_proj", "v_proj"]
        self.tunable_param_count = 0
        self.total_param_count = sum(p.numel() for p in model.parameters())
        self.ratio_mode = "naive"
        self.temperature = 10

        for key, module in self.model.named_modules():
            if any(key.endswith(layer_name) for layer_name in self.linear_module_list) and isinstance(module, nn.Linear):
                parent_module, sub_key = self._get_parent_module_and_key(key)
                setattr(
                    parent_module,
                    sub_key,
                    EntropyMaskingLinear(base_linear=module, masking_prob=self.masking_prob),
                )

        for name, param in self.model.named_parameters():
            param.requires_grad = "tunable_weight" in name

        self.apply_entropy_based_masking()

        tunable_percentage = (self.tunable_param_count / self.total_param_count) * 100
        logger.info(
            "Tunable Parameters: %s / %s (%.6f%%)",
            self.tunable_param_count,
            self.total_param_count,
            tunable_percentage,
        )

    def _get_parent_module_and_key(self, full_name):
        sub_keys = full_name.split(".")
        parent_module = self.model
        for sub_key in sub_keys[:-1]:
            parent_module = getattr(parent_module, sub_key)
        return parent_module, sub_keys[-1]

    def _normalize_key(self, key):
        if ".tunable_weight" in key:
            return key.replace(".tunable_weight", ".weight")
        if ".base_linear.weight" in key:
            return key.replace(".base_linear.weight", ".weight")
        if ".base_linear.bias" in key:
            return key.replace(".base_linear.bias", ".bias")
        return key

    def apply_entropy_based_masking(self):
        entropy_ratios = {}

        for name, module in self.model.named_modules():
            if not isinstance(module, EntropyMaskingLinear):
                continue
            if not any(layer_name in name for layer_name in self.linear_module_list):
                continue

            grad_key = self._normalize_key(name + ".weight")
            weight_key = self._normalize_key(name + ".weight")

            grad_tensor = self.gradients.get(grad_key)
            weight_tensor = self.weights.get(weight_key)
            if grad_tensor is None or weight_tensor is None:
                logger.warning("Gradient or weight missing for %s", name)
                continue

            grad_tensor = grad_tensor.to(self.model.device)
            weight_tensor = weight_tensor.to(self.model.device)
            ratio = grad_tensor.abs() / (weight_tensor.abs() + 1e-8)
            prob = ratio / ratio.sum()
            layer_norm = ratio.norm().item()
            entropy = -torch.sum(prob * torch.log(prob + 1e-8)).item()
            entropy_score = layer_norm * entropy

            if torch.isinf(torch.tensor(layer_norm)) or torch.isnan(torch.tensor(layer_norm)):
                logger.warning("Skipping %s due to invalid norm value.", name)
                continue
            if torch.isnan(torch.tensor(entropy_score)):
                logger.warning("Skipping %s due to invalid entropy score.", name)
                continue

            entropy_ratios[name] = entropy_score

        if not entropy_ratios:
            logger.warning("No valid layers were found for GEM masking.")
            return

        total_tuning_params = int(self.total_param_count * (1 - self.masking_prob))
        entropy_values = torch.tensor(list(entropy_ratios.values()))
        if self.ratio_mode == "temp_softmax":
            entropy_values = torch.log1p(entropy_values)
            entropy_values = torch.exp(entropy_values / self.temperature)
            entropy_values[torch.isnan(entropy_values)] = 0
            entropy_values /= entropy_values.sum()
        else:
            entropy_values /= entropy_values.sum()
        entropy_ratios = dict(zip(entropy_ratios.keys(), entropy_values.tolist()))

        for name, module in self.model.named_modules():
            if name not in entropy_ratios:
                continue

            n_tuning_params = int(total_tuning_params * entropy_ratios[name])
            grad_tensor = self.gradients.get(self._normalize_key(name + ".weight"))
            weight_tensor = self.weights.get(self._normalize_key(name + ".weight"))
            if grad_tensor is None or weight_tensor is None:
                logger.warning("Gradient or weight missing for %s", name)
                continue

            logger.info("Applying GEM mask to layer: %s", name)
            module.apply_mask(grad_tensor, weight_tensor, n_tuning_params)
            if module.masking is not None:
                self.tunable_param_count += (module.masking > 0).sum().item()
