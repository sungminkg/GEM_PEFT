import logging
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class EntropyMaskingLinear(nn.Module):
    def __init__(self, base_Linear: nn.Linear, masking_prob: float = 0.0):
        super().__init__()
        self.base_Linear = base_Linear
        self.masking_prob = masking_prob
        self.masking = None
        self.tunable_weight = nn.Parameter(self.base_Linear.weight.clone().detach())

    def apply_entropy_mask(self, grad_tensor: torch.Tensor, weight_tensor: torch.Tensor, n_tuning_params):
        device = self.base_Linear.weight.device
        grad_tensor = grad_tensor.to(device)
        weight_tensor = weight_tensor.to(device)
        importance_score = grad_tensor.abs() / (weight_tensor.abs() + 1e-8)
        importance_scores_flat = importance_score.view(-1)
        k = min(n_tuning_params, importance_scores_flat.numel())
        if k > 0:
            threshold = torch.topk(importance_scores_flat, k, largest=True, sorted=False).values[-1]
            self.masking = (importance_score >= threshold).float()
        else:
            self.masking = torch.zeros_like(self.base_Linear.weight)

    def apply_gradient_mask(self, grad_tensor: torch.Tensor, n_tuning_params):
        grad_norms = grad_tensor.abs()
        k = min(n_tuning_params, grad_norms.numel())
        if k > 0:
            threshold = torch.topk(grad_norms.view(-1), k, largest=True, sorted=False).values[-1]
            self.masking = (grad_norms >= threshold).float()
        else:
            self.masking = torch.zeros_like(self.tunable_weight)

    def forward(self, x: torch.Tensor):
        masked_weight = self.tunable_weight * self.masking if self.masking is not None else self.tunable_weight
        return F.linear(x, self.base_Linear.weight + masked_weight, self.base_Linear.bias)


class GradientEntropyMasking:
    def __init__(self, model, gradients, weights, masking_prob):
        self.model = model
        self.gradients = gradients
        self.weights = weights
        self.masking_prob = masking_prob
        self.linear_module_list = ['q_proj', 'v_proj']
        self.tunable_param_count = 0
        self.total_param_count = sum(p.numel() for p in model.parameters())

        # Replace target linear modules with masking modules
        for key, module in self.model.named_modules():
            if any(key.endswith(layer_name) for layer_name in self.linear_module_list) and isinstance(module, nn.Linear):
                parent_module, sub_key = self._get_parent_module_and_key(key)
                setattr(parent_module, sub_key, EntropyMaskingLinear(base_Linear=module, masking_prob=self.masking_prob))

        # Only allow gradients for tunable weights
        for name, param in self.model.named_parameters():
            param.requires_grad = "tunable_weight" in name

        self.apply_hybrid_masking()

        tunable_percentage = (self.tunable_param_count / self.total_param_count) * 100
        logger.info(f"Tunable Parameters: {self.tunable_param_count} / {self.total_param_count} ({tunable_percentage:.2f}%)")

    def _get_parent_module_and_key(self, full_name):
        sub_keys = full_name.split(".")
        parent_module = self.model
        for sub_key in sub_keys[:-1]:
            parent_module = getattr(parent_module, sub_key)
        return parent_module, sub_keys[-1]

    def _normalize_key(self, key):
        return key.replace(".tunable_weight", ".weight").replace(".base_Linear.weight", ".weight").replace(".base_Linear.bias", ".bias")

    def apply_hybrid_masking(self):
        entropy_scores = {}
        query_param_count = 0
        value_param_count = 0
        layer_param_sizes = {}

        # Step 1: compute entropy scores (for q_proj)
        for name, module in self.model.named_modules():
            if isinstance(module, EntropyMaskingLinear) and any(n in name for n in self.linear_module_list):
                grad_key = self._normalize_key(name + ".weight")
                weight_key = self._normalize_key(name + ".weight")
                grad_tensor = self.gradients.get(grad_key)
                weight_tensor = self.weights.get(weight_key)
                
                if grad_tensor is None or weight_tensor is None:
                    continue

                grad_tensor = grad_tensor.to(self.model.device)
                weight_tensor = weight_tensor.to(self.model.device)

                param_count = module.base_Linear.weight.numel()
                layer_param_sizes[name] = param_count

                if 'q_proj' in name:
                    ratio = grad_tensor.abs() / (weight_tensor.abs() + 1e-8)
                    prob = ratio / ratio.sum()
                    entropy = -torch.sum(prob * torch.log(prob + 1e-8)).item()
                    layer_norm = ratio.norm().item()
                    entropy_score = layer_norm / (entropy + 1e-8)
                    entropy_scores[name] = entropy_score
                    query_param_count += param_count
                elif 'v_proj' in name:
                    value_param_count += param_count

        total_tuning_params = int((query_param_count + value_param_count) * (1 - self.masking_prob))

        # Step 2: normalize entropy scores
        entropy_values = torch.tensor(list(entropy_scores.values()))
        entropy_values /= entropy_values.sum()
        entropy_ratios = dict(zip(entropy_scores.keys(), entropy_values.tolist()))

        # Step 3: apply masking
        for name, module in self.model.named_modules():
            if isinstance(module, EntropyMaskingLinear):
                grad_key = self._normalize_key(name + ".weight")
                weight_key = self._normalize_key(name + ".weight")
                grad_tensor = self.gradients.get(grad_key)
                weight_tensor = self.weights.get(weight_key)

                if grad_tensor is None or weight_tensor is None:
                    continue

                if 'q_proj' in name and name in entropy_ratios:
                    n_tuning_params = int(total_tuning_params * entropy_ratios[name])
                    module.apply_entropy_mask(grad_tensor, weight_tensor, n_tuning_params)
                elif 'v_proj' in name:
                    v_ratio = layer_param_sizes[name] / value_param_count
                    n_tuning_params = int(total_tuning_params * v_ratio)
                    module.apply_gradient_mask(grad_tensor, n_tuning_params)

                if module.masking is not None:
                    self.tunable_param_count += (module.masking > 0).sum().item()
