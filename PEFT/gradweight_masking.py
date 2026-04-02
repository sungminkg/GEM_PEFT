import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GradWeightMaskingLinear(nn.Module):
    def __init__(self, base_Linear: nn.Linear, masking_prob: float = 0.0):
        super().__init__()
        self.base_Linear = base_Linear
        self.masking_prob = masking_prob
        self.masking = None  # Mask will be dynamically calculated
        self.tunable_weight = nn.Parameter(self.base_Linear.weight.clone().detach())  # Clone the original weight

    def apply_mask(self, k, mask_mode, grad_tensor: torch.Tensor, weight_tensor: torch.Tensor):  
        if mask_mode == 'gradient':
            grad = grad_tensor.abs()
            ratio = grad
            
        elif mask_mode == 'gradweight':
            grad = grad_tensor.to(self.base_Linear.weight.device).abs()
            weight = weight_tensor.to(self.base_Linear.weight.device).abs()
            ratio = grad / (weight + 1e-8)
        elif mask_mode == 'weight':
            weight = weight_tensor.abs()
            ratio = weight
            
        if k > 0:
            threshold = torch.topk(ratio.view(-1), k, largest=True, sorted=False).values[-1]
            if torch.isinf(threshold):
                threshold = torch.topk(grad_tensor.abs().view(-1), k, largest=True, sorted=False).values[-1]
                self.masking = (ratio >= threshold).float().to(self.base_Linear.weight.device).detach().to(self.base_Linear.weight.dtype)
            else:
                self.masking = (ratio >= threshold).float().to(self.base_Linear.weight.device).detach().to(self.base_Linear.weight.dtype)
        else:
            self.masking = torch.zeros_like(self.tunable_weight).to(self.base_Linear.weight.device).detach().to(self.base_Linear.weight.dtype)

    def forward(self, x: torch.Tensor):
        masked_weight = (self.tunable_weight * self.masking if self.masking is not None else self.tunable_weight).to(self.base_Linear.weight.dtype)
        return F.linear(x, self.base_Linear.weight + masked_weight, self.base_Linear.bias)


class GradWeightMasking:
    def __init__(self, model, mask_mode, gradients, weights, masking_prob):
        self.model = model
        self.mask_mode = mask_mode
        self.gradients = gradients
        self.weights = weights
        self.masking_prob = masking_prob
        self.linear_module_list = ['q_proj', 'v_proj']
        self.tunable_param_count = 0
        self.total_param_count = sum(p.numel() for p in model.parameters())

        # Replace specific modules with GradWeightMaskingLinear
        for key, module in self.model.named_modules():
            if any(key.endswith(layer_name) for layer_name in self.linear_module_list) and isinstance(module, nn.Linear):
                parent_module, sub_key = self._get_parent_module_and_key(key)
                grad_masking_layer = GradWeightMaskingLinear(base_Linear=module, masking_prob=self.masking_prob)
                setattr(parent_module, sub_key, grad_masking_layer)

        # Apply masks and count tunable parameters
        self.apply_grad_weight_masking()

        # Calculate and log the percentage of tunable parameters
        tunable_percentage = (self.tunable_param_count / self.total_param_count) * 100
        logger.info(
            "Tunable Parameters: %s / %s (%.6f%%)",
            self.tunable_param_count,
            self.total_param_count,
            tunable_percentage,
        )


    def _tuning_params_per_module(self):
        attention_layer_names = set()
        for name, _ in self.model.named_modules():
            if ".self_attn." in name: 
                layer_name = name.split(".self_attn.")[0]
                attention_layer_names.add(layer_name)
        num_attention_layers = len(attention_layer_names)
        tuning_prob_per_module = (1 - self.masking_prob) / (num_attention_layers * len(self.linear_module_list))
        tuning_params_per_module = int(self.total_param_count * tuning_prob_per_module)
        return tuning_params_per_module
        
    def _get_parent_module_and_key(self, full_name):
        sub_keys = full_name.split(".")
        parent_module = self.model
        for sub_key in sub_keys[:-1]:
            parent_module = getattr(parent_module, sub_key)
        return parent_module, sub_keys[-1]

    def _normalize_key(self, key):
        if ".tunable_weight" in key:
            key = key.replace(".tunable_weight", ".weight")
        elif ".base_Linear.weight" in key:
            key = key.replace(".base_Linear.weight", ".weight")
        elif ".base_Linear.bias" in key:
            key = key.replace(".base_Linear.bias", ".bias")
        return key

    def apply_grad_weight_masking(self):
        for name, module in self.model.named_modules():
            if isinstance(module, GradWeightMaskingLinear) and any(layer_name in name for layer_name in self.linear_module_list):
                grad_key = self._normalize_key(name + ".weight")
                weight_key = self._normalize_key(name + ".weight")

                # Extract gradient and weight tensors
                grad_tensor = self.gradients.get(grad_key, None)
                weight_tensor = self.weights.get(weight_key, None)
                if grad_tensor is not None:
                    grad_tensor = grad_tensor.to(self.model.device)
                if weight_tensor is not None:
                    weight_tensor = weight_tensor.to(self.model.device)
                if grad_tensor is not None and weight_tensor is not None:
                    if torch.isnan(grad_tensor).any() or torch.isinf(grad_tensor).any():
                        logger.warning(f"Skipping {name} due to invalid gradient values (inf/nan).")
                        continue
                    logger.info(f"Applying mask to layer: {name}")
                    k = self._tuning_params_per_module()
                    module.apply_mask(k, self.mask_mode, grad_tensor, weight_tensor)

                    # Count tunable parameters after mask application
                    if module.masking is not None:
                        self.tunable_param_count += (module.masking > 0).sum().item()
                else:
                    if grad_tensor is None:
                        logger.warning(f"Gradient missing for {grad_key}")
                    if weight_tensor is None:
                        logger.warning(f"Weight missing for {weight_key}")
