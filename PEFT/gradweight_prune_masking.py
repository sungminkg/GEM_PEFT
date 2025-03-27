import logging
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GradWeightPruneMaskingLinear(nn.Module):
    def __init__(self, base_Linear: nn.Linear, masking_prob: float = 0.0, p: float = 0.0):
        super().__init__()
        self.base_Linear = base_Linear
        self.masking_prob = masking_prob
        self.p = p  # Proportion of smallest weights to prune
        self.masking = None  # Mask will be dynamically calculated
        self.tunable_weight = nn.Parameter(self.base_Linear.weight.clone().detach()) 

    def apply_mask(self, mask_mode, grad_tensor: torch.Tensor, weight_tensor: torch.Tensor):
        logger.info(f"Applying mask with prune_prob={self.p} and masking_prob={self.masking_prob}")
        num_weights = self.tunable_weight.numel()

        # Step 1: Compute gradient/weight ratio
        if mask_mode == 'gradient':
            ratio = grad_tensor.abs()
        elif mask_mode == 'gradweight':
            ratio = (grad_tensor / (weight_tensor + 1e-12)).abs()

        # Step 2: Apply pruning directly to ratio based on weight magnitude
        prune_k = int(num_weights * self.p)
        if prune_k > 0:
            prune_threshold = torch.topk(weight_tensor.abs().view(-1), prune_k, largest=False).values[-1]
            # Set ratio to 0 for weights below the pruning threshold
            ratio[weight_tensor.abs() <= prune_threshold] = 0
            logger.info(f"Prune threshold: {prune_threshold}")

        # Step 3: Apply masking based on remaining weights
        remaining_weights = int(num_weights * (1 - self.p) * (1 - self.masking_prob))
        if remaining_weights > 0:
            threshold = torch.topk(ratio.view(-1), remaining_weights, largest=True, sorted=False).values[-1]
            self.masking = (ratio >= threshold).float()
            logger.info(f"Masking threshold: {threshold}, Masking sum: {self.masking.sum().item()}")
        else:
            self.masking = torch.zeros_like(self.tunable_weight)

    def forward(self, x: torch.Tensor):
        masked_weight = (self.tunable_weight * self.masking if self.masking is not None else self.tunable_weight).to(self.base_Linear.weight.dtype)
        return F.linear(x, self.base_Linear.weight + masked_weight, self.base_Linear.bias)


class GradWeightPruneMasking:
    def __init__(self, model, mask_mode, gradients, weights, masking_prob, prune_prob):
        self.model = model
        self.mask_mode = mask_mode
        self.gradients = gradients
        self.weights = weights
        self.masking_prob = masking_prob
        self.linear_module_list = ['q_proj', 'v_proj']
        self.prune_prob = prune_prob

        # Replace specific modules with GradWeightPruneMaskingLinear
        for key, module in self.model.named_modules():
            if any(key.endswith(layer_name) for layer_name in self.linear_module_list) and isinstance(module, nn.Linear):
                parent_module, sub_key = self._get_parent_module_and_key(key)
                setattr(parent_module, sub_key, 
                        GradWeightPruneMaskingLinear(base_Linear=module, masking_prob=self.masking_prob, p=self.prune_prob))

        # Update requires_grad for tunable weights
        for name, param in self.model.named_parameters():
            if "tunable_weight" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

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
            if isinstance(module, GradWeightPruneMaskingLinear) and any(layer_name in name for layer_name in self.linear_module_list):
                grad_key = self._normalize_key(name + ".weight")
                weight_key = self._normalize_key(name + ".weight")

                # Extract gradient and weight tensors
                grad_tensor = self.gradients.get(grad_key, None)
                weight_tensor = self.weights.get(weight_key, None)
                if grad_tensor is not None and weight_tensor is not None:
                    logger.info(f"Applying mask to layer: {name}")
                    module.apply_mask(self.mask_mode, grad_tensor, weight_tensor)
                else:
                    if grad_tensor is None:
                        logger.warning(f"Gradient missing for {grad_key}")
                    if weight_tensor is None:
                        logger.warning(f"Weight missing for {weight_key}")
