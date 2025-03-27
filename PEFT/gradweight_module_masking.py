import logging
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

def find_module(root_module: nn.Module, key: str):
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class GradWeightModuleMaskingLinear(nn.Module):
    def __init__(self, base_Linear: nn.Linear, in_dim: int, out_dim: int, masking_prob: float = 0.0):
        super().__init__()
        self.base_Linear = base_Linear
        self.masking_prob = masking_prob
        self.masking = None
        self.tunable_weight = nn.Parameter(self.base_Linear.weight.new_zeros(self.base_Linear.weight.shape))

    def apply_mask(self, grad_tensor: torch.Tensor, weight_tensor: torch.Tensor):
        ratio = grad_tensor.abs()
        # ratio = (grad_tensor / (weight_tensor + 1e-12)).abs()
        k = int(self.tunable_weight.numel() * (1 - self.masking_prob))
        if k > 0:
            threshold = torch.topk(ratio.view(-1), k, largest=True, sorted=False).values[-1]
            self.masking = (ratio >= threshold).float()
        else:
            self.masking = torch.zeros_like(self.tunable_weight)

    def forward(self, x: torch.Tensor):
        masked_weight = (self.tunable_weight * self.masking if self.masking is not None else self.tunable_weight).to(self.base_Linear.weight.dtype)
        return F.linear(x, self.base_Linear.weight + masked_weight, self.base_Linear.bias)


class GradWeightModuleMasking:
    def __init__(self, model, gradients, weights, masking_prob):
        self.model = model
        self.gradients = gradients
        self.weights = weights
        self.masking_prob = masking_prob
        assert 0.0 <= masking_prob <= 1.0
        self.tunable_param_count = 0
        self.total_param_count = sum(p.numel() for p in model.parameters())

        # Layer and module lists for masking
        self.layer_list = ['self_attn', 'fc1']
        # self.layer_list = ['embed_tokens', 'embed_positions', 'self_attn', 'fc1', 'fc2', 'final_layer_norm']
        self.attn_module_list = ['q_proj', 'v_proj']

        # Replace matching layers/modules with GradWeightWMaskingLinear
        for key, module in self.model.named_modules():
            if any(layer in key for layer in self.layer_list) or any(attn_module in key for attn_module in self.attn_module_list):
                if isinstance(module, nn.Linear):
                    parent_module, sub_key, module = find_module(self.model, key)
                    in_dim, out_dim = module.weight.shape
                    setattr(parent_module, sub_key,
                            GradWeightModuleMaskingLinear(module, in_dim, out_dim, masking_prob=self.masking_prob))

        # Update requires_grad for tunable weights
        for n, p in self.model.named_parameters():
            p.requires_grad = "tunable_weight" in n
            
        # Apply masks and count tunable parameters
        self.apply_grad_weight_masking()
        
        # Calculate and log the percentage of tunable parameters
        tunable_percentage = (self.tunable_param_count / self.total_param_count) * 100
        logger.info(f"Tunable Parameters: {self.tunable_param_count} / {self.total_param_count} ({tunable_percentage}%)")


    def _normalize_key(self, key):
        if ".tunable_weight" in key:
            key = key.replace(".tunable_weight", ".weight")
        elif ".base_Linear.weight" in key:
            key = key.replace(".base_Linear.weight", ".weight")
        elif ".base_Linear.bias" in key:
            key = key.replace(".base_Linear.bias", ".bias")
        elif ".fc1.tunable_weight" in key:
            key = key.replace(".fc1.tunable_weight", ".weight")
        elif ".fc2.tunable_weight" in key:
            key = key.replace(".fc2.tunable_weight", ".weight")
        return key

    def apply_grad_weight_masking(self):
        for name, module in self.model.named_modules():
            if isinstance(module, GradWeightModuleMaskingLinear):
                if any(layer in name for layer in self.layer_list) or any(attn_module in name for attn_module in self.attn_module_list):
                    grad_key = self._normalize_key(name + ".weight")
                    weight_key = self._normalize_key(name + ".weight")
                    grad_tensor = self.gradients.get(grad_key, None)
                    weight_tensor = self.weights.get(weight_key, None)
                    if grad_tensor is not None and weight_tensor is not None:
                        logger.info(f"Applying mask to layer/module: {name}")
                        module.apply_mask(grad_tensor, weight_tensor)
                        
                        # Count tunable parameters after mask application
                        if module.masking is not None:
                            self.tunable_param_count += (module.masking > 0).sum().item()
                    else:
                        if grad_tensor is None:
                            logger.warning(f"Gradient missing for {grad_key}")
                        if weight_tensor is None:
                            logger.warning(f"Weight missing for {weight_key}")
