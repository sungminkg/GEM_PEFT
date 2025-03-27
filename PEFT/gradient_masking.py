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


class GradientMaskingLinear(nn.Module):
    def __init__(self, base_Linear: nn.Linear, in_dim: int, out_dim: int, masking_prob: float = 0.0):
        super().__init__()
        self.base_Linear = base_Linear
        self.masking_prob = masking_prob
        self.masking = None  # Mask will be dynamically calculated
        self.tunable_weight = nn.Parameter(self.base_Linear.weight.new_zeros(self.base_Linear.weight.shape))

    def apply_mask(self, grad_tensor: torch.Tensor):
        grad_norms = grad_tensor.abs()
        k = int(self.tunable_weight.numel() * (1 - self.masking_prob))
        if k > 0:
            threshold = torch.topk(grad_norms.view(-1), k, largest=True, sorted=False).values[-1]
            self.masking = (grad_norms >= threshold).float()
        else:
            self.masking = torch.zeros_like(self.tunable_weight)

    def forward(self, x: torch.Tensor):
        tmp1 = F.linear(x, self.base_Linear.weight, self.base_Linear.bias)
        tmp2 = F.linear(x, self.tunable_weight * self.masking if self.masking is not None else self.tunable_weight)
        return tmp1 + tmp2


class GradientMasking:
    def __init__(self, model, gradients, masking_prob):
        self.model = model
        self.gradients = gradients  # Use the provided gradients dictionary
        self.masking_prob = masking_prob
        assert 0.0 <= masking_prob <= 1.0

        self.linear_module_list = ['q_proj', 'v_proj']
        # linear_module_list = ['q_proj', 'v_proj', 'out_proj','fc1', 'fc2']

        # Replace specific modules with GradientMaskingLinear
        for key, module in self.model.named_modules():
            for module_name in self.linear_module_list:
                if key.endswith(module_name):
                    parent_module, sub_key, module = find_module(self.model, key)
                    in_dim, out_dim = module.weight.shape
                    setattr(parent_module, sub_key,
                            GradientMaskingLinear(base_Linear=module, in_dim=in_dim, out_dim=out_dim,
                                                  masking_prob=self.masking_prob))

        # Freeze non-tunable parameters
        for n, p in model.named_parameters():
            if "tunable_weight" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        # Log parameter states
        for n, p in model.named_parameters():
            logger.info(f"{n} {p.requires_grad} {p.dtype}")
            
    def _normalize_key(self, key):
        if ".tunable_weight" in key:
            key = key.replace(".tunable_weight", ".weight")
        elif ".base_Linear.weight" in key:
            key = key.replace(".base_Linear.weight", ".weight")
        elif ".base_Linear.bias" in key:
            key = key.replace(".base_Linear.bias", ".bias")
        return key

    def apply_gradient_masking(self):
        for name, module in self.model.named_modules():
            if isinstance(module, GradientMaskingLinear) and any(layer_name in name for layer_name in self.linear_module_list):
                grad_key = self._normalize_key(name + ".weight")
                grad_tensor = self.gradients.get(grad_key, None)
                if grad_tensor is not None:
                    logger.info(f"Applying mask to layer: {name}")
                    module.apply_mask(grad_tensor)
                else:
                    if grad_tensor is None:
                        logger.warning(f"Gradient missing for {grad_key}")