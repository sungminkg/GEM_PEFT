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
    return parent_module, sub_keys[-1], getattr(parent_module, sub_keys[-1])


class RandomMaskingLinear(torch.nn.Module):
    def __init__(self, base_linear: nn.Linear, in_dim: int, out_dim: int):
        super().__init__()
        self.base_linear = base_linear
        self.tunable_weight = nn.Parameter(self.base_linear.weight.new_zeros((out_dim, in_dim)))
        self.masking = None

    def set_mask(self, k):
        out_dim, in_dim = self.tunable_weight.shape
        total = out_dim * in_dim
        mask = torch.zeros(total, dtype=torch.bool)
        if k > 0:
            selected_indices = torch.randperm(total)[:k]
            mask[selected_indices] = 1
        self.masking = mask.view(out_dim, in_dim).to(self.tunable_weight.device).to(self.tunable_weight.dtype)

    def forward(self, x: torch.Tensor):
        masked_weight = self.tunable_weight * self.masking if self.masking is not None else self.tunable_weight
        return F.linear(x, self.base_linear.weight + masked_weight, self.base_linear.bias)


class RandomMasking:
    def __init__(self, model, masking_ratio):
        self.model = model
        self.masking_ratio = masking_ratio
        assert 0.0 <= masking_ratio <= 1.0

        self.linear_module_list = ["q_proj", "v_proj"]
        self.tunable_param_count = 0
        self.total_param_count = sum(p.numel() for p in model.parameters())

        num_layers = (
            sum(1 for name, _ in model.named_modules() if any(name.endswith(m) for m in self.linear_module_list)) // 2
        )
        total_tuning_params = int(self.total_param_count * (1 - self.masking_ratio))
        k_per_module = total_tuning_params // (num_layers * len(self.linear_module_list))
        logger.info("Target tuning params per module: %s", k_per_module)

        for key, _ in model.named_modules():
            for module_name in self.linear_module_list:
                if key.endswith(module_name):
                    parent_module, sub_key, module = find_module(model, key)
                    in_dim = out_dim = model.config.hidden_size

                    masking_layer = RandomMaskingLinear(module, in_dim, out_dim)
                    masking_layer.set_mask(k_per_module)
                    setattr(parent_module, sub_key, masking_layer)

                    self.tunable_param_count += (masking_layer.masking > 0).sum().item()

        for name, param in model.named_parameters():
            param.requires_grad = "tunable" in name

        percent = 100 * self.tunable_param_count / self.total_param_count
        logger.info(
            "Tunable Parameters: %s / %s (%.4f%%)",
            self.tunable_param_count,
            self.total_param_count,
            percent,
        )
