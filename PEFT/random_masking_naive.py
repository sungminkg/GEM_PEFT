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

class RandomMaskingLinear(torch.nn.Module):
    def __init__(self, base_Linear: nn.Linear, in_dim: int, out_dim: int):
        super().__init__()
        self.base_Linear = base_Linear
        self.tunable_weight = nn.Parameter(self.base_Linear.weight.new_zeros((out_dim, in_dim)))
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
        return F.linear(x, self.base_Linear.weight + masked_weight, self.base_Linear.bias)

class RandomMasking:
    def __init__(self, model, masking_ratio):
        self.model = model
        self.masking_ratio = masking_ratio
        assert 0.0 <= masking_ratio <= 1.0

        self.linear_module_list = ['q_proj', 'v_proj']
        self.tunable_param_count = 0
        self.total_param_count = sum(p.numel() for p in model.parameters())

        # Calculate target params per module
        num_layers = sum(1 for name, _ in model.named_modules() if any(name.endswith(m) for m in self.linear_module_list)) // 2
        total_tuning_params = int(self.total_param_count * (1 - self.masking_ratio))
        k_per_module = total_tuning_params // (num_layers * len(self.linear_module_list))

        logger.info(f"Target tuning params per module: {k_per_module}")

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
        logger.info(f"Tunable Parameters: {self.tunable_param_count} / {self.total_param_count} ({percent:.4f}%)")




# import logging
# import torch
# from torch import nn
# from torch.nn import functional as F

# logger = logging.getLogger(__name__)

# def find_module(root_module: nn.Module, key: str):
#     """
#     Find a module with a specific name in a Transformer model
#     From OpenDelta https://github.com/thunlp/OpenDelta
#     """
#     sub_keys = key.split(".")
#     parent_module = root_module
#     for sub_key in sub_keys[:-1]:
#         parent_module = getattr(parent_module, sub_key)
#     module = getattr(parent_module, sub_keys[-1])
#     return parent_module, sub_keys[-1], module


# class RandomMaskingLinear(torch.nn.Module):
#     def __init__(self, base_Linear: nn.Linear, in_dim: int, out_dim: int, masking_prob: float = 0.0):
#         super().__init__()
#         self.base_Linear = base_Linear
#         self.masking = (torch.rand(out_dim, in_dim) > masking_prob).to(
#             base_Linear.weight.device).detach().to(base_Linear.weight.dtype)

#         self.tunable_weight = nn.Parameter(self.base_Linear.weight.new_zeros((out_dim, in_dim)))

#     def train(self, mode: bool = True):
#         nn.Linear.train(self, mode)

#     def forward(self, x: torch.Tensor):
#         tmp1 = F.linear(x, self.base_Linear.weight, self.base_Linear.bias)
#         tmp2 = F.linear(x, self.tunable_weight * self.masking)
#         return tmp1 + tmp2


# class RandomMasking:
#     def __init__(self, model, masking_prob):
#         self.model = model
#         self.masking_prob = masking_prob
#         assert 0.0 <= masking_prob <= 1.0

#         self.tunable_param_count = 0
#         self.total_param_count = sum(p.numel() for p in model.parameters())

#         linear_module_list = ['q_proj', 'v_proj']

#         for key, _ in self.model.named_modules():
#             for module_name in linear_module_list:
#                 if key[-len(module_name):] == module_name:
#                     parent_module, sub_key, module = find_module(self.model, key)
#                     if module_name == 'fc1':
#                         out_dim = self.model.config.ffn_dim
#                     else:
#                         out_dim = self.model.config.hidden_size
#                     if module_name == 'fc2':
#                         in_dim = self.model.config.ffn_dim
#                     else:
#                         in_dim = self.model.config.hidden_size

#                     # Replace the module with RandomMaskingLinear
#                     masking_layer = RandomMaskingLinear(base_Linear=module, in_dim=in_dim, out_dim=out_dim,
#                                                         masking_prob=self.masking_prob)
#                     setattr(parent_module, sub_key, masking_layer)

#                     # Count actual tunable parameters in the masked layers
#                     self.tunable_param_count += (masking_layer.masking > 0).sum().item()

#         for n, p in model.named_parameters():
#             if "tunable" not in n:
#                 p.requires_grad = False
#             else:
#                 p.requires_grad = True

#         for n, p in model.named_parameters():
#             logger.info(f"{n} {p.requires_grad} {p.dtype}")

#         # Calculate and log the percentage of tunable parameters
#         tunable_percentage = (self.tunable_param_count / self.total_param_count) * 100
#         logger.info(f"Tunable Parameters: {self.tunable_param_count} / {self.total_param_count} ({tunable_percentage}%)")