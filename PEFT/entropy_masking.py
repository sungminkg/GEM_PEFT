# import logging
# import torch
# from torch import nn
# from torch.nn import functional as F

# logger = logging.getLogger(__name__)

# class EntropyMaskingLinear(nn.Module):
#     def __init__(self, base_Linear: nn.Linear, masking_prob: float = 0.0):
#         super().__init__()
#         self.base_Linear = base_Linear
#         self.masking_prob = masking_prob
#         self.masking = None
#         self.tunable_weight = nn.Parameter(self.base_Linear.weight.clone().detach())

#     def apply_mask(self, mask_mode, grad_tensor: torch.Tensor, weight_tensor: torch.Tensor, n_tuning_params):
#         grad_tensor = grad_tensor.to(self.base_Linear.weight.device)
#         weight_tensor = weight_tensor.to(self.base_Linear.weight.device)

#         if n_tuning_params <= 0:
#             self.masking = torch.zeros_like(self.base_Linear.weight, dtype=self.base_Linear.weight.dtype)
#             return

#         if mask_mode == 'gradient':
#             importance_score = grad_tensor.abs()
#         elif mask_mode == 'gradweight':
#             importance_score = grad_tensor.abs() / (weight_tensor.abs() + 1e-8)
#         else:
#             raise ValueError("Invalid mask_mode. Choose from ['gradient', 'gradweight']")

#         importance_scores_flat = importance_score.view(-1)
#         k = min(n_tuning_params, importance_scores_flat.numel())
#         if k > 0:
#             threshold = torch.topk(importance_scores_flat, k, largest=True, sorted=False).values[-1]
#             print(f'Layer: {self.base_Linear}, n_tuning_params: {n_tuning_params}, Threshold: {threshold:.6f}')
#             self.masking = (importance_score >= threshold).float().to(self.base_Linear.weight.dtype)
#         else:
#             self.masking = torch.zeros_like(self.base_Linear.weight, dtype=self.base_Linear.weight.dtype)

#     def forward(self, x: torch.Tensor):
#         if self.masking is not None:
#             masked_weight = (self.tunable_weight * self.masking).to(self.base_Linear.weight.dtype)
#         else:
#             masked_weight = self.tunable_weight.to(self.base_Linear.weight.dtype)
#         return F.linear(x, self.base_Linear.weight + masked_weight, self.base_Linear.bias)


# class EntropyBasedMasking:
#     def __init__(self, model, mask_mode, gradients, weights, masking_prob):
#         self.model = model
#         self.mask_mode = mask_mode
#         self.gradients = gradients
#         self.weights = weights
#         self.masking_prob = masking_prob
#         self.linear_module_list = ['q_proj', 'v_proj']
#         self.tunable_param_count = 0
#         self.total_param_count = sum(p.numel() for p in model.parameters())
#         self.ratio_mode = 'naive'
#         self.temperature = 10

#         for key, module in self.model.named_modules():
#             if any(key.endswith(layer_name) for layer_name in self.linear_module_list) and isinstance(module, nn.Linear):
#                 parent_module, sub_key = self._get_parent_module_and_key(key)
#                 setattr(parent_module, sub_key,
#                         EntropyMaskingLinear(base_Linear=module, masking_prob=self.masking_prob))

#         for name, param in self.model.named_parameters():
#             param.requires_grad = "tunable_weight" in name

#         self.apply_entropy_based_masking()
#         tunable_percentage = (self.tunable_param_count / self.total_param_count) * 100
#         logger.info(f"Tunable Parameters: {self.tunable_param_count} / {self.total_param_count} ({tunable_percentage:.2f}%)")

#     def _get_parent_module_and_key(self, full_name):
#         sub_keys = full_name.split(".")
#         parent_module = self.model
#         for sub_key in sub_keys[:-1]:
#             parent_module = getattr(parent_module, sub_key)
#         return parent_module, sub_keys[-1]

#     def _normalize_key(self, key):
#         return key.replace(".tunable_weight", ".weight").replace(".base_Linear.weight", ".weight").replace(".base_Linear.bias", ".bias")

#     def apply_entropy_based_masking(self):
#         entropy_ratios = {}
#         for name, module in self.model.named_modules():
#             if isinstance(module, EntropyMaskingLinear) and any(l in name for l in self.linear_module_list):
#                 grad_key = self._normalize_key(name + ".weight")
#                 weight_key = self._normalize_key(name + ".weight")
#                 grad_tensor = self.gradients.get(grad_key)
#                 weight_tensor = self.weights.get(weight_key)

#                 if grad_tensor is None or weight_tensor is None:
#                     logger.warning(f"Gradient or weight missing for {name}")
#                     continue

#                 grad_tensor = grad_tensor.to(self.model.device)
#                 weight_tensor = weight_tensor.to(self.model.device)

#                 if self.mask_mode == 'gradient':
#                     prob = grad_tensor.abs() / grad_tensor.abs().sum()
#                     layer_norm = grad_tensor.norm().item()
#                 else:
#                     ratio = grad_tensor.abs() / (weight_tensor.abs() + 1e-8)
#                     prob = ratio / ratio.sum()
#                     layer_norm = ratio.norm().item()

#                 entropy = -torch.sum(prob * torch.log(prob + 1e-8)).item()
#                 entropy_score = layer_norm / (entropy + 1e-8)
#                 entropy_ratios[name] = entropy_score

#         entropy_values = torch.tensor(list(entropy_ratios.values()))
#         if self.ratio_mode == 'temp_softmax':
#             entropy_values = torch.exp(torch.log1p(entropy_values) / self.temperature)
#             entropy_values[torch.isnan(entropy_values)] = 0
#             entropy_values /= entropy_values.sum()
#         elif self.ratio_mode == 'naive':
#             entropy_values /= entropy_values.sum()
            
#         entropy_ratios = dict(zip(entropy_ratios.keys(), entropy_values.tolist()))
#         total_tuning_params = int(self.total_param_count * (1 - self.masking_prob))

#         for name, module in self.model.named_modules():
#             if name in entropy_ratios:
#                 layer_ratio = entropy_ratios[name]
#                 n_tuning_params = int(total_tuning_params * layer_ratio)
#                 grad_tensor = self.gradients.get(self._normalize_key(name + ".weight"))
#                 weight_tensor = self.weights.get(self._normalize_key(name + ".weight"))
#                 if grad_tensor is not None and weight_tensor is not None:
#                     logger.info(f"Applying entropy-based mask to layer: {name}, Ratio: {layer_ratio:.6f}, Params: {n_tuning_params}")
#                     module.apply_mask(self.mask_mode, grad_tensor, weight_tensor, n_tuning_params)
#                     if module.masking is not None:
#                         self.tunable_param_count += (module.masking > 0).sum().item()







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

    def apply_mask(self, mask_mode, grad_tensor: torch.Tensor, weight_tensor: torch.Tensor, n_tuning_params):
        grad_tensor = grad_tensor.to(self.base_Linear.weight.device)
        weight_tensor = weight_tensor.to(self.base_Linear.weight.device)
        
        if n_tuning_params <= 0:
            self.masking = torch.zeros_like(self.base_Linear.weight, dtype=self.base_Linear.weight.dtype, device=self.base_Linear.weight.device)
            return

        if mask_mode == 'gradient':
            importance_score = grad_tensor.abs()
        elif mask_mode == 'gradweight':
            importance_score = (grad_tensor.abs()) / (weight_tensor.abs() + 1e-8)
        else:
            raise ValueError("Invalid mask_mode. Choose from ['gradient', 'gradweight']")

        importance_scores_flat = importance_score.view(-1)

        k = min(n_tuning_params, importance_scores_flat.numel())
        if k > 0:
            threshold = torch.topk(importance_scores_flat, k, largest=True, sorted=False).values[-1]
            print(f'Layer: {self.base_Linear}, n_tuning_params: {n_tuning_params}, Threshold: {threshold:.6f}')
            self.masking = (importance_score >= threshold).float().to(self.base_Linear.weight.device).detach().to(self.base_Linear.weight.dtype)
        else:
            self.masking = torch.zeros_like(self.base_Linear.weight, dtype=self.base_Linear.weight.dtype, device=self.base_Linear.weight.device)

    def forward(self, x: torch.Tensor):
        if self.masking is not None:
            masking = self.masking.to(self.tunable_weight.device)
            masked_weight = (self.tunable_weight * masking).to(self.base_Linear.weight.dtype)
        else:
            masked_weight = self.tunable_weight.to(self.base_Linear.weight.dtype)

        return F.linear(x, self.base_Linear.weight + masked_weight, self.base_Linear.bias)



class EntropyBasedMasking:
    def __init__(self, model, mask_mode, gradients, weights, masking_prob):
        self.model = model
        self.mask_mode = mask_mode
        self.gradients = gradients
        self.weights = weights
        self.masking_prob = masking_prob
        self.linear_module_list = ['q_proj', 'v_proj']
        self.tunable_param_count = 0
        self.total_param_count = sum(p.numel() for p in model.parameters())
        self.ratio_mode = 'naive'    # naive, temp_softmax
        self.temperature = 10
        
        for key, module in self.model.named_modules():
            if any(key.endswith(layer_name) for layer_name in self.linear_module_list) and isinstance(module, nn.Linear):
                parent_module, sub_key = self._get_parent_module_and_key(key)
                setattr(parent_module, sub_key, 
                        EntropyMaskingLinear(base_Linear=module, masking_prob=self.masking_prob))

        for name, param in self.model.named_parameters():
            if "tunable_weight" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.apply_entropy_based_masking()
        
        tunable_percentage = (self.tunable_param_count / self.total_param_count) * 100
        logger.info(f"Tunable Parameters: {self.tunable_param_count} / {self.total_param_count} ({tunable_percentage}%)")

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

    def apply_entropy_based_masking(self):
        entropy_ratios = {}

        for name, module in self.model.named_modules():
            if isinstance(module, EntropyMaskingLinear) and any(layer_name in name for layer_name in self.linear_module_list):
                grad_key = self._normalize_key(name + ".weight")
                weight_key = self._normalize_key(name + ".weight")

                grad_tensor = self.gradients.get(grad_key, None)
                weight_tensor = self.weights.get(weight_key, None)
                if grad_tensor is not None:
                    grad_tensor = grad_tensor.to(self.model.device)
                if weight_tensor is not None:
                    weight_tensor = weight_tensor.to(self.model.device)

                if grad_tensor is not None and weight_tensor is not None:
                    if self.mask_mode == 'gradient':
                        prob = grad_tensor.abs() / grad_tensor.abs().sum()
                        layer_norm = grad_tensor.norm().item()
                    elif self.mask_mode == 'gradweight':
                        ratio = grad_tensor.abs() / (weight_tensor.abs() + 1e-8)
                        prob = ratio / ratio.sum()
                        layer_norm = ratio.norm().item()
                    else:
                        raise ValueError("Invalid mask_mode. Choose from ['gradient', 'gradweight']")

                    entropy = -torch.sum(prob * torch.log(prob + 1e-8)).item()
                    entropy_score = layer_norm / (entropy + 1e-8)
                    
                    if torch.isinf(torch.tensor(layer_norm)) or torch.isnan(torch.tensor(layer_norm)):
                        logger.warning(f"Skipping {name} due to invalid norm value (inf/nan).")
                        continue

                    if torch.isnan(torch.tensor(entropy_score)):
                        logger.warning(f"Skipping {name} due to entropy_score being nan.")
                        continue
                    
                    entropy_ratios[name] = entropy_score
                    print(f'--- name:{name}, layer norm:{layer_norm}, entropy: {entropy}, entropy_score:{entropy_score}')
                else:
                    logger.warning(f"Gradient or weight missing for {name}")

        total_tuning_params = int(self.total_param_count * (1 - self.masking_prob))

        entropy_values = torch.tensor(list(entropy_ratios.values()))
        if self.ratio_mode == 'temp_softmax':
            entropy_values = torch.log1p(entropy_values)
            entropy_values = torch.exp(entropy_values / self.temperature)
            entropy_values[torch.isnan(entropy_values)] = 0  
            entropy_values /= entropy_values.sum() 
        elif self.ratio_mode == 'naive':
            entropy_values /= entropy_values.sum()

        entropy_ratios = dict(zip(entropy_ratios.keys(), entropy_values.tolist()))

        for name, module in self.model.named_modules():
            if name in entropy_ratios:
                layer_ratio = entropy_ratios[name] 
                n_tuning_params = int(total_tuning_params * layer_ratio)
                print(f'--- Layer: {name}, Layer Ratio: {layer_ratio:.6f}, Tuning Params: {n_tuning_params}')

                logger.info(f"Applying entropy-based mask to layer: {name}")
                grad_tensor = self.gradients.get(self._normalize_key(name + ".weight"), None)
                weight_tensor = self.weights.get(self._normalize_key(name + ".weight"), None)

                if grad_tensor is not None and weight_tensor is not None:
                    module.apply_mask(self.mask_mode, grad_tensor, weight_tensor, n_tuning_params)
                    if module.masking is not None:
                        self.tunable_param_count += (module.masking > 0).sum().item()
                else:
                    logger.warning(f"Gradient or weight missing for {name}")
