import logging
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GradWeightSelectMaskingLinear(nn.Module):
    def __init__(self, base_Linear: nn.Linear, masking_prob: float = 0.0):
        super().__init__()
        self.base_Linear = base_Linear
        self.masking_prob = masking_prob
        self.masking = None  # Mask will be dynamically calculated
        self.tunable_weight = nn.Parameter(self.base_Linear.weight.clone().detach())  # Clone the original weight

    def apply_mask(self, mask_mode, grad_tensor: torch.Tensor, weight_tensor: torch.Tensor, n_tuning_params): 
        if n_tuning_params <= 0:
            self.masking = torch.zeros_like(self.base_Linear.weight, dtype=self.base_Linear.weight.dtype, device=self.base_Linear.weight.device)
            return

        importance_score = grad_tensor / (weight_tensor + 1e-8)  
        importance_scores_flat = importance_score.view(-1)

        k = min(n_tuning_params, importance_scores_flat.numel()) 
        if k > 0:
            threshold = torch.topk(importance_scores_flat, k, largest=True, sorted=False).values[-1]
            print(f'Layer: {self.base_Linear}, n_tuning_params: {n_tuning_params}, Threshold: {threshold:.6f}')

            self.masking = (importance_score >= threshold).float().to(self.base_Linear.weight.device).detach().to(self.base_Linear.weight.dtype)
        else:
            self.masking = torch.zeros_like(self.base_Linear.weight, dtype=self.base_Linear.weight.dtype, device=self.base_Linear.weight.device)
        
    def forward(self, x: torch.Tensor):
        masked_weight = (self.tunable_weight * self.masking if self.masking is not None else self.tunable_weight).to(self.base_Linear.weight.dtype)
        return F.linear(x, self.base_Linear.weight + masked_weight, self.base_Linear.bias)


class GradWeightSelectMasking:
    def __init__(self, model, mask_mode, gradients, weights, masking_prob):
        self.model = model
        self.mask_mode = mask_mode
        self.gradients = gradients
        self.weights = weights
        self.masking_prob = masking_prob
        self.linear_module_list = ['q_proj', 'v_proj']
        self.tunable_param_count = 0
        self.total_param_count = sum(p.numel() for p in model.parameters())
        self.ratio_mode = 'temp_softmax'    # naive, temp_softmax
        self.temperature = 5

        # Replace specific modules with GradWeightMaskingLinear
        for key, module in self.model.named_modules():
            if any(key.endswith(layer_name) for layer_name in self.linear_module_list) and isinstance(module, nn.Linear):
                parent_module, sub_key = self._get_parent_module_and_key(key)
                setattr(parent_module, sub_key, 
                        GradWeightSelectMaskingLinear(base_Linear=module, masking_prob=self.masking_prob))

        # Update requires_grad for tunable weights
        for name, param in self.model.named_parameters():
            if "tunable_weight" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        # Apply masks and count tunable parameters
        self.apply_grad_weight_select_masking()

        # Calculate and log the percentage of tunable parameters
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

    def apply_grad_weight_select_masking(self):
        layerwise_ratios = {}

        for name, module in self.model.named_modules():
            if isinstance(module, GradWeightSelectMaskingLinear) and any(layer_name in name for layer_name in self.linear_module_list):
                grad_tensor = self.gradients.get(name + ".weight", None)
                weight_tensor = self.weights.get(name + ".weight", None)

                if grad_tensor is not None and weight_tensor is not None:
                    if  torch.isnan(grad_tensor).any() or torch.isinf(grad_tensor).any():
                        logger.warning(f"Warning: Gradient contains NaN or Inf in {name}. Setting grad_norm to 0.")
                        layerwise_ratio = 0
                    else:
                        layerwise_ratio = (grad_tensor / (weight_tensor + 1e-8)).abs().norm().item()
                        # layerwise_ratio = grad_tensor.abs().norm().item() / weight_tensor.abs().norm().item()
                    print(f'--- name: {name} layerwise ratio: {layerwise_ratio}')
                    layerwise_ratios[name] = layerwise_ratio

        total_tuning_params = sum(p.numel() for p in self.model.parameters()) * (1 - self.masking_prob)
        
        layerwise_values = torch.tensor(list(layerwise_ratios.values()))
        if self.ratio_mode == 'naive':
            layerwise_values /= layerwise_values.sum()
            layerwise_ratios = dict(zip(layerwise_ratios.keys(), layerwise_values.tolist()))
        elif self.ratio_mode == 'temp_softmax':
            layerwise_values = torch.log1p(layerwise_values)
            layerwise_values = torch.exp(layerwise_values / self.temperature)
            layerwise_values[torch.isnan(layerwise_values)] = 0  
            layerwise_values /= layerwise_values.sum() 
            layerwise_ratios = dict(zip(layerwise_ratios.keys(), layerwise_values.tolist()))

        for name, module in self.model.named_modules():
            if name in layerwise_ratios:
                layer_ratio = layerwise_ratios[name]
                if torch.isnan(torch.tensor(layer_ratio)):
                    layer_ratio = 0.0 
                n_tuning_params = int(total_tuning_params * layer_ratio) 

                grad_key = self._normalize_key(name + ".weight")
                weight_key = self._normalize_key(name + ".weight")
                grad_tensor = self.gradients.get(grad_key, None)
                weight_tensor = self.weights.get(weight_key, None)

                if grad_tensor is not None and weight_tensor is not None:
                    logger.info(f"\n--------Applying mask to layer: {name}------------")
                    print(f"Layer: {name}, Layer Ratio: {layer_ratio:.4f}, Tuning Params: {n_tuning_params}")

                    grad = grad_tensor.abs()
                    weight = weight_tensor.abs()

                    module.apply_mask(self.mask_mode, grad, weight, n_tuning_params)
                    
                    # Count tunable parameters after mask application
                    if module.masking is not None:
                        self.tunable_param_count += (module.masking > 0).sum().item()
                else:
                    if grad_tensor is None:
                        logger.warning(f"Gradient missing for {grad_key}")
                    if weight_tensor is None:
                        logger.warning(f"Weight missing for {weight_key}")
