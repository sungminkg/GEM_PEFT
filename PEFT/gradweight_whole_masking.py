import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)

class GradWeightWholeMasking:
    def __init__(self, model, mask_mode, gradients, weights, masking_prob):
        self.model = model
        self.mask_mode = mask_mode
        self.gradients = gradients
        self.weights = weights
        self.masking_prob = masking_prob
        self.linear_module_list = ['q_proj', 'v_proj']

        self.tunable_param_count = 0
        self.total_param_count = sum(p.numel() for p in model.parameters())

        # Collect all tunable weights and gradients
        all_gradients = []
        all_weights = []
        layer_grad_norms = {}
        layer_weight_norms = {}

        for name, module in self.model.named_modules():
            if any(name.endswith(layer_name) for layer_name in self.linear_module_list) and isinstance(module, nn.Linear):
                grad_key = self._normalize_key(name + ".weight")
                weight_key = self._normalize_key(name + ".weight")

                grad_tensor = self.gradients.get(grad_key, None)
                weight_tensor = self.weights.get(weight_key, None)

                if grad_tensor is not None and weight_tensor is not None:
                    # Scaling Gradient and Weight
                    grad = grad_tensor.abs()
                    grad.clamp_(max=65504)
                    grad = grad / grad.max() * 100
                    
                    weight = weight_tensor.abs()
                    weight.clamp_(max=65504)
                    weight_rms = torch.sqrt(torch.mean(weight**2)) + 1e-5   # RMS scaling
                    weight = weight / weight_rms
                    weight = torch.where(weight < 1e-5, 1e-5, weight)
                    
                    # Calculate L2 norms for the layer
                    layer_grad_norms[name] = grad.norm()
                    layer_weight_norms[name] = weight.norm()
                    all_gradients.append((grad, name))
                    all_weights.append((weight, name))
                else:
                    if grad_tensor is None:
                        logger.warning(f"Gradient missing for {grad_key}")
                    if weight_tensor is None:
                        logger.warning(f"Weight missing for {weight_key}")

        # Combine all gradients and weights into a single tensor
        if all_gradients and all_weights:
            if self.mask_mode == 'gradient':
                ratio = torch.cat([grad.view(-1) for grad, _ in all_gradients])
            elif self.mask_mode == 'gradweight':
                ratio = torch.cat([
                    (grad.view(-1) / (weight.view(-1) + 1e-5))
                    for (grad, _), (weight, _) in zip(all_gradients, all_weights)
                ])
            elif self.mask_mode == 'gradient_ln':
                ratio = torch.cat([
                    (grad.view(-1) / (layer_grad_norms[name] + 1e-5))
                    for grad, name in all_gradients
                ])
            elif self.mask_mode == 'gradweight_ln':
                ratio = torch.cat([
                    ((grad.view(-1) / (layer_grad_norms[name] + 1e-5)) /
                     (weight.view(-1) / (layer_weight_norms[name] + 1e-5)))
                    for (grad, name), (weight, _) in zip(all_gradients, all_weights)
                ])
            else:
                raise ValueError(f"Unsupported mask mode: {self.mask_mode}")

            # Apply global masking
            k = int(self.total_param_count * (1 - self.masking_prob))
            if k > 0:
                threshold = torch.topk(ratio, k, largest=True, sorted=False).values[-1]
                print(f'k: {k} Threshold is:{threshold}\n')
                global_mask = (ratio >= threshold).float()
            else:
                global_mask = torch.zeros_like(ratio)

            # Update tunable weights and count tunable parameters
            start_idx = 0
            for name, module in self.model.named_modules():
                if any(name.endswith(layer_name) for layer_name in self.linear_module_list) and isinstance(module, nn.Linear):
                    weight_key = self._normalize_key(name + ".weight")
                    weight_tensor = self.weights.get(weight_key, None)

                    if weight_tensor is not None:
                        numel = weight_tensor.numel()
                        mask = global_mask[start_idx: start_idx + numel].view(weight_tensor.shape)
                        start_idx += numel

                        # Apply mask to weights
                        masked_weight = weight_tensor * mask
                        self.weights[weight_key] = masked_weight
                        
                        # Apply masking directly to the model's tunable weights
                        if hasattr(module, 'weight'):
                            module.weight.data = masked_weight

                        # Count tunable parameters
                        self.tunable_param_count += mask.sum().item()

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
        return key
