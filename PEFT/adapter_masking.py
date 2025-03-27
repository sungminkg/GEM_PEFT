import logging
import torch

logger = logging.getLogger(__name__)

class AdapterMasking:
    def __init__(self, model, gradients, weights, masking_prob):
        self.model = model
        self.gradients = gradients
        self.weights = weights
        self.masking_prob = masking_prob
        self.tunable_weights = self._get_tunable_weights()
        self.masking = []

    def _get_tunable_weights(self):
        tunable_weights = []
        for name, param in self.model.named_parameters():
            if "adapter" in name:
                tunable_weights.append((name, param))
        return tunable_weights

    def _normalize_key(self, key):
        if ".tunable_weight" in key:
            key = key.replace(".tunable_weight", ".weight")
        return key

    def apply_adapter_masking(self):
        self.masking = []
        for name, tunable_weight in self.tunable_weights:
            grad_key = self._normalize_key(name)
            weight_key = self._normalize_key(name)

            # Extract gradient and weight tensors
            grad_tensor = self.gradients.get(grad_key, None)
            weight_tensor = self.weights.get(weight_key, None)
            print('----------------------------')
            print(grad_tensor)
            print('----------------------------')
            print(weight_tensor)
            print('----------------------------')

            if grad_tensor is not None and weight_tensor is not None:
                ratio = grad_tensor.abs()
                # ratio = (grad_tensor / (weight_tensor + 1e-12)).abs()  # Use gradient-to-weight ratio
                k = int(tunable_weight.numel() * (1 - self.masking_prob))
                if k > 0:
                    threshold = torch.topk(ratio.view(-1), k, largest=True, sorted=False).values[-1]
                    mask = (ratio >= threshold).float()
                else:
                    mask = torch.zeros_like(tunable_weight)
            else:
                if grad_tensor is None:
                    logger.warning(f"Gradient missing for {grad_key}")
                if weight_tensor is None:
                    logger.warning(f"Weight missing for {weight_key}")
                mask = torch.zeros_like(tunable_weight)

            # Apply the mask in-place
            with torch.no_grad():
                tunable_weight.mul_(mask)
            self.masking.append(mask)


# import torch

# class AdapterMasking:
#     def __init__(self, model, gradients, masking_prob):
#         self.model = model
#         self.gradients = gradients
#         self.masking_prob = masking_prob
#         self.tunable_weights = self._get_tunable_weights()
#         self.masking = []

#     def _get_tunable_weights(self):
#         tunable_weights = []
#         for name, param in self.model.named_parameters():
#             if "adapter" in name and param.requires_grad:
#                 tunable_weights.append((name, param))
#         return tunable_weights

#     def apply_adapter_masking(self):
#         self.masking = []
#         for name, tunable_weight in self.tunable_weights:
#             grad_tensor = self.gradients.get(name, None)
#             if grad_tensor is not None:
#                 grad_abs = grad_tensor.abs()
#                 weight_abs = tunable_weight.abs()
#                 print(weight_abs)
#                 #ratio = grad_abs / (weight_abs + 1e-12)
#                 ratio = grad_abs
#                 k = int(tunable_weight.numel() * (1 - self.masking_prob))
#                 if k > 0:
#                     threshold = torch.topk(ratio.view(-1), k, largest=True, sorted=False).values[-1]
#                     mask = (ratio >= threshold).float()
#                 else:
#                     mask = torch.zeros_like(tunable_weight)
#             else:
#                 mask = torch.zeros_like(tunable_weight)

#             # Apply the mask in-place
#             with torch.no_grad():
#                 tunable_weight.mul_(mask)
#             self.masking.append(mask)
