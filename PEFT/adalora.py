import logging
import torch
from torch import nn
from torch.nn import functional as F
import math

logger = logging.getLogger(__name__)


def find_module(root_module: nn.Module, key: str):
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class AdaLoRALinear(nn.Module):
    def __init__(
            self,
            base_Linear: nn.Linear,
            in_features: int,
            out_features: int,
            r: int = 8,
            lora_alpha: float = 16,
            total_step: int = 1000,
            target_rank: int = 6,
            init_warmup: int = 200,
            final_warmup: int = 900,
            mask_interval: int = 10,
            beta1: float = 0.85
    ):
        super().__init__()
        self.base_Linear = base_Linear
        self.r = r
        self.scaling = lora_alpha / r
        self.total_step = total_step
        self.target_rank = target_rank
        self.init_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1

        self.global_step = 0
        self.importance_score = torch.zeros(r)

        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base_Linear.weight.requires_grad = False
        if self.base_Linear.bias is not None:
            self.base_Linear.bias.requires_grad = False

        self.mask = torch.ones(r, dtype=torch.bool)  # 처음에는 전체 사용

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.base_Linear.weight, self.base_Linear.bias)
        if self.r > 0:
            A = self.lora_A[self.mask]
            B = self.lora_B[:, self.mask]
            result += (x @ A.T @ B.T) * self.scaling
        return result

    def update_importance_score(self):
        if self.lora_A.grad is not None and self.lora_B.grad is not None:
            with torch.no_grad():
                score = (
                    (self.lora_A.grad * self.lora_A).sum(dim=1).abs()
                    + (self.lora_B.grad * self.lora_B).sum(dim=0).abs()
                )
                self.importance_score = self.beta1 * self.importance_score + (1 - self.beta1) * score

    def adjust_rank(self):
        if self.global_step % self.mask_interval == 0:
            if self.global_step < self.init_warmup:
                target_r = self.r
            elif self.global_step > self.final_warmup:
                target_r = self.target_rank
            else:
                progress = (self.global_step - self.init_warmup) / (self.final_warmup - self.init_warmup)
                target_r = int(self.r - (self.r - self.target_rank) * progress)

            topk = torch.topk(self.importance_score, target_r)
            new_mask = torch.zeros_like(self.mask)
            new_mask[topk.indices] = True
            self.mask = new_mask

        self.global_step += 1


class AdaLoRA:
    def __init__(self, model, r, alpha, **kwargs):
        self.model = model
        self.hidden_dim = model.config.hidden_size

        if model.config.model_type == "opt":
            attention_name = "attn"
        elif model.config.model_type == "roberta":
            attention_name = "attention"
        elif model.config.model_type in ["llama", "phi"]:
            attention_name = "self_attn"
        else:
            raise NotImplementedError

        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                logger.info(f"Inject AdaLoRA to: {key}")
                _, _, attn = find_module(model, key)
                device = attn.q_proj.weight.device
                attn.q_proj = AdaLoRALinear(
                    base_Linear=attn.q_proj,
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                    r=r,
                    lora_alpha=alpha,
                    **kwargs
                ).to(device)
                attn.v_proj = AdaLoRALinear(
                    base_Linear=attn.v_proj,
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                    r=r,
                    lora_alpha=alpha,
                    **kwargs
                ).to(device)

        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True