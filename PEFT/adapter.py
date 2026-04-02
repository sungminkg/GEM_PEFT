import logging
import re

from torch import nn

logger = logging.getLogger(__name__)


def decoder_layer_forward_hook(self, *args, **kwargs):
    hidden_states = args[0]

    if len(args) >= 6 or any(k in kwargs for k in ["attention_mask", "layer_head_mask", "past_key_value"]):
        attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        layer_head_mask = kwargs.get("layer_head_mask", args[2] if len(args) > 2 else None)
        past_key_value = kwargs.get("past_key_value", args[3] if len(args) > 3 else None)
        output_attentions = kwargs.get("output_attentions", args[4] if len(args) > 4 else False)
        use_cache = kwargs.get("use_cache", args[5] if len(args) > 5 else False)
    else:
        raise ValueError("decoder_layer_forward_hook: wrong number of arguments")

    device = self.self_attn_device
    hidden_states = hidden_states.to(device)
    residual = hidden_states

    if self.peft_model_type == "opt":
        self_attn_norm = self.self_attn_layer_norm
        final_norm = self.final_layer_norm
        pre_norm = self.do_layer_norm_before
    elif self.peft_model_type == "llama":
        self_attn_norm = self.self_attn_layer_norm
        final_norm = self.final_layer_norm
        pre_norm = True
    elif self.peft_model_type in ["phi", "phi-2", "microsoft_phi"]:
        self_attn_norm = self.input_layernorm
        final_norm = getattr(self, 'post_attention_layernorm', getattr(self, 'ffn_layernorm', None))
        pre_norm = True
    else:
        raise ValueError("Unknown PEFT model type")

    if pre_norm and self_attn_norm is not None:
        hidden_states = self_attn_norm(hidden_states)

    if self.peft_model_type in ["phi", "phi-2", "microsoft_phi"]:
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **{k: v for k, v in kwargs.items() if k in ["position_ids"]}
        )
        if len(attn_outputs) == 2:
            hidden_states, present_key_value = attn_outputs
            self_attn_weights = None
        elif len(attn_outputs) == 3:
            hidden_states, self_attn_weights, present_key_value = attn_outputs
        else:
            raise ValueError("Unexpected number of outputs from self.self_attn for Phi model")
    else:
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

    if hasattr(self, "dropout") and hasattr(self.dropout, 'p'):
        dropout_p = self.dropout.p
    elif hasattr(self, "resid_dropout") and hasattr(self.resid_dropout, 'p'):
        dropout_p = self.resid_dropout.p
    elif hasattr(self, "config") and hasattr(self.config, "hidden_dropout"):
        dropout_p = self.config.hidden_dropout
    else:
        dropout_p = 0.0

    hidden_states = self.adapter1(hidden_states) + hidden_states
    hidden_states = nn.functional.dropout(hidden_states, p=dropout_p, training=self.training)
    hidden_states = residual + hidden_states

    if not pre_norm and self_attn_norm is not None:
        hidden_states = self_attn_norm(hidden_states)

    hidden_states_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual = hidden_states

    if pre_norm and final_norm is not None:
        hidden_states = final_norm(hidden_states)

    # Handle the model-specific feed-forward block.
    if hasattr(self, 'ffn'):  # Phi-style block
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.adapter2(hidden_states) + hidden_states
    elif hasattr(self, 'mlp'):  # Some backbones expose the FFN as `mlp`
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.adapter2(hidden_states) + hidden_states
    else:  # OPT / LLaMA-style block
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.adapter2(hidden_states) + hidden_states

    hidden_states = nn.functional.dropout(hidden_states, p=dropout_p, training=self.training)
    hidden_states = (residual + hidden_states).view(hidden_states_shape)

    if not pre_norm and final_norm is not None:
        hidden_states = final_norm(hidden_states)

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


class Adapter:
    def __init__(self, model, r, act_type="relu"):
        self.model = model
        self.r = r

        model_type = model.config.model_type.lower()
        assert model_type in ["opt", "llama", "phi", "phi-2", "microsoft_phi"]

        pattern = r'^model\.decoder\.layers\.\d+$' if model_type == "opt" else r'^model\.layers\.\d+$'
        act_fn_dict = {
            "None": nn.Identity(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh()
        }
        act_fn = act_fn_dict[act_type]

        # Record injected adapter modules and register them after traversal.
        adapter_modules_to_register = []

        for name, module in model.named_modules():
            if re.match(pattern, name):
                if model_type == "opt":
                    out_proj_module = module.self_attn.out_proj
                elif model_type == "llama":
                    out_proj_module = module.self_attn.o_proj
                elif model_type in ["phi", "phi-2", "microsoft_phi"]:
                    out_proj_module = module.self_attn.dense
                else:
                    raise ValueError("Unsupported model type.")

                device = out_proj_module.weight.device
                embed_dim = out_proj_module.in_features

                logger.info(f"Inject adapter to {name}")
                module.peft_model_type = model_type

                module.adapter1 = nn.Sequential(
                    nn.Linear(embed_dim, r, device=device, dtype=out_proj_module.weight.dtype),
                    act_fn,
                    nn.Linear(r, embed_dim, device=device, dtype=out_proj_module.weight.dtype)
                )
                module.adapter2 = nn.Sequential(
                    nn.Linear(embed_dim, r, device=device, dtype=out_proj_module.weight.dtype),
                    act_fn,
                    nn.Linear(r, embed_dim, device=device, dtype=out_proj_module.weight.dtype)
                )

                nn.init.zeros_(module.adapter1[2].weight)
                nn.init.zeros_(module.adapter1[2].bias)
                nn.init.zeros_(module.adapter2[2].weight)
                nn.init.zeros_(module.adapter2[2].bias)

                module.self_attn_device = device
                module.forward = decoder_layer_forward_hook.__get__(module, type(module))

                adapter_modules_to_register.append((name, module))
        
        for name, param in model.named_parameters():
            if "adapter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Register adapter modules after named_modules traversal to avoid mutation issues.
        for name, module in adapter_modules_to_register:
            setattr(self.model, f"{name.replace('.', '_')}_adapter1", module.adapter1)
            setattr(self.model, f"{name.replace('.', '_')}_adapter2", module.adapter2)
