
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Union
from diffusers.utils import logging
from PIL import Image
import torch.nn.functional as F

import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.attention import Attention
import os
from tqdm import tqdm
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.models.attention import JointTransformerBlock
import numpy as np


attn_maps = dict()


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""



class AttnJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.attn_map = None
        self.timestep = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        timestep = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        self.timestep = int(timestep[0])
        # cross attention
        attention_scores = torch.matmul(query, encoder_hidden_states_key_proj.transpose(-2, -1)) / torch.sqrt(torch.tensor(input_ndim, dtype=torch.float32))

        # Apply softmax to get attention weights
        self.attn_map = attention_scores[1].detach().cpu()


        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # import pdb; pdb.set_trace()
        hidden_states = hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states



def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output



def SD3TransformerForward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, timestep=timestep
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)



def JointTranformerForward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor, timestep: torch.LongTensor
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        # import pdb; pdb.set_trace()
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
        cross_attention_kwargs = {
            'timestep': timestep
        }
        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states, **cross_attention_kwargs
        )
        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states



def set_layer_with_name_and_path(model, target_name="attn2", current_path=""):
    a = [layer.__class__.__name__ for name, layer in model.named_children()]
    stop = 1
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        # Replace the forward method of the transformer
        if hasattr(model, 'forward'):
            model.forward = SD3TransformerForward.__get__(model, SD3Transformer2DModel)
            print(f"Replaced forward method in model: SD3Transformer2DModel")
    for name, layer in model.named_children():
        # Check if the current layer is the target layer
        if layer.__class__.__name__ == 'JointTransformerBlock':
            # Replace the forward method of the transformer
            if hasattr(layer, 'forward'):
                layer.forward = JointTranformerForward.__get__(layer, JointTransformerBlock)
                print(f"Replaced forward method in layer: {current_path + '.' + name if current_path else name}")
            # Replace the __call__ method of the attn processor
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'processor'):
                # layer.attn.processor.__call__ = JointAttnProcessorCall
                # print(f"Replaced __call__ method in layer: {current_path + '.' + name if current_path else name}")
                layer.attn.processor = AttnJointAttnProcessor2_0()
                print(f"Replaced processor in layer: {current_path + '.' + name if current_path else name}")
 


 

        new_path = current_path + '.' + name if current_path else name
        set_layer_with_name_and_path(layer, target_name, new_path)



def hook_fn(name,detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            timestep = module.processor.timestep
            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach else module.processor.attn_map
            del module.processor.attn_map

    return forward_hook



def register_cross_attention_hook(transformer):
    # a = list(unet.named_modules())
    for name, module in transformer.named_modules():

        if not name.split('.')[-1].startswith('attn'):
            # only for sd3
            continue

        hook = module.register_forward_hook(hook_fn(name))
    
    return transformer


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        token = tokenizer.decoder[text_input_id.item()]
        tokens.append(token)
    return tokens



def resize_and_save(tokenizer, prompt, timestep=None, path=None, max_height=256, max_width=256, save_path='attn_maps'):
    resized_map = None

    if path is None:
        for path_ in list(attn_maps[timestep].keys()):
            
            value = attn_maps[timestep][path_]
            # value = torch.mean(value,axis=0).squeeze(0)
            vis_seq_len, seq_len = value.shape
            h, w = torch.sqrt(torch.tensor(vis_seq_len)).int(), torch.sqrt(torch.tensor(vis_seq_len)).int()
            value = value.view(h, w, seq_len)
            value = value.permute(2,0,1)

            max_height = max(h, max_height)
            max_width = max(w, max_width)
            value = F.interpolate(
                value.to(dtype=torch.float32).unsqueeze(0),
                size=(max_height, max_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0) # (77,64,64)
            resized_map = resized_map + value if resized_map is not None else value
    else:
        value = attn_maps[timestep][path]
        value = torch.mean(value,axis=0).squeeze(0)
        seq_len, h, w = value.shape
        max_height = max(h, max_height)
        max_width = max(w, max_width)
        value = F.interpolate(
            value.to(dtype=torch.float32).unsqueeze(0),
            size=(max_height, max_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0) # (77,64,64)
        resized_map = value

    # match with tokens
    tokens = prompt2tokens(tokenizer, prompt)
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token

    # init dirs
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + f'/{timestep}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if path is not None:
        save_path = save_path + f'/{path}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
    for i, (token, token_attn_map) in enumerate(zip(tokens, resized_map)):
        if token == bos_token:
            continue
        if token == eos_token:
            break
        token = token.replace('</w>','')
        token = f'{i}_<{token}>.jpg'

        # min-max normalization(for visualization purpose)
        token_attn_map = token_attn_map.numpy()
        normalized_token_attn_map = (token_attn_map - np.min(token_attn_map)) / (np.max(token_attn_map) - np.min(token_attn_map)) * 255
        normalized_token_attn_map = normalized_token_attn_map.astype(np.uint8)

        # save the image
        image = Image.fromarray(normalized_token_attn_map)
        image.save(os.path.join(save_path, token))


# def save_by_timesteps_and_path(tokenizer, prompt, max_height, max_width, save_path='attn_maps_by_timesteps_path'):
#     for timestep in tqdm(attn_maps.keys(),total=len(list(attn_maps.keys()))):
#         for path in attn_maps[timestep].keys():
#             resize_and_save(tokenizer, prompt, timestep, path, max_height=max_height, max_width=max_width, save_path=save_path)

def save_by_timesteps(tokenizer, prompt, max_height, max_width, save_path='attn_maps_by_timesteps'):
    for timestep in tqdm(attn_maps.keys(),total=len(list(attn_maps.keys()))):
        resize_and_save(tokenizer, prompt, timestep, None, max_height=max_height, max_width=max_width, save_path=save_path)