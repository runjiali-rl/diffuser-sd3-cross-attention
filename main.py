from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
import PIL.Image
import numpy as np
from tqdm import tqdm
import torch
import inspect
from typing import List, Optional, Union
import time
import argparse
import os
from cross_attention import set_layer_with_name_and_path, save_by_timesteps, register_cross_attention_hook


# set random seed
torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using a pretrained diffusion model.")
    parser.add_argument(
        "--model",
        type=str,
        default="stable_diffusion_3",
        help="The model to use for generating images.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a human in a japanese style village.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="dog, unrealistic, cartoon",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/homes/55/runjia/scratch/diffusion_model_weights",
        help="The directory to cache the model weights in.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="The number of images to generate per prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="The number of diffusion steps used when generating samples with a pre-trained model.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="The interval at which to display the generated samples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use for inference.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./2D_experiments/generated_images",
        help="The directory to save the generated images to.",
    )

    parser.add_argument("--guidance_scale", type=float, default=7)
    parser.add_argument("--guidance_rescale", type=float, default=0)

    return parser.parse_args()

def display_sample(image, i):
    if isinstance(image, PIL.Image.Image):
        image_pil = image
    else:
        image = image.permute(0, 2, 3, 1)
        image = image.cpu().detach().numpy()
        image_processed = (image * 255).clip(0, 255)
        image_processed = image_processed.astype(np.uint8)

        image_pil = PIL.Image.fromarray(image_processed[0])
    if not os.path.exists("./2D_experiments/generated_images"):
        os.makedirs("./2D_experiments/generated_images")
    image_pil.save(f"./2D_experiments/generated_images/sample_{i}.png")


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps





def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def predict_noise_residual(model,
                           latent_model_input,
                           t,
                           prompt_embeds,
                           timestep_cond,
                           guidance_scale,
                           guidance_rescale,
                           pooled_prompt_embeds=None,
                           model_name="stable_diffusion_2"):
    """
    Predicts the noise residual and performs guidance.

    Args:
        model: The model used to predict the noise residual.
        latent_model_input: The input to the model.
        t: The current timestep.
        prompt_embeds: The prompt embeddings.
        timestep_cond: The timestep condition.
        guidance_scale: The scale for classifier-free guidance.
        guidance_rescale: The rescale value for guidance.

    Returns:
        torch.Tensor: The predicted noise residual after guidance.
    """
    # Predict the noise residual
    if model_name == "stable_diffusion_2":
        noise_pred = model.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=None,
            # added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    elif model_name == "stable_diffusion_3":
        assert pooled_prompt_embeds is not None, "pooled_prompt_embeds must be provided for stable_diffusion_3"
        noise_pred = model.transformer(
            hidden_states=latent_model_input,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            # added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

    # Perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    if guidance_rescale > 0.0:
        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

    return noise_pred




if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    if model_name == "stable_diffusion_2":
        repo_id = "stabilityai/stable-diffusion-2-1-base"
    elif model_name == "stable_diffusion_3":
        repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"


    model = StableDiffusion3Pipeline.from_pretrained(repo_id,
                                            use_safetensors=True,
                                            torch_dtype=torch.float16,
                                            cache_dir=args.cache_dir,)
    




    set_layer_with_name_and_path(model.transformer)
    register_cross_attention_hook(model.transformer)


    model = model.to("cuda")
    model.enable_model_cpu_offload()


    prompt = args.prompt
    negative_prompt = args.negative_prompt

    part_scale = 1


    if repo_id == "stabilityai/stable-diffusion-2-1-base":
        height = model.unet.config.sample_size * model.vae_scale_factor
        width = model.unet.config.sample_size * model.vae_scale_factor
    elif repo_id == "stabilityai/stable-diffusion-3-medium-diffusers":
        height = model.default_sample_size * model.vae_scale_factor
        width = model.default_sample_size * model.vae_scale_factor
    elif repo_id == "DeepFloyd/IF-I-XL-v1.0":
        height = model.unet.config.sample_size 
        width = model.unet.config.sample_size


    # repo_id = "google/ddpm-cat-256"

    guidance_scale = args.guidance_scale
    guidance_rescale = args.guidance_rescale
    do_classifier_free_guidance = True
    num_images_per_prompt = args.num_images_per_prompt

    num_inference_steps = args.num_inference_steps
    interval = args.interval
    device = args.device
    save_dir= args.save_dir



    print("encoding text prompts")


    if model_name == "stable_diffusion_2":
        prompt_embed, negative_prompt_embed = model.encode_prompt(
            prompt,
            model.device,
            num_images_per_prompt,
            True,
            negative_prompt,
        )
        prompt_embed = torch.cat([negative_prompt_embed, prompt_embed])
        pooled_prompt_embed = None

    elif model_name == "stable_diffusion_3":
        (   prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = model.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            num_images_per_prompt = num_images_per_prompt,
            do_classifier_free_guidance=True,
            max_sequence_length=256,
        )
        prompt_embed = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embed = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        stop = 1

    

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        model.scheduler, num_inference_steps, device
    )

    # 5. Prepare latent variables
    if model_name == "stable_diffusion_2":
        num_channels_latents = model.unet.config.in_channels
    elif model_name == "stable_diffusion_3":
        num_channels_latents = model.transformer.config.in_channels
    latents = model.prepare_latents(
        1 * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embed.dtype,
        device,
        generator=None,
        latents=None,
    )

  
    if model_name == "stable_diffusion_2":
          # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = model.prepare_extra_step_kwargs(None, 0)

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if model.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(model.guidance_scale - 1).repeat(1 * num_images_per_prompt)
            timestep_cond = model.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=model.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * model.scheduler.order
        model._num_timesteps = len(timesteps)
    elif model_name == "stable_diffusion_3":
        num_warmup_steps = max(len(timesteps) - num_inference_steps * model.scheduler.order, 0)
        model._num_timesteps = len(timesteps)
        stop = 1

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) #if model.do_classifier_free_guidance else latents
            if model_name == "stable_diffusion_2":
                latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
                time_step = t
            elif model_name == "stable_diffusion_3":
                time_step = t.expand(latent_model_input.shape[0])
                timestep_cond = None
           

            noise_pred = predict_noise_residual(model,
                                            latent_model_input,
                                            time_step,
                                            prompt_embed,
                                            timestep_cond,
                                            guidance_scale,
                                            guidance_rescale,
                                            pooled_prompt_embed,
                                            model_name=model_name,)
    

            # compute the previous noisy sample x_t -> x_t-1
            if model_name == "stable_diffusion_2":
                latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            elif model_name == "stable_diffusion_3":
                latents = model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
      
            if i % interval == 0:
                if model_name == "stable_diffusion_3":
                    saved_latents = (latents / model.vae.config.scaling_factor) + model.vae.config.shift_factor
                    image = model.vae.decode(saved_latents, return_dict=False)[0]
                    image = model.image_processor.postprocess(image, output_type='pil')[0]
                else:
                    image = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False, generator=None)[0]
                display_sample(image, i)


    save_by_timesteps(model.tokenizer, prompt, height, width)




