# Cross Attention Map
This codebase is built upon [Attention-map](https://github.com/wooyeolBaek/attention-map)


This repository is for extracting and visualizing attention maps for [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers), compatible with the latest Diffusers code 


## Examples

<!-- <img src="./assets/t2i.png" alt="attn_map">
<img src="./assets/attn_maps.png" alt="attn_map"> -->
<img src="./assets/hf_spaces.png" alt="hf_spaces">

<details>
<summary>6_kangaroo</summary>
<div markdown="1">

<img src="./assets/6_<kangaroo>.png" alt="6_kangaroo">

</div>
</details>


<details>
<summary>10_hoodie</summary>
<div markdown="1">

<img src="./assets/10_<hoodie>.png" alt="10_hoodie">

</div>
</details>


<details>
<summary>13_sunglasses</summary>
<div markdown="1">

<img src="./assets/13_<sunglasses>.png" alt="13_sunglasses">

</div>
</details>


## Initialize
```shell

conda create -n attn python=3.11 -y
conda activate attn

pip install -r requirements.txt
```

## Visualize
Visualize Cross Attention Map for Text-to-Image
```shell
python main.py
```

## How to use