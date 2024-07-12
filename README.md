# Cross Attention Map for Stable Diffusion 3
This codebase is built upon [Attention-map](https://github.com/wooyeolBaek/attention-map)


This repository is for extracting and visualizing attention maps for [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers), compatible with the latest Diffusers code 


## Examples

<!-- <img src="./assets/t2i.png" alt="attn_map">
<img src="./assets/attn_maps.png" alt="attn_map"> -->


<details>
<summary>6_kangaroo</summary>
<div markdown="1">

<img src="./assets/original.png" alt="a beer with a deer's antler">

</div>
</details>


<details>
<summary>10_hoodie</summary>
<div markdown="1">

<img src="./assets/body.png" alt="a beer body">

</div>
</details>


<details>
<summary>13_sunglasses</summary>
<div markdown="1">

<img src="./assets/antler" alt="deer's antler">

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