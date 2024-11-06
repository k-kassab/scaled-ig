# Scaled Inverse Graphics: Efficiently Learning Large Sets of 3D Scenes

**Official paper implementation**
> Karim Kassab*, Antoine Schnepf*, Jean-Yves Franceschi, Laurent Caraffa, Flavian Vasile, Jeremie Mary, Andrew Comport, Valérie Gouet-Brunet (* indicates equal contribution)<br>
| [Project Page](https://scaled-ig.github.io) | [Full Paper](https://arxiv.org/abs/2410.23742v1) |<br>

<b>Abstract:</b> *While the field of inverse graphics has been witnessing continuous growth, techniques devised thus far predominantly focus on learning individual scene representations. In contrast, learning large sets of scenes has been a considerable bottleneck in NeRF developments, as repeatedly applying inverse graphics on a sequence of scenes, though essential for various applications, remains largely prohibitive in terms of resource costs. We introduce a framework termed "scaled inverse graphics", aimed at efficiently learning large sets of scene representations, and propose a novel method to this end. It operates in two stages: (i) training a compression model on a subset of scenes, then (ii) training NeRF models on the resulting smaller representations, thereby reducing the optimization space per new scene. In practice, we compact the representation of scenes by learning NeRFs in a latent space to reduce the image resolution, and sharing information across scenes to reduce NeRF representation complexity. We experimentally show that our method presents both the lowest training time and memory footprint in scaled inverse graphics compared to other methods applied independently on each scene.*

![LatentScenes](assets/ls_videos.gif)

## Setup
In this section we detail how prepare the environment for training numerous scenes.

### Environment 
Our code has been tested on:
- Linux (Debian)
- Python 3.11.9
- Pytorch 2.0.1
- CUDA 11.8
- `L4` and `A100` NVIDIA GPUs


You can use Anaconda to create the environment:
```
conda create --name igae -y python=3.11.9
conda activate igae
```
Then, you can install pytorch with Cuda 11.8 using the following command:
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 --upgrade
```
_You may have to adapt the cuda version according to your hardware, we recommend using CUDA >= 11.8_

To install the remaining requirements, execute:
```
pip install -r requirements.txt
```

## Usage

### Download data
(coming soon)
Download and untar the data (about 26 GB).

### Define data directory
You must specify the path to the scaled-ig-data by defining the environment variable DATA_DIR
```
export DATA_DIR=".../scaled-ig-data"
```
or by changing the variable ``DATA_DIR`` in igae/datasets/dataset.py .

### Training numerous scenes
This section illustrates how to learn numerous cars from shapenet.
#### Stage 1
To launch stage 1, run:
```
bash run_stage1.sh stage1/cars.yaml
```
#### Stage 2
To launch stage 2, run:
```
bash run_stage2.sh stage2/cars.yaml
```

## Visualization / evaluation
We visualize and evaluate our method using [wandb](https://wandb.ai/site). 
You can get quickstarted [here](https://docs.wandb.ai/quickstart).

## Notice 
This work is closely related to our paper [Bringing NeRFs to the Latent Space:
Inverse Graphics Autoencoder](https://ig-ae.github.io), whose [implementation](https://github.com/k-kassab/igae) encompasses functionalities for both projects.
Feel free to explore our other work if it piques your interest.

## A Note on License
This code is open-source. We share most of it under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
However, parts of the code in [igae](igae) are distributed under a more restrictive license.
Refer to the igae [README](igae/README.md) for more details.

## Citation

If you find this research project useful, please consider citing our work:
```
@article{scaled-ig,
        title={{Scaled Inverse Graphics: Efficiently Learning Large Sets of 3D Scenes}}, 
        author={Karim Kassab and Antoine Schnepf and Jean-Yves Franceschi and Laurent Caraffa and Flavian Vasile and Jeremie Mary and Andrew Comport and Valérie Gouet-Brunet},
        journal={arXiv preprint arXiv:2410.23742},
        year={2024}
}
```