<div align="center">
<h2>
Templates for 3D Object Pose Estimation Revisited:<br>  Generalization to New objects and Robustness to Occlusions
<p></p>
</h2>

<h3>
<a href="https://nv-nguyen.github.io/" target="_blank"><nobr>Van Nguyen Nguyen</nobr></a> &emsp;
<a href="https://yinlinhu.github.io/" target="_blank"><nobr>Yinlin Hu</nobr></a> &emsp;
<a href="https://youngxiao13.github.io/" target="_blank"><nobr>Yang Xiao</nobr></a> &emsp;
<a href="https://people.epfl.ch/mathieu.salzmann" target="_blank"><nobr>Mathieu Salzmann</nobr></a> &emsp;
<a href="https://vincentlepetit.github.io/" target="_blank"><nobr>Vincent Lepetit</nobr></a>

<p></p>

<a href="https://nv-nguyen.github.io/template-pose/"><img 
src="https://img.shields.io/badge/-Webpage-blue.svg?colorA=333&logo=html5" height=35em></a>
<a href="https://arxiv.org/abs/2203.17234"><img 
src="https://img.shields.io/badge/-Paper-blue.svg?colorA=333&logo=arxiv" height=35em></a>
<a href="https://colab.research.google.com/drive/18Si4X7fcKFHvFuMS-FRVkDyTvlOsr78H?usp=sharing"><img 
src="https://img.shields.io/badge/-Demo-blue.svg?colorA=333&logo=googlecolab" height=35em></a>
<p></p>

<p align="center">
  <img src=./media/qualitative.gif width="80%"/>
</p>

</h3>
</div>

If our project is helpful for your research, please consider citing : 
``` Bash
@inproceedings{nguyen2022template,
    title={Templates for 3D Object Pose Estimation Revisited: Generalization to New objects and Robustness to Occlusions},
    author={Nguyen, Van Nguyen and Hu, Yinlin and Xiao, Yang and Salzmann, Mathieu and Lepetit, Vincent},
    booktitle={Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}}
```
You can also put a star :star:, if the code is useful to you.

If you like this project, check out related works from our group:
- [CNOS: A Strong Baseline for CAD-based Novel Object Segmentation (ICCV 2023 R6D)](https://github.com/nv-nguyen/cnos) 
- [NOPE: Novel Object Pose Estimation from a Single Image (arXiv 2023)](https://github.com/nv-nguyen/nope) 
- [PIZZA: A Powerful Image-only Zero-Shot Zero-CAD Approach to 6DoF Tracking
(3DV 2022)](https://github.com/nv-nguyen/pizza)
- [BOP visualization toolkit](https://github.com/nv-nguyen/bop_viz_kit)

![Teaser image](./media/method.png)

##  Updates (WIP)
We have introduced additional features and updates to the codebase:
- Adding wandb logger: [training loggers](https://api.wandb.ai/links/nv-nguyen/8hkk35s4), [testing loggers](https://wandb.ai/nv-nguyen/template-pose-released/reports/Visualizations-of-template-pose--Vmlldzo0MzY0NDI2?accessToken=hyet783s3ujnbtvmqda71q2nv8haira63f5n23c3fhorb1oe949qmehucplaxfd8)
- Cropping in LINEMOD settings is done with input bounding boxes (there is also predicted in-plane rotation)
- Releasing synthetic templates with Pyrender for faster rendering 
- Releasing ready-to-use universal model pretrained on different datasets of BOP challenge [HomebrewedDB, HOPE, RU-APC, IC-BIN, IC-MI, TUD-L, T-LESS](https://bop.felk.cvut.cz/datasets/)
- Adding code to generate poses (OpenCV coordinate) from icosahedron with Blender
- Parsing with [hydra](https://github.com/facebookresearch/hydra) library, simplifying training_step, testing_step with [pytorch lightning](https://lightning.ai/)
- Path structure (of pretrained models, dataset) is defined as in our recent project [NOPE](https://github.com/nv-nguyen/nope)
<details><summary>Click to expand</summary>

```bash
$ROOT_DIR
    ├── datasets
        ├── linemod 
            ├── models
            ├── test
        ├── tless
        ├── ruapc 
        ├── ...
        ├── templates	
    ├── pretrained
        ├── moco_v2_800ep_pretrain.pth
    ├── results
        ├── experiment1
            ├── wandb
            ├── checkpoint
        ├── experiment2
```

</details>
If you don't want these features, you can use the last version of codebase with the following command:

```
git checkout 50a1087
```

This repository is running with the Weight and Bias logger. Ensure that you update this [user's configuration](https://github.com/nv-nguyen/template-pose/blob/main/configs/user/default.yaml) before conducting any experiments. 

## Installation :construction_worker:

<details><summary>Click to expand</summary>

### 1. Create conda environment
```
conda env create -f environment.yml
conda activate template

# require only for evaluation: pytorch3d 0.7.0
git clone https://github.com/facebookresearch/pytorch3d.git
python -m pip install -e .
```

### 2. Datasets
First, create template poses from icosahedron:
```
blenderproc run src/poses/create_poses.py
```
Next, download and process BOP datasets
```
./src/scripts/download_and_process_datasets.sh
```
There are two options for the final step (rendering synthetic templates from CAD models):

#### Option 1: Download pre-rendered synthetic templates:
```
python -m src.scripts.download_prerendered_templates
```
Optional: This pre-rendered template set can be manually downloaded from [here](https://drive.google.com/drive/folders/1p9eJ8dTxR3rVinvaFxPw5N_3IGSlS2_E?usp=sharing) (12GB).
#### Option 2: Rendering synthetic templates from scratch (this will take around 1 hour with Nvidia V100)

```
./src/scripts/render_pyrender_all.sh
```

<details><summary>Click to expand</summary>

It is important to verify that all the datasets are correctly downloaded and processed. For example, by counting the number of images of each folder:


```
for dir in $ROOT_DIR/datasets/*     
do
    echo ${dir}
    find ${dir} -name "*.png" | wc -l     
done
```

If everything is fine, here are the number of images that you should get:

```bash
├── $ROOT_DIR/datasets
    ├── hb # 55080
    ├── hope # 1968
    ├── icbin # 19016
    ├── icmi # 31512
    ├── lm # 49822
    ├── olm # 4856	
    ├── ruapc #	143486
    ├── tless # 309600
    ├── tudl # 153152
    ├── templates (12GB) # 84102
```
</details>

</details>


 ##  Launch a training  :rocket:

<details><summary>Click to expand</summary>

### 0. (Optional) We use pretrained weight from MoCo v2. You can download it from [here](https://drive.google.com/file/d/1DwlMVdj7rPh3TZ2QfKDU4t4460gofm7i/view?usp=share_link) or run:

```
python -m src.scripts.download_moco_weights
```

If you don't want to use pretrained weights, you can remove the path in [this line](https://drive.google.com/drive/folders/1p9eJ8dTxR3rVinvaFxPw5N_3IGSlS2_E?usp=sharing).
### 1. Training on all BOP datasets except LINEMOD and T-LESS (only objects 19-30)
```
python train.py name_exp=train_all
```

The parsing is done with Hydra library. You can override anything in the configuration by passing arguments. For example:

```
# experiment 1: change batch_size, using data augmentation, update name_exp
python train.py machine.batch_size=2 use_augmentation=True name_exp=train_augmentation

# experiment 2: change batch_size, using data augmentation, update name_exp, update_lr
python train.py machine.batch_size=2 use_augmentation=True model.lr=0.001 name_exp=train_augmentation_lr0.001
```

Please check out this [training loggers](https://api.wandb.ai/links/nv-nguyen/8hkk35s4) to see how the training loss looks like.

</details>

##  Reproduce quantitative results 

Please note that all testing objects are unseen during training!

<details><summary>Click to expand</summary>

### 0. You can download it from [this link](https://drive.google.com/drive/folders/11SQYPrG3pX31Qszf8R13s7Aaa5MO57lb?usp=sharing) or run:

```
python -m src.scripts.download_checkpoint
```

TODO: This is not the final checkpoint. We will update it soon.

### 1. LINEMOD's objects

```
python test_lm.py name_exp=test_lm model.checkpoint_path=$CHECKPOINT_PATH
```

### 2. TLESS's objects

```
python test_tless.py name_exp=test_tless model.checkpoint_path=$CHECKPOINT_PATH
```

Please check out this [testing loggers](https://wandb.ai/nv-nguyen/template-pose-released/reports/Visualizations-of-template-pose--Vmlldzo0MzY0NDI2?accessToken=hyet783s3ujnbtvmqda71q2nv8haira63f5n23c3fhorb1oe949qmehucplaxfd8) to see how the retrieved results looks like.

</details>


## Acknowledgement

The code is adapted from [Nope](https://github.com/nv-nguyen/nope), [Temos](https://github.com/Mathux/Temos), [Unicorn](https://github.com/monniert/unicorn), [PoseContrast](https://github.com/YoungXIAO13/PoseContrast), [CosyPose](https://github.com/ylabbe/cosypose) and [BOP Toolkit](https://github.com/thodan/bop_toolkit). 

The authors thank Martin Sundermeyer, Paul Wohlhart and Shreyas Hampali for their fast reply, feedback!

## Contact
If you have any question, feel free to create an issue or contact the first author at van-nguyen.nguyen@enpc.fr

##  TODO
- Update checkpoints
- Tutorial of training/testing on custom datasets
- Gradio demo with similarity visualization
- Release universal pretrained models