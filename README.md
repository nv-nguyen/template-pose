## template-pose (CVPR 2022) <br><sub>Official PyTorch implementation </sub>

![Teaser image](./media/method.png)

**Templates for 3D Object Pose Estimation Revisited: Generalization to New objects and Robustness to Occlusions**<br>
[Van Nguyen Nguyen](https://nv-nguyen.github.io/), 
[Yinlin Hu](https://yinlinhu.github.io/), 
[Yang Xiao](https://youngxiao13.github.io/), 
[Mathieu Salzmann](https://people.epfl.ch/mathieu.salzmann) and 
[Vincent Lepetit](https://vincentlepetit.github.io/) <br>
**[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Nguyen_Templates_for_3D_Object_Pose_Estimation_Revisited_Generalization_to_New_CVPR_2022_paper.pdf)
, [Project Page](https://nv-nguyen.github.io/template-pose/)**

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
- [NOPE: Novel Object Pose Estimation from a Single Image (arXiv 2023)](https://github.com/nv-nguyen/nope) 
- [PIZZA: A Powerful Image-only Zero-Shot Zero-CAD Approach to 6DoF Tracking
(3DV 2022)](https://github.com/nv-nguyen/pizza)
- [BOP visualization toolkit](https://github.com/nv-nguyen/bop_viz_kit)


##  Updates (WIP)
We have introduced additional features and updates to the codebase:
- Releasing synthetic templates with Pyrender for faster rendering 
- Releasing ready-to-use universal model pretrained on different datasets of BOP challenge [Linemod, HomebrewedDB, HOPE, RU-APC, IC-BIN, IC-MI, TUD-L, T-LESS](https://bop.felk.cvut.cz/datasets/)
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
./src/scripts/render_all.sh
```)
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

### 0. (Optional) We use pretrained weight from MoCo v2. You can download it from [here]() or run:

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

</details>

## Acknowledgement

The code is adapted from [Nope](https://github.com/nv-nguyen/nope), [Temos](https://github.com/Mathux/Temos), [PoseContrast](https://github.com/YoungXIAO13/PoseContrast), [CosyPose](https://github.com/ylabbe/cosypose) and [BOP Toolkit](https://github.com/thodan/bop_toolkit). 

The authors thank Martin Sundermeyer, Paul Wohlhart and Shreyas Hampali for their fast reply, feedback!

## Contact
If you have any question, feel free to create an issue or contact the first author at van-nguyen.nguyen@enpc.fr

##  TODO
- Tutorial of training/testing on custom datasets
- Gradio demo with similarity visualization
- Release universal pretrained models