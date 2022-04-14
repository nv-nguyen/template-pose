Please change [this line](https://github.com/nv-nguyen/template-pose/blob/main/config.json#L2) to set the location of $DATA folder. 

The datasets can be also downloaded manually with the provided links in [config.json](https://github.com/nv-nguyen/template-pose/blob/main/config.json). We recommend the following structure to keep the processing pipeline simple:
### Final structure of folder $DATA

```bash
$DATA
    ├── linemod 
        ├── models
        ├── opencv_pose
        ├── LINEMOD
        ├── occlusionLINEMOD
    ├── tless
        ├── models
        ├── opencv_pose
        ├── train
        └── test
    ├── templates	
        ├── linemod
            ├── train
            ├── test
        ├── tless
    ├── SUN397
    ├── LINEMOD.json # query-template pairwise for LINEMOD
    ├── occlusionLINEMOD.json # query-template pairwise for Occlusion-LINEMOD
    ├── tless_train.json # query-template pairwise for training split of T-LESS
    ├── tless_test.json # query-template pairwise for testing split of T-LESS
    ├── crop_image64 # pre-cropped images for LINEMOD
    ├── crop_image224 # pre-cropped images for LINEMOD
    └── zip	# tmp dir
```
There are 5 main steps if you process data from scratch with [download_and_process_from_scratch.sh](https://github.com/nv-nguyen/template-pose/blob/main/data/download_and_process_from_scratch.sh):
### 1. Download datasets:
```bash
python -m data.download --dataset all
```
### 2. Process ground-truth poses
Convert the coordinate system to [BOP datasets format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md) and save GT poses of each object separately:
```bash
python -m data.process_gt_linemod
python -m data.process_gt_tless
```
### 3. Render or download templates
To render templates (which takes around ~ one day on V100 16GB):
```bash
python -m data.render_templates --dataset linemod --disable_output --num_workers 4
python -m data.render_templates --dataset tless --disable_output --num_workers 4
```
Or to use our pre-rendered templates, please manually download from this [link](https://drive.google.com/file/d/1agQ5ERrFR3RjdlDf-tEDSZvLYamUsftv/view?usp=sharing) and put it into $DATA/templates
### 4. Crop images (only for LINEMOD)
Crop images of LINEMOD, OcclusionLINEMOD and its templates with GT poses as done in [this paper](https://arxiv.org/abs/1502.05908.pdf)
```bash
python -m data.crop_image_linemod
```
### 5. Compute neighbors with GT poses
This step outputs a dataframe which associate a template for each query images. This dataframe will be used in training loops.
```bash
python -m data.create_dataframe_linemod
python -m data.create_dataframe_tless --split train
python -m data.create_dataframe_tless --split test
```
