[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg) 
<!--- ![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg) --->
# Repository for my Master's Thesis 'Visuallocalization under challenging conditions'
<b> based on Nvidia's 'Geometry-Aware Learning of Maps for Camera Localization' </b>

## Credits
This repository is a fork of [NVIDIA's mapnet repository](https://github.com/NVlabs/geomapnet)

[Leonhard Feiner](https://github.com/LeonhardFeiner) contributed equally to include support for learning semantic labels (later on called multitask learning) until commit [7e5c754](https://github.com/a1302z/geomapnet/commit/7e5c754136a3cd2c04f1ad01b240908092e04636). 
### License

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 


## Documentation 

The original CVPR 2018 paper can be found at

[Samarth Brahmbhatt, Jinwei Gu, Kihwan Kim, James Hays, and Jan Kautz. Geometry-Aware Learning of Maps for Camera Localization. CVPR 2018.](https://arxiv.org/abs/1712.03342).

Video to original result: \
[![mapnet](./figures/mapnet.png)](https://www.youtube.com/watch?v=X6mF_IbOb4A)

## Modifications to NVIDIA repository
Modifications by Leonhard Feiner and Alexander Ziller: 
 - Support for DeepLoc dataset
 - Development of Dual-Input model (additional semantics as input to model)
 - Development of Multitask model (additional semantics as output to model)
 
Modifications by Alexander Ziller:
 - Support for AachenDayNight and Cambridge Landmarks dataset
 - Including augmentations for AachenDayNight dataset (using CycleGANs)






## Setup

The original implementation used python 2.7 and was upgraded to python 3.5. Unfortunately the Robot Car SDK used in some parts of the code is only available in python 2.7. Therefore training and validating the robot car dataset requires to use python 2.7. The compatibility of current code with python 2.7 and with robot car is not fully tested. We recommend to use python 3.5.

MapNet uses a Conda environment that makes it easy to install all dependencies.

1. Install [Anaconda](https://www.anaconda.com/download/) with Python 3.7.

2. Create the `mapnet` Conda environment: `conda env create -f environment_py3.yml`.

3. Activate the environment: `conda activate mapnet_release`.

## Data
Currently supported datasets:
 - [DeepLoc](http://deeploc.cs.uni-freiburg.de/)
 - [CambridgeLandmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/)
 - [AachenDayNight](https://www.visuallocalization.net)
 - 7Scenes
 - RobotCar (see remark in [Setup](##-setup ) )
 
### Getting Started
To use a dataset:
 1. Create in data/deepslam_data a directory with the corresponding name e.g. AachenDayNight
 2. Download data into this directory
 3. Go to scripts directory and run dataset_mean.py and calc_pose_stats.py (also good to verify data structure is correct)
<!---
We support the
[7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/), the [Oxford RobotCar](http://robotcar-dataset.robots.ox.ac.uk/) and the [DeepLoc](http://deeploc.cs.uni-freiburg.de/) datasets.
now. You can also write your own PyTorch dataloader for other datasets and put it in the
`dataset_loaders` directory. Refer to 
[this README file](./dataset_loaders/README.md) for more details.

The datasets live in the `data/deepslam_data` directory. We provide skeletons
with symlinks to get you started. Let us call your 7Scenes download directory
7SCENES_DIR and your main RobotCar download directory (in which you untar all
the downloads from the website) ROBOTCAR_DIR. You will need to make the following
symlinks:

`
cd data/deepslam_data &&
ln -s 7SCENES_DIR 7Scenes &&
ln -s ROBOTCAR_DIR RobotCar_download 
ln -s DEEPLOC_DIR DeepLoc 
`

---

#### Special instructions for RobotCar: (only needed for RobotCar data)

1. Download
[this fork](https://github.com/samarth-robo/robotcar-dataset-sdk/tree/master) of
the dataset SDK, and run `cd scripts && ./make_robotcar_symlinks.sh` after
editing the `ROBOTCAR_SDK_ROOT` variable in it appropriately.

2. For each sequence, you need to download the `stereo_centre`, `vo` and `gps`
tar files from the dataset website.

3. The directory for each 'scene' (e.g. `full`) has .txt files defining the
train/test split. While training MapNet++,
you must put the sequences for self-supervised learning (dataset T in the paper)
in the `test_split.txt` file. The dataloader for the MapNet++ models will use
both images and ground-truth pose from sequences in `train_split.txt` and only
images from the sequences in `test_split.txt`.

4. To make training faster, we pre-processed the images using
`scripts/process_robotcar_images.py`. This script undistorts the images using
the camera models provided by the dataset, and scales them such that the shortest
side is 256 pixels.

---
--->


## Running the code

### Models
The following models are available
 - Posenet: Standard visuallocalization network by Alex Kendall
 - Mapnet: Base where this repository is forked from
 - Mapnet++: Mapnet with additional visual odometry and GPS data
 - SemanticOutput: Model does not learn localization but only semantics
 - Dual-Input (semanticV3): Mapnet with additional semantics as input
 - Multitask: Mapnet with additional semantics as output (learning target)

### Training
The executable script is `scripts/train.py`. Please go to the `scripts` folder to run these commands. (The DeepLoc dataset does not require a scene) For example:


- PoseNet on `DeepLoc`: `python train.py --dataset DeepLoc --config_file configs/posenet.ini --model posenet`

- MapNet on `DeepLoc`: `python train.py --dataset DeepLoc --config_file configs/mapnet.ini --model mapnet`

- Dual-Input on `DeepLoc`: `python train.py --dataset DeepLoc --config_file configs/mapnet-multiinput.ini --model semanticV3 --device 0
--learn_beta --learn_gamma`

- Multitask  on `DeepLoc`: `python train.py --dataset DeepLoc --config_file configs/uncertainty-criterion.ini --model multitask
--learn_beta --learn_gamma --learn_sigma --uncertainty_criterion`

<!---
- PoseNet on `chess` from `7Scenes`: `python train.py --dataset 7Scenes
--scene chess --config_file configs/posenet.ini --model posenet --device 0
--learn_beta --learn_gamma`

![train.png](./figures/train.png)

- MapNet on `chess` from `7Scenes`: `python train.py --dataset 7Scenes
--scene chess --config_file configs/mapnet.ini --model mapnet
--device 0 --learn_beta --learn_gamma`

- MapNet++ is finetuned on top of a trained MapNet model:
`python train.py --dataset 7Scenes --checkpoint <trained_mapnet_model.pth.tar>
--scene chess --config_file configs/mapnet++_7Scenes.ini --model mapnet++
--device 0 --learn_beta --learn_gamma`

![train.png](./figures/train.png)


- MapNet on `chess` from `7Scenes`: `python train.py --dataset 7Scenes
--scene chess --config_file configs/mapnet.ini --model mapnet
--device 0 --learn_beta --learn_gamma`

--->
- MapNet++ model on `heads` from a pretrained MapNet model: `python train.py --dataset 7Scenes --checkpoint logs/7Scenes_heads_mapnet_mapnet_learn_beta_learn_gamma/epoch_250.pth.tar --scene heads --config_file configs/mapnet++_7Scenes.ini --model mapnet++ --device 0 --learn_beta --learn_gamma`


For MapNet++ training, you will need visual odometry (VO) data (or other
sensory inputs such as noisy GPS measurements). For 7Scenes, we provided the
preprocessed VO computed with the DSO method. For RobotCar, we use the provided
stereo_vo. If you plan to use your own VO data (especially from a monocular
camera) for MapNet++ training, you will need to first align the VO with the
world coordinate (for rotation and scale). Please refer to the "Align VO"
section below for more detailed instructions.


The meanings of various command-line parameters are documented in
`scripts/train.py`. The values of various hyperparameters are defined in a 
separate .ini file. We provide some examples in the `scripts/configs` directory,
along with a [README](./scripts/configs/README.md) file explaining some
hyper-parameters.

If you have `visdom = yes` in the config file, you will need to start a Visdom
server for logging the training progress:

`python -m visdom.server -env_path=scripts/logs/`.


### Demo/Inference
<!--- The trained models for all experiments (7Scenes and RobotCar) presented in the paper can be downloaded
[here](https://drive.google.com/open?id=1J2QG_nHrRTKcDf9CGXRK9MWH1h-GuMLy). --->
The inference script is `scripts/eval.py`. Here are some examples, assuming
the models are downloaded in `scripts/logs`. Please go to the `scripts` folder to run the commands.

#### DeepLoc
The DeepLoc dataset does not require a scene.

- PoseNet:
```
$ python eval.py --dataset DeepLoc --model posenet \
--weights logs/DeepLoc__posenet_posenet_learn_beta/epoch_300.pth.tar \
--config_file configs/posenet.ini --val
```

- MapNet:
```
$ python eval.py --dataset DeepLoc --model mapnet \
--weights logs/DeepLoc__mapnet_mapnet/epoch_300.pth.tar \
--config_file configs/mapnet.ini --val --pose_graph
```

- For evaluating on the `train` split remove the `--val` flag

- To save the results to disk without showing them on screen (useful for scripts),
add the `--output_dir ../results/` flag

- See [this README file](./scripts/configs/README.md)
for more information on hyper-parameters and which config files to use.


- Dual-Input:
```
$ python eval.py --dataset DeepLoc --model semanticV3 \
--weights logs/DeepLoc__semanticV3_mapnet-multiinput/epoch_300.pth.tar \
--config_file configs/mapnet-multiinput.ini --val
```

- Multitask:
```
$ python eval.py --dataset DeepLoc --model multitask \
--weights logs/DeepLoc__multitask_uncertainty-criterion_learn_beta_learn_gamma_learn_sigma_uncertainty_criterion/epoch_300.pth.tar \
--config_file configs/uncertainty-criterion.ini --val
```

---

<!--- 
#### 7_Scenes
- MapNet++ with pose-graph optimization (i.e., MapNet+PGO) on `heads`:
```
$ python eval.py --dataset 7Scenes --scene heads --model mapnet++ \
--weights logs/7Scenes_heads_mapnet++_mapnet++_7Scenes/epoch_005.pth.tar \
--config_file configs/pgo_inference_7Scenes.ini --val --pose_graph
Median error in translation = 0.12 m
Median error in rotation    = 8.46 degrees
```
![7Scenes_heads_mapnet+pgo](./figures/7Scenes_heads_mapnet+pgo.png)


- For evaluating on the `train` split remove the `--val` flag

- To save the results to disk without showing them on screen (useful for scripts),
add the `--output_dir ../results/` flag

- See [this README file](./scripts/configs/README.md)
for more information on hyper-parameters and which config files to use.


- MapNet++ on `heads`:
```
$ python eval.py --dataset 7Scenes --scene heads --model mapnet++ \
--weights logs/7Scenes_heads_mapnet++_mapnet++_7Scenes/epoch_005.pth.tar \
--config_file configs/mapnet.ini --val
Median error in translation = 0.13 m
Median error in rotation    = 11.13 degrees
```

- MapNet on `heads`:
```
$ python eval.py --dataset 7Scenes --scene heads --model mapnet \
--weights logs/7Scenes_heads_mapnet_mapnet_learn_beta_learn_gamma/epoch_250.pth.tar \
--config_file configs/mapnet.ini --val
Median error in translation = 0.18 m
Median error in rotation    = 13.33 degrees
```

- PoseNet (CVPR2017) on `heads`:
```
$ python eval.py --dataset 7Scenes --scene heads --model posenet \
--weights logs/7Scenes_heads_posenet_posenet_learn_beta_logq/epoch_300.pth.tar \
--config_file configs/posenet.ini --val
Median error in translation = 0.19 m
Median error in rotation    = 12.15 degrees
```

#### RobotCar
- MapNet++ with pose-graph optimization on `loop`:
```
$ python eval.py --dataset RobotCar --scene loop --model mapnet++ \
--weights logs/RobotCar_loop_mapnet++_mapnet++_RobotCar_learn_beta_learn_gamma_2seq/epoch_005.pth.tar \
--config_file configs/pgo_inference_RobotCar.ini --val --pose_graph
Mean error in translation = 6.74 m
Mean error in rotation    = 2.23 degrees
```
![RobotCar_loop_mapnet+pgo](./figures/RobotCar_loop_mapnet+pgo.png)

- MapNet++ on `loop`:
```
$ python eval.py --dataset RobotCar --scene loop --model mapnet++ \
--weights logs/RobotCar_loop_mapnet++_mapnet++_RobotCar_learn_beta_learn_gamma_2seq/epoch_005.pth.tar \
--config_file configs/mapnet.ini --val
Mean error in translation = 6.95 m
Mean error in rotation    = 2.38 degrees
```

- MapNet on `loop`:
```
$ python eval.py --dataset RobotCar --scene loop --model mapnet \
--weights logs/RobotCar_loop_mapnet_mapnet_learn_beta_learn_gamma/epoch_300.pth.tar \
--config_file configs/mapnet.ini --val
Mean error in translation = 9.84 m
Mean error in rotation    = 3.96 degrees
```
--->


---
### Visual Explanations of Model
During the Practical we included [code](https://github.com/1Konny/gradcam_plus_plus-pytorch) to calculate maps as described in the [GradCam++ paper](https://arxiv.org/pdf/1710.11063.pdf). This can be calculated by:
```
python show_gradcampp.py --dataset DeepLoc --model multitask --val --n_activation_maps 3 --layer_name layer1,layer2 --config_file configs/uncertainty-criterion.ini --weights logs/DeepLoc__multitask_multitask-new-criterion_learn_beta_learn_gamma_learn_sigma_seed13/epoch_300.pth.tar
```


### Network Attention Visualization
Calculates the network attention visualizations and saves them in a video

- For the MapNet model trained on `chess` in `7Scenes`:
```
$ python plot_activations.py --dataset 7Scenes --scene chess
--weights <filename.pth.tar> --device 1 --val --config_file configs/mapnet.ini
--output_dir ../results/
```
Check [here](https://www.youtube.com/watch?v=hKlE45mJ2yY) for an example video of 
computed network attention of PoseNet vs. MapNet++.


---

### Other Tools 

#### Align VO to the ground truth poses
This has to be done before using VO in MapNet++ training. The executable script
is `scripts/align_vo_poses.py`.

- For the first sequence from `chess` in `7Scenes`:
`python align_vo_poses.py --dataset 7Scenes --scene chess --seq 1 --vo_lib dso`.
Note that alignment for `7Scenes` needs to be done separately for each sequence,
and so the `--seq` flag is needed

- For all `7Scenes` you can also use the script `align_vo_poses_7scenes.sh`
The script stores the information at the proper location in `data`

#### Mean and stdev pixel statistics across a dataset
This must be calculated before any training. Use the `scripts/dataset_mean.py`,
which also saves the information at the proper location. We provide pre-computed
values for RobotCar and 7Scenes.

#### Calculate pose translation statistics
Calculates the mean and stdev and saves them automatically to appropriate files
`python calc_pose_stats.py --dataset 7Scenes --scene redkitchen`
This information is needed to normalize the pose regression targets, so this
script must be run before any training. We provide pre-computed values for 
RobotCar and 7Scenes.

#### Plot the ground truth and VO poses for debugging
`python plot_vo_poses.py --dataset 7Scenes --scene heads --vo_lib dso --val`. To save the
output instead of displaying on screen, add the `--output_dir ../results/` flag

#### Process RobotCar GPS
The `scripts/process_robotcar_gps.py` script must be run before using GPS for
MapNet++ training. It converts the csv file into a format usable for training.

#### Demosaic and undistort RobotCar images
This is advisable to do beforehand to speed up training. The
`scripts/process_robotcar_images.py` script will do that and save the output
images to a `centre_processed` directory in the `stereo` directory. After the
script finishes, you must rename this directory to `centre` so that the dataloader
uses these undistorted and demosaiced images.

---

### Citation
Citation for original MapNet:

```
@inproceedings{mapnet2018,
  title={Geometry-Aware Learning of Maps for Camera Localization},
  author={Samarth Brahmbhatt and Jinwei Gu and Kihwan Kim and James Hays and Jan Kautz},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```
