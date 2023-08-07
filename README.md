# VLPL: Vision Language Pseudo Label for Multi-label Learning with Single Positive Labels
This is the official Pytorch implementation paper "VLPL: Vision Language Pseudo Label for Multi-label Learning with Single Positive Labels".

Authors: Xin Xing, Zhexiao Xiong, Abby Stylianou, Liyu Gong, and Nathan Jacobs

Corresponding author: Xin Xing (xin.xing@uky.edu)

### Abstract

We address the task of multi-label image classification, which is essentially single-label image classification without the constraint that there is a single class present in the image. This task is similar to object detection, without the need to localize or count individual objects. Unfortunately, much like object detection, obtaining high-quality multi-label annotations is time-consuming and error-prone. To address this challenge, we consider the single-positive label setting, in which only a single positive class is annotated, even when multiple classes are present in a given image. The current state-of-the-art (SOTA) methods for this setting mainly propose novel loss functions to improve model performance. Several works have attempted to use pseudo-labels, but these approaches haven’t worked well. We propose a novel model called Vision-Language Pseudo-Labeling (VLPL) which uses a vision-language model to suggest strong positive and negative pseudo-labels. We demonstrate the effectiveness of the proposed VLPL model on four popular benchmarks: Pascal VOC, MS-COCO, NUS-WIDE, and CUB-Birds datasets. The results of VLPL outperform several strong baselines and indicate the effectiveness of the proposed approach. Furthermore, we explore the backbone architecture and outperform the SOTA method by 5.4% on Pascal VOC, 15.6% on MS-COCO, 15.2% on NUS-WIDE, and 11.3% on CUB-Birds.

## 🛠️ Installation
1. Create a Conda environment for the code:
```
conda create --name SPML python=3.8.8
```
2. Activate the environment:
```
conda activate SPML
```
3. Install the dependencies:
```
pip install -r requirements.txt
```

## 📖 Preparing Datasets
### Downloading Data
#### PASCAL VOC

1. Run the following commands:

```
cd {PATH-TO-THIS-CODE}/data/pascal
curl http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar --output pascal_raw.tar
tar -xf pascal_raw.tar
rm pascal_raw.tar
```

#### MS-COCO

1. Run the following commands:

```
cd {PATH-TO-THIS-CODE}/data/coco
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip --output coco_annotations.zip
curl http://images.cocodataset.org/zips/train2014.zip --output coco_train_raw.zip
curl http://images.cocodataset.org/zips/val2014.zip --output coco_val_raw.zip
unzip -q coco_annotations.zip
unzip -q coco_train_raw.zip
unzip -q coco_val_raw.zip
rm coco_annotations.zip
rm coco_train_raw.zip
rm coco_val_raw.zip
```

#### NUS-WIDE

1.  Follow the instructions in [this website](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) to download the raw images of NUS-WIDE named `Flickr.zip`.
2.  Run the following commands:
```
mv {PATH-TO-DOWNLOAD-FILES}/Flickr.zip {PATH-TO-THIS-CODE}/data/nuswide
unzip -q Flickr.zip
rm Flickr.zip
```

#### CUB

1.  Download `CUB_200_2011.tgz` in [this website](https://data.caltech.edu/records/20098).
2.  Run the following commands:
```
mv {PATH-TO-DOWNLOAD-FILES}/CUB_200_2011.tgz {PATH-TO-THIS-CODE}/data/cub
tar -xf CUB_200_2011.tgz
rm CUB_200_2011.tgz
```

### Formatting Data
For PASCAL VOC, MS-COCO, and CUB, use Python code to format data:
```
cd {PATH-TO-THIS-CODE}
python preproc/format_pascal.py
python preproc/format_coco.py
python preproc/format_cub.py
```
For NUS-WIDE, please download the formatted files [here](https://drive.google.com/drive/folders/1YL7WhnGpd-pjbtPL5r6IKiPeYFVdpYne?usp=sharing) and move them to the corresponding path:
```
mv {PATH-TO-DOWNLOAD-FILES}/{DOWNLOAD-FILES} {PATH-TO-THIS-CODE}/data/nuswide
```
`{DOWNLOAD-FILES}` should be replaced by `formatted_train_images.npy`, `formatted_train_labels.npy`, `formatted_val_images.npy`, or `formatted_train_labels.npy`.


### Generating Single Positive Annotations
In the last step, run `generate_observed_labels.py` to yield single positive annotations from full annotations of each dataset:
```
python preproc/generate_observed_labels.py --dataset {DATASET}
```
`{DATASET}` should be replaced by `pascal`, `coco`, `nuswide`, or `cub`.

## 🦍 Training and Evaluation
Run `main.py` to train and evaluate a model:
```
python main.py -d {DATASET} -l {LOSS} -g {GPU} -m {model} -t {tempurature} -th {threshold}  -p {partical} -s {PYTORCH-SEED}
```
Command-line arguments are as follows:
1. `{DATASET}`: The adopted dataset. (*default*: `pascal` | *available*: `pascal`, `coco`, `nuswide`, or `cub`)
2. `{LOSS}`: The method used for training. (*default*: `EM_PL` | *available*: `bce`, `iun`, `an`, `EM`, `EM_APL`, or `EM_PL`)
3. `{GPU}`: The GPU index. (*default*: `0`)
4. `{PYTORCH-SEED}`: The seed of PyTorch. (*default*: `0`)
5. `{model}`: The model of backbone. (*default*: `resnet50`| *available*: `resnet50`, `vit_clip`, `convnext_xlarge_22k`, or `convnext_xlarge_1k`)
6. `{tempurature}`: the temperature scalar of the softmax function.
7. `{threshold}`: the threshold for the positive pseudo-label. (*default*: `0.3`)
8. `{partical}`: the percentage of the negative pseudo-label. (*default*: `0.0`)

For example, to train and evaluate a model on the PASCAL VOC dataset using  EM loss+ VLPL, please run:
```
python main.py -d pascal -l EM_PL 
```

## Results:

## Acknowledgement:
Many thanks to the authors of [single-positive-multi-label](https://github.com/elijahcole/single-positive-multi-label) and [SPML-AckTheUnknown
](https://github.com/Correr-Zhou/SPML-AckTheUnknown). Our scripts are highly based on their scripts.

