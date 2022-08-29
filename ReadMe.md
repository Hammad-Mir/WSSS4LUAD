# Weakly Supervised Semantic Segmentation for Lung Adenocarcinoma histopathological images using Vision Transformers

The repository contains all the source code for Weakly supervised semantic segmentation for lung adenocarcinoma histopathological images. We explored the feasibility of using transformer based image classifier for pseudo-label generation for tissue semantic segmentation.

![](https://i.imgur.com/lqrGRyC.jpg)

The directory structure of the repository is as follws:

WSSS4LUAD
+ |
+ |---training:
+ |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
+ |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- classification_transformer: contains all the training notebooks for the classification
+ |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (transformer) networks.
+ |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
+ |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- segmentation: contains all the training notebooks for the segmentation networks.
+ |
+ |---image_preprocessing_augmentation: contains notebooks for image augmentation
+ |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(cutmix and padding).
+ |
+ |---cam_visualization: contains notebooks for visualizing CAM from various networks using
+ |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;various methods discussed in the paper.
+ |
+ |---demo: contains all the notebooks to demonstrate the working of various CAM generation
+ |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;methods for transformer based networks, CAM refinement, pseudo-label generation
+ |&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and segmentation model performance.
+ |
+ |---requirements: list of libraries required for the project

## Analysis

To run the analysis, use the notebooks from the demo directory. The best weights for best performing CAM generating models as well as segmentation model should be downloaded from the links below and placed in the models directory in the demo directory. Note that to run the Demo-GETAM-for-WSSS4LUAD-viz and Demo-Transformer-Explainability notebooks, CUDA is required.

A set of training images as well as the validation and test set images are provided in the dataset directory inside the demo directory.

The weights for the best performing models can be downloaded from the google google drive links below.

<center>

| Network/Task  | Link  |
|---|---|
| model_vit_base_patch32_224_2.pth          | [GDrive](https://drive.google.com/file/d/1MEWlWZ5yGTnMqImiJX3qVDytej0lHqkH/view?usp=sharing)  |
| model_vit_base_patch16_224_1.pth          | [GDrive](https://drive.google.com/file/d/1dpdFgqHvJB4XN-fkV-91-twCxJtca_3l/view?usp=sharing)  |
| cutmix_hila_vit_base_patch16_224_01.pth   | [GDrive](https://drive.google.com/file/d/1-0K0fa2ldf0VKSa3UwdTEiWV0LH3TaHN/view?usp=sharing)  |
| border_cutmix_GETAM.pth                   | [GDrive](https://drive.google.com/file/d/1In2msf66Q7Ea4lEW638pYzXWwr5nr7X7/view?usp=sharing)  |
| deeplabv3plus_dJ_par_resnet50_01.pth      | [GDrive](https://drive.google.com/file/d/1cQDRNmfM7_RmIIjQjOIhjRsBQySnT171/view?usp=sharing)  |

</center>