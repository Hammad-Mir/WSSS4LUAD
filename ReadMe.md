# Weakly Supervised Semantic Segmentation for Lung Adenocarcinoma histopathological images using Vision Transformers

The repository contains all the source code for Weakly supervised semantic segmentation for lung adenocarcinoma histopathological images. We explored the feasibility of using transformer based image classifier for pseudo-label generation for tissue semantic segmentation.

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

To run the analysis, use the notebooks from the demo directory. The best weights for best performing CAM generating models as well as segmentation model are stored in model directory and available on google colab. Note that to run the Demo GETAM for WSSS4LUAD viz and Demo Transformer Explainability notebooks, CUDA is required.

A set of images from the dataset to run the analysis will also be provided.
