# Weakly Supervised Semantic Segmentation for Lung Adenocarcinoma histopathological images using Vision Transformers

The repository contains all the source code for Weakly supervised semantic segmentation for lung adenocarcinoma histopathological images. We explored the feasibility of using transformer based image classifier for pseudo-label generation for tissue semantic segmentation.

The directory structure of the repository is as follws:

WSSS4LUAD
+ ┣ cam_visualization
+ ┣ demo
+ ┣ image_preprocessing_augmentation
+ ┣ training
+ ┃ ┗ classification_transformer
+ ┃ ┗ segmentation
+ ┣ .gitignore
+ ┣ ReadMe.md
+ ┣ ReadMe.txt
+ ┗ requirements.txt

To run the analysis, use the notebooks from the demo directory. The best weights for best performing CAM generating models as well as segmentation model are stored in model directory and available on google colab. Note that to run the Demo GETAM for WSSS4LUAD viz and Demo Transformer Explainability notebooks, CUDA is required.

A set of images from the dataset to run the analysis will also be provided.