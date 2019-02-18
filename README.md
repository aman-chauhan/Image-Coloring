# Image-Coloring

Deep Neural Net for coloring grayscale images using local and global image features

## Packages and Versions
-   `python` - version 3.6
-   `tensorflow` - version 1.12
-   `keras` - version 2.2.4
-   `keras-applications` - version 1.0.7
-   `imageio`

## Transfer Learning Models

-   DenseNet-121
-   Inception-Resnet-V2
-   Inception-V3
-   ResNet50
-   VGG19
-   Xception

## Folders

-   `docs` - Folder to store static Website
-   `weights` - Folder to store all model weights
-   `models` - Folder to store all model codes
-   `logs` - Folder to store all logs

## Files

-   `download.sh` - Shell script to download the dataset.
-   `clean_places365.py` - Python script to remove grayscale images from dataset
    -   `python clean_places365.py train` - to clean train.txt
    -   `python clean_places365.py val` - to clean val.txt
-   `train.txt` - Cleaned training files list
-   `val.txt` - Cleaned validation files list
-   `model_shape.py` - Python script to generate model shapes for docs.
