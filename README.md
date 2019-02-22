# Image-Coloring

Deep Neural Net for coloring grayscale images using local and global image features

## Packages and Versions

-   `python` - version 3.6
-   `tensorflow` - version 1.12
-   `keras` - version 2.1.5
-   `keras-applications` - version 1.0.7
-   `imageio`
-   `Pillow`
-   `scikit-image`

## Transfer Learning Model Dictionary

-   `densenet` - DenseNet-121
-   `inceptresnet` - Inception-Resnet-V2
-   `inception` - Inception-V3
-   `resnet` - ResNet50
-   `vgg` - VGG19
-   `xception` - Xception

## Folders

-   `docs` - Folder to store static Website
-   `weights` - Folder to store all model weights
-   `models` - Folder to store all model codes
-   `logs` - Folder to store all logs

## Files

-   `download.sh` - Shell script to download the dataset.
-   `classes.txt` - Sorted list of all classes in dataset, generated by `download.sh`.
-   `clean_places365.py` - Python script to remove grayscale images from dataset
    -   `python clean_places365.py train` - to clean train.txt
    -   `python clean_places365.py val` - to clean val.txt
-   `train.txt` and `val.txt` - Cleaned training and validation files list.
-   `model_shape.py` - Python script to generate model shapes for docs.
-   `generator.py` - DataGenerator class for feeding the images to the model for training and evaluation.
-   `checkpoint.py` - ModelCheckpoint class for saving model weights whenever validation loss decreases after an epoch.
-   `IC_train.py` - Python script to train the network.
    -   `python IC_train.py <model_type> <batch_size>` - to train a specific type of model with the given batch size.
    -   Example - `python IC_train.py inception 32` - to train the Image-Coloring model using transfer learning layers from InceptionV3. See **_Transfer Learning Model Dictionary_** section to see the valid and available transfer learning models.

## [NOTE] Keras Version issues

-   During this and many other projects, I noticed that there are many open issues regarding model checkpointing in Keras 2.2 onwards. For this reason, I have stuck to training the models on Keras 2.1.5.
-   One of the issues comes with saving the model (weights + architecture + optimizer state) with Lambda layers which have Keras/Tensorflow arguments as input.
    -   See this [issue](https://github.com/keras-team/keras/issues/8343) and [issue](https://github.com/keras-team/keras/issues/10528). They suggest changing a lot of things, amounting to hacking the system to work.
    -   This [answer on StackOverflow](https://stackoverflow.com/questions/47066635/checkpointing-keras-model-typeerror-cant-pickle-thread-lock-objects) gives a very detailed explanation of the issue. This basically involves implementing Tensor-to-numpy-to-Tensor transformations. These seem too expensive an operation for a task that is going to be executed millions of times during training.
-   If we avoid saving the model along with architecture and optimizer state and just save the weights, we can solve the above issue. But since Keras 2+, saving models with Siamese networks or Transfer learning layers tend to fail. It tends to throw `ValueError: axes don't match array` when we are trying to load the saved weights into the model again. This issue doesn't exist below 2.1.6 to my knowledge right now.
    -   See this [issue](https://github.com/keras-team/keras/issues/10428). [This](https://github.com/keras-team/keras/issues/10428#issuecomment-418303297) suggestion seems to work everytime. In this project, thankfully, Tensorflow 1.12 doesn't seem to cause any issue. Maybe someone should update the issue...
    -   Another [issue](https://github.com/experiencor/keras-yolo2/issues/358) where the same error arises. The same suggestion seems to work again!
