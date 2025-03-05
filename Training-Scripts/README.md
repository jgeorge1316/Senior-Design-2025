# Scripts used for training the model
Please note that the custom dataset is not publicly available.

## [smallDataset.py](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/Training-Scripts/smallDataset.py)
Initial script used to setup a small dataset for training. This script sets up the structure for training the model. It takes in a folder which contains the folders with all of the images for each class. It then takes 100 images from each class, and splits them into train, val, and test (70%, 15%, 15% respectively). The script was made with help from ChatGPT [citation](https://chatgpt.com/share/67c75da4-b03c-800b-a3d8-bdb66cd86d36). The structure of the dataset for training is detailed [here](https://docs.ultralytics.com/datasets/classify/) in the Ultralytics docs.

## [dataset_setup.py](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/Training-Scripts/dataset_setup.py)
This script sets up the structure for training the model. It takes in a folder which contains the folders with all of the images for each class. It then takes 1500 images from each class (or the total amount of images for that class), and splits them into train, val, and test (70%, 15%, 15% respectively). The script was made with help from ChatGPT [citation](https://chatgpt.com/share/67c75d6e-6510-800b-8d84-d80337eb2ba5).

Update: The script should now add copy remaining images (if there are any after initial sampling of 1500) to the test folder. This will allow us to fully test a custom trained model on the entire dataset, while excluding the images used in training and validation. This should actually be automatically done as well as part of [train.py](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/Training-Scripts/train.py), since it will go through the entire test set. This has **Not** been tested yet, neither has a model been trained using the folder that was generated.

TODO: Train and test a model using the dataset (dataset6 on Lambda AI Workstation) created by the updated script above^


## [train.py](https://github.com/jgeorge1316/Senior-Design-2025/blob/main/Training-Scripts/train.py)
Script that trains a model. It takes in pretrained YOLO11 classifcation [weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt) and a custom dataset to train a model. Created using Ultralytics [Classify](https://docs.ultralytics.com/tasks/classify/) and [dataset](https://docs.ultralytics.com/datasets/classify/) docs.