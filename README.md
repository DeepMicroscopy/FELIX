# FELIX - Feline Evaluation with Layperson and ML Interaction Experience Dataset

## Overview

This repository provides the materials used in *A validated task for studying human-AI interaction in multiclass image classification*. The dataset consists of 60 creative commons-licensed images of cats to be classified into big cats (pantherinae), small cats (felinae) that are not domestic cats, and domestic cats (felis catus). The dataset is complemented by extensive data from 302 human subjects' evaluations and recommendations by a machine learning pipeline trained in the same task. 

<img src="paper_figures/Figure.png" width="800px" align="center"/>


## Contents

- **image_files:** The `image_files/` directory contains datasets used in the experiments. Each image file has a 3-digit name for unique identification of the image. For convenience, the image files are stored in subfolders: `image_files/Recommender_test_set`, and `image_files/Experimental_study_set`. The final image dataset of our proxy task refers to all 30 images in the recommender test dataset and all 30 images in the experimental study dataset. The 30 images excluded from the study are provided as well in the folder `image_files/Excluded`. 

- **imagenet_info:** The `imagenet_info/` directory contains some files that were used to train a classifier on the subset of cat images from the ImageNet training set.  

- **study_data:** The `study_data/` directory contains two files. The ```Image_Level_Data.csv``` file contains all information about each image including an indication to which dataset it belongs, the original web location, artist and licensing information, as well as classification results from the user study. The ```Subject_Level_Data.csv``` file contains all information about each subject in the study including the classification results and demographics. All participants are anonymized. 

- **lib:** The `lib/` directory contains the source code used to train the classifier.

## Downloading ImageNet Dataset

A subset of the ImageNet dataset is used to train the classifier. If you would like to train your own algorithm. Follow these steps and prepare the dataset. 

1. The original dataset with additional labels for object detection is semi-publicly available at [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data). To download the dataset you need to register a Kaggle account and join the challenge.

2. Download the data via the `kaggle-api` (see guidelines [here](https://github.com/Kaggle/kaggle-api#api-credentials)) or use the `Download All` button.

3. After the download is complete extract and unuzip the files in a directory of your choice. 

4. If you want to train a cat classifier, create a new virtual environment and install the `requirements.txt` file, e.g. with 


```
pip install -r requirements.txt 
```

5. Open the `00_load_imagenet.ipynb` notebook and follow the steps to prepare the dataset. Be sure to adapt the directories accordingly.

6. Training your own classifier can be down using the `07_train_model_catstudy.ipynb` notebook.


## License

The images located in the ```image_files/``` directory were obtained with the necessary rights and modified to meet our specific requirements. As a result, we are able to publish these images for public use under the Creative Commons 4.0 BY-SA License.

The original images used in this study were primarily sourced from Wikimedia Commons, and they are subject to various Creative Commons licenses. Additionally, certain images were obtained from stock image databases, for which we have obtained the required licenses to incorporate them into our project in the form that is provided in this repository.

The ```Image_Level_Data.csv``` file links each image to its original location, acknowledging the artists and providing original license information for transparency and compliance.


## Acknowledgements

This work was supported by the Bavarian Institute for Digital Transformation (bidt) under the grant "Responsibility Gaps in Human Machine Interaction (ReGInA)".




