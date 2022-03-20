# Deepfake Detection
This repository contains code for the application of transfer learning on the MesoNet architecture and FaceForensics++ Dataset.

# Setup

Create a directory to store all of the videos and metadata from the dataset. One subdirectory, named 'dfdc_train_part_0' will be dedicated to just the videos. 
The other, 'metadata' should contain two files, testdata.json and traindata.json that contain information about the train and test dataset. One entry in traindata.json would be structured as follows:

{"name": {"label": (0 if real 1 if fake), "split": "train or test", "original": (name of original file if label is 1)}}
{"kmcdjxmnoa.mp4": {"label": 1, "split": "train", "original": "sttnfyptum.mp4"}}

So the directory structure is:

-dir
--dfdc_train_part_0
----videos
--metadata
----traindata.json
----testdata.json

# Run Code

After the data is structured properly run 'gendata.py' to preprocess the data and place the (large) dataset into pickle files for our model.

Run 'train.py' to finlly train and then evaluate the model on the train and test datasets respectively. 