import numpy as np
import random
from classifiers import *
from pipeline import *
from datetime import datetime
from pathlib import Path
import json
import os
from tqdm import tqdm
import pickle

classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

path = os.getcwd()

with open(str(path) + str('/metadata/traindata.json')) as json_file:
    train_data = json.load(json_file)
with open(str(path) + str('/metadata/testdata.json')) as json_file:
    test_data = json.load(json_file)

X_train = []
y_train = []

X_test = []
y_test = []

frame_subsample_count = 10

# with open(str(path) + str('/dfdc_train_part_0/metadata.json')) as json_file:
#     data = json.load(json_file)

# test_dict = {}
# train_dict = {}
# for i in data.keys():
#     if data[i]["label"] == "FAKE":
#         data[i]["label"] = 1
#     else:
#         data[i]["label"] = 0
#     temp = random.random()
#     if temp < 0.2:
#         data[i]["split"] = "test"
#         test_dict[i] = (data[i])
#     else:
#         train_dict[i] = (data[i])
# json_object = json.dumps(test_dict)
# jsonFile = open("testdata.json", "w")
# jsonFile.write(json_object)
# train_object = json.dumps(train_dict)
# trainFile = open("traindata.json","w")
# trainFile.write(train_object)
# trainFile.close()
# jsonFile.close()


#####ABOVE THIS IS JUST CREATING TRAIN TEST FILES

print('Generating test data...')
for i in tqdm(test_data.keys()):
    face_finder = FaceFinder(str(path)+'/dfdc_train_part_0/'+str(i), load_first_face=False)
    skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
    face_finder.find_faces(resize=0.5, skipstep=skipstep)

    gen = FaceBatchGenerator(face_finder)
    X_test.append(gen.next_batch(frame_subsample_count))
    y_test.append(test_data[i]["label"])

with open(str(path) + '/X_test.pickle', 'wb') as f:
    pickle.dump(X_test, f)
with open(str(path) + '/y_test.pickle', 'wb') as f:
    pickle.dump(y_test, f)
