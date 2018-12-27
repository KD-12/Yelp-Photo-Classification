#https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/48
import gzip
import numpy as np
import cv2
import tarfile
import glob
import re
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pandas as pd
import itertools

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop

from sklearn import preprocessing
import time
import pickle

class MyInceptionFeatureExtractor(nn.Module):
    def __init__(self, inception, transform_input=False):
        super(MyInceptionFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        # stop where you want, copy paste from the model def

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        x = F.max_pool2d(x, kernel_size=x.shape[-1])
        return x


def create_photo_to_biz_dict(path):
    photo_to_biz = defaultdict(list)
    df = pd.read_csv(path, dtype={0: str, 1:str})
    for (photo_id,biz_id) in df.values:
        photo_to_biz[photo_id].append(biz_id)

    return photo_to_biz

def img_norm(img):
    return 2 * (np.float32(img) / 255 - 0.5)
def extract_features(data, model):
    batch_data = []

    minibatch_data = np.reshape(data, (len(data), 3, 229, 229))
    if use_gpu:
        batch_data = torch.from_numpy(minibatch_data).float().cuda(0)
    else:
        batch_data = torch.from_numpy(minibatch_data).float()

    x_var = Variable(batch_data)

    output_scores = model(x_var)

    temp_feature_array = np.array(output_scores.data[:])

    return temp_feature_array

def generate_batch(mode, batch_num, photo_to_biz, model):
    path = glob.glob("./input/"+mode+"_photos/"+ "*.jpg")
    id_list = []
    data = []
    img_features = []

    for index , file_name in enumerate(path):
        if (index+1) % 25 == 0:
            print("number of images processed are:", (index+1))
            feature = extract_features(data, model)
            img_features.extend(np.reshape(feature,(feature.shape[0],feature.shape[1])))
            data = []

        photo_id = file_name.split("/")[-1].split("\\")[-1].replace(".jpg", "")
        if photo_id in photo_to_biz:
            id_list.append(photo_id)

            img = cv2.imread(file_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (229, 229), cv2.INTER_LINEAR)

            data.append(img_norm(img))

    """
    with open(mode+'_img_features_all_inception.pkl', 'wb') as f:
        pickle.dump(img_features, f)
    with open(mode+'_id_list_all_inception.pkl', 'wb') as f:
        pickle.dump(id_list, f)
    """

if __name__ == "__main__":
    print("Extracting Inception V3 Features:")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using System GPU for the Classifier")
    #### Create necessary dictionarys ##########################################
    train_photo_to_biz = create_photo_to_biz_dict("./train_photo_to_biz_ids.csv")
    test_photo_to_biz = create_photo_to_biz_dict("./test_photo_to_biz.csv")

    inception = models.inception_v3(pretrained=True)
    my_inception_model = MyInceptionFeatureExtractor(inception)
    if use_gpu:
        my_inception_model = my_inception_model.cuda()

    print("Creating train_img_features...........")
    #generate_batch("train",50, train_photo_to_biz,my_inception_model)
    print("Creating test_img_features.............")
    #generate_batch("test", 50, test_photo_to_biz, my_inception_model)

    with open('train_img_features_all_inception.pkl', 'rb') as f:
        train_img_features = pickle.load(f)
    with open('train_id_list_all_inception.pkl', 'rb') as f:
        train_id_list = pickle.load(f)

    with open('test_img_features_all_inception.pkl', 'rb') as f:
        test_img_features = pickle.load(f)
    with open('test_id_list_all_inception.pkl', 'rb') as f:
        test_id_list = pickle.load(f)
