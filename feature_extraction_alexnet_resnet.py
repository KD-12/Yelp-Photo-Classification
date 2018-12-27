#https://github.com/u1234x1234/kaggle-yelp-restaurant-photo-classification
#http://blog.kaggle.com/2016/04/28/yelp-restaurant-photo-classification-winners-interview-1st-place-dmitrii-tsybulevskii/
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


def tar_extract(path):
    tar = tarfile.open(path,"r:gz")
    tar.extractall()
    tar.close()

def create_photo_to_biz_dict(path):
    photo_to_biz = defaultdict(list)
    df = pd.read_csv(path, dtype={0: str, 1:str})
    for (photo_id,biz_id) in df.values:
        photo_to_biz[photo_id].append(biz_id)

    return photo_to_biz

def make_img_set(path):
    t = tarfile.open(path, 'r', encoding="Latin-1")
    list1 = t.getmembers()
    for e, member in enumerate(list1):
        #print(member.name)
        if e > 0:
            if "._" not in member.name:
                t.extract(member, "input")

def check_img(img_name):
    path = glob.glob("./input/train_photos/"+img_name)[0]
    img = cv2.imread(path)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure
    plt.imshow(new_img)
    plt.show()

def img_norm(img):
    return 2 * (np.float32(img) / 255 - 0.5)

def extract_features(data, model):
    batch_data = []
    minibatch_data = np.reshape(data, (len(data), 3, 224, 224))
    if use_gpu:
        batch_data = torch.from_numpy(minibatch_data).float().cuda(0)
    else:
        batch_data = torch.from_numpy(minibatch_data).float()

    x_var = Variable(batch_data)

    output_scores = model(x_var)
    #print("shape of resnet features", output_scores.shape)
    x = output_scores.data[:].cpu().numpy()
    x = x.reshape(len(data),2048)
    return x
    #return np.array(output_scores.data[:])[:,:500] Alexnet Features

def generate_batch(mode, batch_num, photo_to_biz, model, pickler):
    path = glob.glob("./input/"+mode+"_photos/"+ "*.jpg")
    id_list = []
    data = []

    for index , file_name in enumerate(path):

        if (index+1) % 25 == 0:
            print("number of images processed are:", (index+1))
            #img_features.extend(extract_features(data, model)) # cannot fit all features in memory
            pickler.dump(extract_features(data, model)) #iterative pickle
            data = []
            #img_features = []

        photo_id = file_name.split("/")[-1].split("\\")[-1].replace(".jpg", "")
        if photo_id in photo_to_biz:
            id_list.append(photo_id)

            img = cv2.imread(file_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
            img = np.transpose(img, [2, 0, 1])
            data.append(img_norm(img))
    # previous whole pickle method
    """
    with open(mode+'_img_features_25.pkl', 'wb') as f:
        pickle.dump(img_features, f)
    with open(mode+'_id_list_25.pkl', 'wb') as f:
        pickle.dump(id_list, f)
    """

if __name__ == "__main__":
    print("Starting Yelp Classification")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using System GPU for the Classifier")

    ##### extract all dictionary csv ###############
    #tar_extract("./all/train_photo_to_biz.csv.tgz")
    #tar_extract("./all/test_photo_to_biz.csv.tgz")
    train_photo_to_biz = create_photo_to_biz_dict("./train_photo_to_biz_ids.csv")
    test_photo_to_biz = create_photo_to_biz_dict("./test_photo_to_biz.csv")
    #tar_extract("./all/train.csv.tgz")


    ##### extract all images from tar zip###########
    #make_img_set('./all/train_photos.tar')
    #make_img_set('./all/test_photos.tar')
    #check_img("15.jpg")

    ##### load pretrained models ##################
    alexnet_model = models.alexnet(pretrained=True)
    if use_gpu:
        alexnet_model = alexnet_model.cuda()
    for child in list(alexnet_model.classifier.children()):
            for param in list(child.parameters()):
                param.requires_grad = False
    features = list(alexnet_model.classifier.children())[:-1]
    alexnet_model.classifier = nn.Sequential(*features)

    resnet_model = models.resnet50(pretrained=True)
    if use_gpu:
        resnet_model = resnet_model.cuda()
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
    for param in resnet_model.parameters():
            param.requires_grad = False

    # pickle file for train features
    pickle_2048 = open('iterative_resnet_pickle_train.pkl', 'wb')
    pickler = pickle.Pickler(pickle_2048)
    print("Creating train_img_features...........")
    generate_batch("train",50, train_photo_to_biz, resnet_model, pickler)
    pickle_2048.close()

    # pickle file for test features
    pickle_2048 = open('iterative_resnet_pickle_test.pkl', 'wb')
    pickler = pickle.Pickler(pickle_2048)
    print("Creating test_img_features.............")
    generate_batch("test", 50, test_photo_to_biz, resnet_model, pickler)
    pickle_2048.close()
