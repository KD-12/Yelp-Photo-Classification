#https://github.com/u1234x1234/kaggle-yelp-restaurant-photo-classification
#https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html
#http://blog.kaggle.com/2016/04/28/yelp-restaurant-photo-classification-winners-interview-1st-place-dmitrii-tsybulevskii/
import gzip
import numpy as np
import cv2
import tarfile
import glob
import re

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
from torch.utils import data

from torch.utils.data.dataset import Dataset

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_similarity_score
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import time
import pickle
import csv

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.data)

class Linear_Fully_connected(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(Linear_Fully_connected,self).__init__()


        self.linear = nn.Linear(input_shape, output_shape)
        self.norm = nn.BatchNorm1d(output_shape)
        self.relu = nn.LeakyReLU()

    def forward(self,input):
        output = self.linear(input)
        output = self.norm(output)
        output = self.relu(output)

        return output


class Network_Architecture(nn.Module):
    def __init__(self,feature_shape, label_shape):
        super(Network_Architecture,self).__init__()


        self.layer1 = Linear_Fully_connected(feature_shape, 2500)

        self.dropout1 = nn.Dropout(p=0.5)
        self.layer2 = Linear_Fully_connected(2500,512)

        self.dropout2 = nn.Dropout(p=0.5)
        self.layer3 = Linear_Fully_connected(512, 250)

        self.layer4 = Linear_Fully_connected(250, 9)

        self.network = nn.Sequential(self.layer1,self.dropout1, self.layer2,self.dropout2,
                                     self.layer3,self.layer4)
    def forward(self, input):
        output = self.network(input)
        return torch.sigmoid(output)

def train(model, loss_function, optimizer, train_dataloader, val_data_loader,num_epochs):
    print("Entered training phase")

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        model.train()

        print("starting epoch number:",epoch+1)

        for step_num, tup in enumerate(train_dataloader):
            #print("here epoch", epoch+1)
            if use_gpu:
                x_var = Variable(tup[0].cuda())
                y_var = Variable(tup[1].long().cuda())
            else:
                x_var = Variable(tup[0])
                y_var = Variable(tup[1].long())

            optimizer.zero_grad()
            output_scores = model(x_var)
            loss = loss_function(output_scores, y_var)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data.item() #* tup[0].size(0)
            temp_pred = output_scores.data.cpu().numpy()

            prediction = temp_pred > 0.5
            accuracy = sum(np.array(prediction) == tup[1].numpy()) / 200
            train_acc += accuracy.mean(axis = 0)

        train_acc_epoch = (float(train_acc)/len(train_dataloader))*100
        train_loss_epoch = float(train_loss)/len(train_dataloader)
        print("Epoch {}, Train Accuracy: {}, Train Loss: {}".format(epoch+1, train_acc_epoch, train_loss_epoch))
        evaluate(model, val_data_loader,"val")

def evaluate(model, testloader,mode):
    print("inside model evaluation")
    model.eval()
    prediction = []
    final_prediction = []
    val_loss = 0.0
    val_acc = 0.0
    for step_num, tup in enumerate(testloader):
        if (step_num + 1) % 25 == 0:
            print("evaluated ",(step_num+1)*25," bussinesses")
        if mode == "val":

            if use_gpu:
                test_output = model(tup[0].cuda())
                y_var = Variable(tup[1].long().cuda())
            else:
                test_output = model(tup[0])
                y_var = Variable(tup[1].long())
            loss = loss_function(test_output, y_var)

            val_loss += loss.cpu().data.item() #
            temp_pred = test_output.data.cpu().numpy()
            prediction = temp_pred > 0.5
            accuracy = sum(np.array(prediction) == tup[1].numpy())/100
            #print("not mean",accuracy)
            #print("accuracy",accuracy.mean(axis = 0))
            val_acc += accuracy.mean(axis = 0)

            #print("length of val prediction is:",prediction.shape)
        else:
            if use_gpu:
                test_output = model(tup.cuda())
            else:
                test_output = model(tup)
            temp_pred = test_output.data.cpu().numpy()
            #print(temp_pred.shape)

            final_prediction.extend(temp_pred)

    if mode == "val":
        val_acc_epoch = (float(val_acc)/(len(testloader)))*100
        val_loss_epoch = float(val_loss/(len(testloader)))
        print("Validation Accuracy: {}, Validation Loss: {}".format(val_acc_epoch, val_loss_epoch))
    if mode != "val":
        print("length of neural prediction is:",np.asarray(final_prediction).shape)


    return np.asarray(final_prediction)

##### Binary Relevance (not being used) ########################################
"""
def lr_classifier(X, Y, test_data):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    clf = LogisticRegression(C=100,solver='lbfgs', max_iter= 4000)
    #print("val shape",X_test.shape)
    predictions = np.zeros((X_test.shape[0], 9))
    preds_br = np.zeros((test_data.shape[0], 9))
    #print(predictions.shape)
    #print(Y.shape)
    for i in range(0, 9):
        try:
            clf.fit(X_train, y_train[:, i])
            predictions[:, i] = clf.predict(X_test)#[:, 1]
            preds_br[:, i] = clf.predict(test_data)
        except:
            print("No label found for class:",i)


    print(predictions[0])
    #print(y_test[0])
    accuracy = sum(np.array(predictions) == y_test) / float(len(X_test))
    #print(accuracy)
    print("The accuracy of Classifier on validation set is %.2f " % ((accuracy.mean(axis = 0)*100)))
    return preds_br


def svm_classifier(X, Y, test_data):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    chi2sampler = AdditiveChi2Sampler(sample_steps=2)
    X_transformed = chi2sampler.fit_transform(X_train, y_train)
    svm = LinearSVC(C=10, max_iter=7000)
    predictions = np.zeros((X_test.shape[0], 9))
    preds_br = np.zeros((test_data.shape[0], 9))
    for i in range(0, 9):
        try:
            svm.fit(X_train, y_train[:, i])
            predictions[:, i] = svm.predict(X_test)#[:, 1]
            preds_br[:, i] = svm.predict(test_data)
        except:
            print("No label found for class:",i)
            continue
    accuracy = sum(np.array(predictions) == y_test) / float(len(X_test))
    print("The accuracy of Classifier on validation set is %.2f " % ((accuracy.mean(axis = 0)*100)))
    return preds_br
"""
def chain_classifier(train_data, val_data, train_label, val_label, test_data, model):
    num_chains =  40
    #mlp_lr (128) test 61.6
    #mlp_lr (1204) test 59
    #mlp_lr
    #40 test 60.6
    #mlp classifier
    #40 test 59.06 #20 test 59  #2 63% val and 58% test
    #lr
    # 40 57 % #20 53 test accuracy #2 val accuracy 53% #10 Val accuracy 53 %
    chain_model = [ClassifierChain(model, order='random', random_state=i) for i in range(num_chains)]
    for chain in chain_model:
        chain.fit(train_data, train_label)
    chained_val_preds = []
    chained_test_preds = []
    chained_val_preds = np.array([chain.predict(val_data) for chain in chain_model])
    val_preds = np.asarray(chained_val_preds).mean(axis=0)
    print(jaccard_similarity_score(val_label, val_preds >= 0.5))
    chained_test_preds = np.array([chain.predict(test_data) for chain in chain_model])
    test_preds = np.asarray(chained_test_preds).mean(axis=0)
    return test_preds

def ensemble_classifier(train_data, val_data, train_label, val_label, test_data):
    lr_model = LogisticRegression(C=100,solver='lbfgs', max_iter= 80000)
    svm = LinearSVC(C=10, max_iter=70000)
    xgboost_model = XGBClassifier( max_depth=3, learning_rate=0.1, n_estimators=100, subsample=0.6, colsample_bytree=1.0,
                                   booster="gblinear",objective='reg:logistic')
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100, 100,),learning_rate= "adaptive", max_iter= 400)

    preds_lr_chain = chain_classifier(train_data, val_data, train_label, val_label, test_data, lr_model)
    #preds_xgb_chain = chain_classifier(X_train, X_val, y_train, y_val, test_data, xgboost_model) (val accuracy 48%)
    #preds_svm_chain = chain_classifier(X_train, X_val, y_train, y_val, test_data, svm) (val accuracy 45%)
    preds_mlp_chain = chain_classifier(train_data, val_data, train_label, val_label, test_data, mlp_model)
    ensemble_mlp_lr = (4*preds_mlp_chain + 3*preds_lr_chain)/7
    return ensemble_mlp_lr, preds_mlp_chain, preds_lr_chain

def create_photo_to_biz_dict(path):
    photo_to_biz = defaultdict(list)
    df = pd.read_csv(path, dtype={0: str, 1:str})
    for (photo_id,biz_id) in df.values:
        photo_to_biz[photo_id].append(biz_id)

    return photo_to_biz

def biz_to_label(path):
    biz_to_labels = {}
    df = pd.read_csv(path)
    for (biz_id, labels) in df.values:
        biz_to_labels[str(biz_id)] = labels
    return biz_to_labels

def creat_biz_features(mode,img_features, id_list, photo_to_biz,feature_type):
    print(mode+" img_ids",len(id_list))
    print("len of "+mode+" img features",len(img_features))
    if feature_type != "pca":
        preprocessing.normalize(img_features, copy=False)

    biz_features_dict = defaultdict(list)
    for (image_id, feature) in zip(id_list, img_features):
        biz_id_list = photo_to_biz.get(image_id)
        for biz_id in biz_id_list:
            biz_features_dict[biz_id].append(feature)
    print(mode+" business_id->features",len(biz_features_dict))
    return biz_features_dict

def create_train_data(biz_features_dict, Y):
    y = np.zeros((len(biz_features_dict), 9))
    x = np.array([])
    for i, (biz_id, features) in enumerate(biz_features_dict.items()):
        feature_mean = np.array(features).mean(axis=0)
        x = np.vstack((x, feature_mean)) if x.size else feature_mean
        if type(Y[biz_id]) == float and np.isnan(Y[biz_id]):
            continue
        for label in Y[biz_id].split(' '):
            y[i, int(label)] = 1
    print("shape of train set",x.shape)
    print("shape of labels",y.shape)
    return x, y

def create_test_data(biz_features_dict):
    x = np.array([])
    for i, (biz_id, features) in enumerate(biz_features_dict.items()):
        feature_mean = np.array(features).mean(axis=0)
        x = np.vstack((x, feature_mean)) if x.size else feature_mean

    return x
def feature_pooling(train_img_features, train_id_list, train_photo_to_biz, test_img_features, test_id_list, test_photo_to_biz, Y, feature_type):

    train_biz_features_dict = creat_biz_features("train",train_img_features, train_id_list, train_photo_to_biz,feature_type)
    test_biz_features_dict = creat_biz_features("test",test_img_features, test_id_list, test_photo_to_biz,feature_type)
    train_data, train_label = create_train_data(train_biz_features_dict, Y)
    test_data = create_test_data(test_biz_features_dict)

    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.10, random_state=42)
    return X_train, X_val, y_train, y_val, test_data

def write_result_file(output_preds):
    ids = pd.read_csv('sample_submission.csv').values[:, 0]
    print("ids read from sample submission file",len(ids))
    print("number of predictions made", len(output_preds))
    with open("result_lr_nn", "w") as f:
        f.write('business_id,labels')
        f.write("\n")
        for biz_id, pred in zip(ids, output_preds):
            true_indices = [i for i, x in enumerate(pred) if x]
            true_indices = [str(x) for x in true_indices]
            f.write(biz_id + ',' + ' '.join(true_indices)+"\n")

if __name__ == "__main__":
    print("Starting Ensemble Classification")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using System GPU for the Classifier")

    start = time.time()
    train_photo_to_biz = create_photo_to_biz_dict("./train_photo_to_biz_ids.csv")
    test_photo_to_biz = create_photo_to_biz_dict("./test_photo_to_biz.csv")
    Y = biz_to_label("./train.csv")
    with open('train_id_list_all_inception.pkl', 'rb') as f:
        train_id_list = pickle.load(f)
    with open('test_id_list_all_inception.pkl', 'rb') as f:
        test_id_list = pickle.load(f)
    final_prediction = np.zeros((1,10000,9))
    end = time.time()
    print("Prepared all necessary dictionarys in:", end-start)
    print("Prepare for bisuness wise training")

    ############# Inception V3 predictions #####################################
    print("Predictions with Inception V3")
    start = time.time()
    train_img_features = []
    with open('train_img_features_all_inception.pkl', 'rb') as f:
        train_img_features = pickle.load(f)
    test_img_features = []
    with open('test_img_features_all_inception.pkl', 'rb') as f:
        test_img_features = pickle.load(f)
    X_train, X_val, y_train, y_val, test_data = feature_pooling(train_img_features, train_id_list, train_photo_to_biz,
                                                     test_img_features, test_id_list, test_photo_to_biz,
                                                     Y, "not_pca")

    ensembled_preds_inception_748, preds_mlp_inception_748, preds_lr_inception_748 = ensemble_classifier(X_train, X_val, y_train, y_val, test_data)
    with open('ensembled_preds_inception_748.pkl', 'wb') as f:
        pickle.dump(ensembled_preds_inception_748, f)
    with open('preds_mlp_inception_748.pkl', 'wb') as f:
        pickle.dump(preds_mlp_inception_748, f)
    with open('preds_lr_inception_748.pkl', 'wb') as f:
        pickle.dump(preds_lr_inception_748, f)

    final_prediction = np.concatenate((final_prediction,ensembled_preds_inception_748[np.newaxis,:,:]), axis=0)
    end = time.time()
    print("Finished predictions with Inception V3 in ",end-start)
    ############### Alexnet prediction ############
    print("Predictions with Alexnet")
    start = time.time()
    train_img_features = []
    with open('train_img_features_all.pkl', 'rb') as f:
        train_img_features = pickle.load(f)
    test_img_features = []
    with open('test_img_features_all.pkl', 'rb') as f:
        test_img_features = pickle.load(f)
    X_train, X_val, y_train, y_val, test_data = feature_pooling(train_img_features, train_id_list, train_photo_to_biz,
                                                                test_img_features, test_id_list, test_photo_to_biz,
                                                                Y, "not_pca")

    ensembled_preds_alexnet_500, preds_mlp_alexnet_500, preds_lr_alexnet_500 = ensemble_classifier(X_train, X_val, y_train, y_val, test_data)
    with open('ensembled_preds_alexnet_500.pkl', 'wb') as f:
        pickle.dump(ensembled_preds_alexnet_500, f)
    with open('preds_mlp_alexnet_500.pkl', 'wb') as f:
        pickle.dump(preds_mlp_alexnet_500, f)
    with open('preds_lr_alexnet_500.pkl', 'wb') as f:
        pickle.dump(preds_lr_alexnet_500, f)
    final_prediction = np.concatenate((final_prediction, ensembled_preds_alexnet_500[np.newaxis,:,:]), axis=0)
    end = time.time()
    print("Finished predictions with Alexnet in ",end-start)
    ############## L2 normalized->Concatenated(resnet(2048), alexnet(500), inceptionV3(748))->PCA(1024) #############
    print("Predictions with Mixed features with dim= 1024")
    start = time.time()
    train_img_features = []
    train_pca = open('train_pca_features.pkl', 'rb')
    unpickler = pickle.Unpickler(train_pca)
    while True:
        try:
            train_img_features.extend(unpickler.load())
        except EOFError:
            break
    test_img_features = []
    test_pca = open('test_pca_features.pkl', 'rb')
    unpickler = pickle.Unpickler(test_pca)
    while True:
        try:
            test_img_features.extend(unpickler.load())
        except EOFError:
            break
    X_train, X_val, y_train, y_val, test_data = feature_pooling(train_img_features, train_id_list, train_photo_to_biz,
                                                                test_img_features, test_id_list, test_photo_to_biz,
                                                                Y, "pca")

    ensembled_preds_resnet_plus_1024,preds_mlp_resnet_plus_1024,preds_lr_resnet_plus_1024 = ensemble_classifier(X_train, X_val, y_train, y_val, test_data)
    with open('ensembled_preds_resnet_plus_1024.pkl', 'wb') as f:
        pickle.dump(ensembled_preds_resnet_plus_1024, f)
    with open('preds_mlp_resnet_plus_1024.pkl', 'wb') as f:
        pickle.dump(preds_mlp_resnet_plus_1024, f)
    with open('preds_lr_resnet_plus_1024.pkl', 'wb') as f:
        pickle.dump(preds_lr_resnet_plus_1024, f)
    end = time.time()
    final_prediction = np.concatenate((final_prediction, ensembled_preds_resnet_plus_1024[np.newaxis,:,:]), axis=0)
    print("Finished predictions with Mixed features with dim= 1024 in ",end-start)
    ############## L2 normalized->Concatenated(resnet(2048), alexnet(500), inceptionV3(748))->PCA(128) ################
    print("Predictions with Mixed features with dim= 128")
    start = time.time()
    train_img_features = []
    train_pca = open('train_pca_features_128.pkl', 'rb')
    unpickler = pickle.Unpickler(train_pca)
    while True:
        try:
            train_img_features.extend(unpickler.load())
        except EOFError:
            break
    test_img_features = []
    test_pca = open('test_pca_features_128.pkl', 'rb')
    unpickler = pickle.Unpickler(test_pca)
    while True:
        try:
            test_img_features.extend(unpickler.load())
        except EOFError:
            break
    X_train, X_val, y_train, y_val, test_data = feature_pooling(train_img_features, train_id_list, train_photo_to_biz,
                                                                test_img_features, test_id_list, test_photo_to_biz,
                                                                Y, "pca")

    ensembled_preds_resnet_plus_128, preds_mlp_resnet_plus_128, preds_lr_resnet_plus_128 = ensemble_classifier(X_train, X_val, y_train, y_val, test_data)
    with open('ensembled_preds_resnet_plus_128.pkl', 'wb') as f:
        pickle.dump(ensembled_preds_resnet_plus_128, f)
    with open('preds_mlp_resnet_plus_128.pkl', 'wb') as f:
        pickle.dump(preds_mlp_resnet_plus_128, f)
    with open('preds_lr_resnet_plus_128.pkl', 'wb') as f:
        pickle.dump(preds_lr_resnet_plus_128, f)

    final_prediction = np.concatenate((final_prediction, ensembled_preds_resnet_plus_128[np.newaxis,:,:]), axis=0)
    end = time.time()
    print("Finished predictions with Mixed features with dim= 128 in ",end-start)
    final_prediction = final_prediction.mean(axis = 0)
    with open('final_prediction.pkl', 'wb') as f:
        pickle.dump(final_prediction, f)

    with open("out.csv","w") as f:
        wr = csv.writer(f)
        wr.writerows(final_prediction)


    ##### Neural Network Training ##############################################
    """
    print("shape of testdata",test_data.shape)

    model = Network_Architecture(768, 9)
    #optimizer = Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
    loss_function = nn.MultiLabelMarginLoss()
    #loss_function = nn.BCELoss()

    if use_gpu:
        model = model.cuda()
    #train_loader = generate_batch(train_data, train_label, "train")
    #test_loader = generate_batch(test_data,None,"test")
    train_data_loader = MyDataset(X_train, y_train)
    print("shape of validation set is:",X_val.shape)
    val_data_loader = MyDataset(X_val, y_val)
    #test_data_loader = MyDataset(test_data,None)
    params = {'batch_size': 200,
              'shuffle': True,
              'num_workers': 6}
    training_generator = data.DataLoader(train_data_loader, **params)
    validation_generator = data.DataLoader(val_data_loader, batch_size= 100)
    testing_generator = data.DataLoader(test_data, batch_size=100)
    print("len of train generator",len(training_generator))
    print("len of validation_generator", len(validation_generator))
    print("len of test_generator", len(testing_generator))
    train(model, loss_function, optimizer,training_generator, validation_generator, 400)
    #val_preds_nn = evaluate(model, validation_generator,"val",X_val.shape[0])
    preds_nn = evaluate(model, testing_generator,"test")

    """

    #with open('final_prediction.pkl', 'rb') as f:
    #    final_prediction = pickle.load(f)
    final_prediction = final_prediction >= 0.2
    write_result_file(final_prediction)
