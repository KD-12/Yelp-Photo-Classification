#https://stackoverflow.com/questions/32191219/python-pca-on-matrix-too-large-to-fit-into-memory

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import IncrementalPCA
import time
import pickle


def pca_fit(unpickler, alexnet_features, inception_features):
    print('inside pca fit')
    sklearn_pca = IncrementalPCA(n_components=1024, batch_size=25)
    i = 0
    while True:
        try:
            resnet_features = []
            for j in range(200):
                resnet_features.extend(unpickler.load())
            concat = np.concatenate((preprocessing.normalize(resnet_features),
                            preprocessing.normalize(alexnet_features[i:i+5000]),
                            preprocessing.normalize(inception_features[i:i+5000])), axis=1)
            i += 5000
            print('finished concat of ', i, ' images. Shape ', concat.shape)
            sklearn_pca = sklearn_pca.partial_fit(concat)
            print('finished fit on ', i, ' images')
        except EOFError:
            break
    return sklearn_pca


def pca_transform(unpickler, alexnet_features, inception_features, pickler, sklearn_pca):
    print('inside pca transform')
    i = 0
    while True:
        try:
            resnet_features = []
            for j in range(200):
                resnet_features.extend(unpickler.load())
            concat = np.concatenate((preprocessing.normalize(resnet_features),
                            preprocessing.normalize(alexnet_features[i:i+5000]),
                            preprocessing.normalize(inception_features[i:i+5000])), axis=1)
            i += 5000
            print('finished concat of ', i, ' images')
            concat_transform = sklearn_pca.transform(concat)
            pickler.dump(concat_transform)
            print('finished transform on ', i, ' images. Shape ', concat_transform.shape)
        except EOFError:
            break


if __name__ == "__main__":
    print("Starting PCA for Dimentionality Reduction")
    

    sklearn_pca = IncrementalPCA(n_components=2048) #feature size 2048
    #sklearn_pca = IncrementalPCA(n_components=128) # feature size 128

    ###### Reading Alexnet Pickled Features ####################################
    with open('test_img_features_all.pkl', 'rb') as f:
        test_alexnet_features = pickle.load(f)
    with open('train_img_features_all.pkl', 'rb') as f:
        train_alexnet_features = pickle.load(f)

    ###### Reading Inception V3 Pickled Features ###############################
    with open('train_img_features_all_inception.pkl', 'rb') as f:
        train_inception_features = pickle.load(f)
    with open('test_img_features_all_inception.pkl', 'rb') as f:
        test_inception_features = pickle.load(f)


    #### Train PCA partial fit #################################################
    train_resnet_f = open('pca_test.pkl', 'rb')
    unpickler = pickle.Unpickler(train_resnet_f)
    unpickler.load()
    sklearn_pca = pca_fit(unpickler, train_alexnet_features, train_inception_features)
    train_resnet_f.close()

    print('finished train pca fit')

    ##### Train PCA transform ##################################################
    train_resnet_f = open('iterative_resnet_pickle_train.pkl', 'rb')
    unpickler = pickle.Unpickler(train_resnet_f)
    unpickler.load()

    train_pca = open('train_pca_features.pkl', 'wb')
    pickler = pickle.Pickler(train_pca)
    pca_transform(unpickler, train_alexnet_features, train_inception_features, pickler, sklearn_pca)
    train_resnet_f.close()
    train_pca.close()

    print('finished train pca transform')


    #### Test PCA partial fit ##################################################
    test_resnet_f = open('iterative_resnet_pickle_test.pkl', 'rb')
    unpickler = pickle.Unpickler(test_resnet_f)
    unpickler.load()
    sklearn_pca = pca_fit(unpickler, test_alexnet_features, test_inception_features)
    test_resnet_f.close()

    print('finished test pca fit')

    ##### Test PCA transform ###################################################
    test_resnet_f = open('iterative_resnet_pickle_test.pkl', 'rb')
    unpickler = pickle.Unpickler(test_resnet_f)
    unpickler.load()

    test_pca = open('test_pca_features.pkl', 'wb')
    pickler = pickle.Pickler(test_pca)
    pca_transform(unpickler, test_alexnet_features, test_inception_features, pickler, sklearn_pca)
    test_resnet_f.close()
    test_pca.close()

    print('finished test pca transform')
