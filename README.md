# Yelp-Photo-Classification
Problem Definition:

Given photos that belong to a business and asked to predict the business
attributes which are our Labels.

0: good_for_lunch
1: good_for_dinner
2: takes_reservations
3: outdoor_seating
4: restaurant_is_expensive
5: has_alcohol
6: has_table_service
7: ambience_is_classy
8: good_for_kids

Data Set can be taken from: https://www.kaggle.com/c/yelp-restaurant-photo-classification/data

Approach:

Step 1: Feature Generation (feature_extraction_alexnet_resnet.py and feature_extraction_inception.py)

Step 2: Iterative PCA (PCA.py)

Step 3: Feature pooling 

Step 4: Ensemble Classifier (neural_training_ensemble.py)
