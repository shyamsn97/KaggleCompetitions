# KaggleCompetitions
Various Kaggle Competitions such as:

Kaggle Competition: IEEE's Signal Processing Society - Camera Model Identification

The purpose of this project was to use a data set of pictures taken from a variety of phones in many different locations and ultimately predict, for a new photo, what phone/device it was taken from. My approach was to use a Convolutional neural net, based on the idea that different devices probably leave different local noise patterns that differentiate their photos from others.

-Utilized opencv to manipulate the image data.
-Constructed and implemented a Multi-Channelled Convolutional Neural Network using keras and tensorflow.

Kaggle competition link: https://www.kaggle.com/c/sp-society-camera-model-identification

-------------------------------------------------------------------------------------------------------

Kaggle Competition: Classifying Comments

The purpose of this project was to use a dataset of comments to identify patterns in varying classes of "toxic" comments and ultimately create a model that can accurately classify a comment as either toxic, severe toxic, threat, insult, identity hate, or clean. My approach was motivated by the idea that words are somewhat independent from each other and we can model the conditional distribution of comments as a multinomial distribution.

-Constructed a Bayesian network using word frequencies to estimate conditional probabilities that describe relationships between unique words and types of comments.
-Constructed a multinomial logistic regression model to classify the comments as  toxic, severe toxic, threat, insult, identity hate, or clean.

Kaggle competition link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

-------------------------------------------------------------------------------------------------------

Kaggle Competition: Titanic-Machine Learning From Disaster

The purpose of this project was to use the Titanic dataset to generate a model to understand what characteristics of passengers, such as ticket costs, age, etc., affected their chances of survival, and use this data to accurately predict survival.

-Performed EDA on the popular Titanic dataset, utilizing statistical modelling techniques and feature selection/engineering to generate useful input data.
-Generated multiple statistical models to analyze correlations and reveal underlying characteristics.
-Constructed and implemented a density based anomaly detection using K-means.
-Using sci-kit-learn, implemented an ensemble model that used a combination of Random Forests, Gradient Boosting,  AdaBoost, and SVM.
-Constructed and implemented a Deep Neural Network Classifier using tensorflow.

