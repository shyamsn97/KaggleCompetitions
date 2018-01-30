import pandas as pd 
import numpy as np

import preprocess
import generate_prob
#multinomial logistic regression



def thetax(probs,weights):
	return probs.dot(weights)

def softmax(thetavals):
	soft = np.exp(thetavals)
	softmax = soft.div(soft.sum(axis=1), axis=0).fillna(0)
	return softmax

def gradient(softmax,probs,yvals):
	m = np.array(yvals.loc[:,"toxic"] - softmax.loc[:,"toxic"])
	x = probs.mul(m,axis=0)
	grad = np.asfarray(-1*x.sum(axis=0))

	columns = list(softmax.columns)[1:]
	for i in range(columns):
		m = np.array(yvals.loc[:,columns[i]] - softmax.loc[:,columns[i]])
		x = probs.mul(m,axis=0)
		newgrad = np.asfarray(-1*x.sum(axis=0))
		grad = np.column_stack((grad,newgrad))

	return grad

def cross_entropy(output,softmaxlay):
	return -1*(np.log(softmaxlay).mul(output,axis=0).sum(axis=1).sum(axis=0))

def descent(probs,yvals,softmaxlayer,weights,alpha,decision):
	cross_entropy = cross_entropy(yvals,softmaxlayer)
	entrop = cross_entropy+2
	theta = thetax(probs,weights)
	soft = softmax(thetavals)
	gradients = gradient(soft,probs,yvals)
	while abs(cross_entropy - entrop) >= decision:
		entrop = cross_entropy
		weights = weights - alpha*gradients
		theta = thetax(probs,weights)
		soft = softmax(thetavals)
		gradients = gradient(soft,probs,yvals)
		cross_entropy = cross_entropy(yvals,softmax)
	return weights

def foward_prop(test,weights,probs):
	words = testwordlist["comment_text"]
	for i in range(len(words)):
		probabilities = compute_probs(probs,words[i])
		probs = thetax(probabilities,weights)
		probs = softmax(probs)
		probs = probs[:(len(probs)-1)]
		test.loc[i,"toxic":] = probs
	return test


words=pd.read_csv("csv/testprobs.csv")
probs = words.loc[:,"toxic":]
soft = np.exp(probs)
softmax = soft.div(soft.sum(axis=1), axis=0).fillna(0)

train = pd.read_csv("csv/train.csv")
trainval = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
trainval = trainval.div(trainval.sum(axis=1), axis=0).fillna(0)
train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = trainval

y_train = train.iloc[:, 2:]
y_train['clean'] = 1 - y_train.sum(axis=1) >= 0.98  
train['clean'] = y_train['clean'].astype(int)
