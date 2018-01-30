import pandas as pd 
import numpy as np

import preprocess 


words=pd.read_csv("csv/teststuff.csv")
train=pd.read_csv("csv/train.csv")

trainval = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
trainval = trainval.div(trainval.sum(axis=1), axis=0).fillna(0)
train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = trainval

types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','clean']
cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


y_train = train.iloc[:, 2:]
y_train['clean'] = 1 - y_train.sum(axis=1) >= 1  
kinds, counts = zip(*y_train.sum(axis=0).iteritems())
counts = np.asfarray(counts)
counts = counts/sum(counts)

# words = generate(words)
# probs = counts*words.loc[:,"toxic":]


def mult(l):
	ans = 1
	for i in range(len(l)):
		ans = l[i] * ans
	return ans


def conditionalindep(df):
	columns = list(df.columns)
	probs = []
	for i in range(len(columns)):
		prob = mult(df[columns[i]].tolist())
		probs.append(prob)
	probs = np.asfarray(probs)
	return probs

def generate(probdf):
	print probdf.sum(axis=0).tolist()[0:]
	probdf[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','clean']] = (probdf[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','clean']]/probdf.sum(axis=0).tolist()[0:]).fillna(0)
	return probdf

def compute_probs(probs,comment):
	types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','clean']
	wordlist = preprocess.text_to_wordlist(comment,True)
	probabilities = conditionalindep(probs.loc[probs["words"].isin(wordlist),types])
	return probabilities

def trainprobabilities(wordscsv,traincsv):
	words=pd.read_csv(wordscsv)
	train=pd.read_csv(traincsv)

	trainval = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
	trainval = trainval.div(trainval.sum(axis=1), axis=0).fillna(0)
	train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = trainval

	types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','clean']
	cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


	y_train = train.iloc[:, 2:]
	y_train['clean'] = 1 - y_train.sum(axis=1) >= 1  
	train['clean'] = y_train['clean'].astype(int)
	kinds, counts = zip(*y_train.sum(axis=0).iteritems())
	counts = np.asfarray(counts)
	counts = counts/sum(counts)

	words = generate(words)
	probs = words.copy()
	probs[types] = counts*words.loc[:,"toxic":]
	trainprobs = train.copy()
	trainprobs = trainprobs.loc[:,"id":]
	
	for index, row in trainprobs.iterrows():
		vals = compute_probs(probs,trainprobs.loc[index,"comment_text"])
		trainprobs.loc[index,"toxic":] = vals
		print index

	probtrain = trainprobs[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','clean']]
	probtrain = probtrain.div(probtrain.sum(axis=1), axis=0).fillna(0)
	trainprobs.loc[:,"toxic":] = probtrain
	return trainprobs 


a = trainprobabilities("csv/teststuff.csv","csv/train.csv")
a.to_csv('csv/testprobs.csv', index=False)

