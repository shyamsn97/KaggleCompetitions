import pandas as pd 
import numpy as np

import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer



train=pd.read_csv("csv/train.csv")
copytrain = train.copy()
test=pd.read_csv("csv/test.csv")


trainval = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
trainval = trainval.div(trainval.sum(axis=1), axis=0).fillna(0)
train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = trainval

#comments = train["comment_text"]
# print set(comments[1].split())
# vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode')

# train_test_comment_text = vectorizer.fit_transform(comments)
# print train_test_comment_text

def text_to_wordlist(text, remove_stopwords, stem_words=True):
	# Clean the text, with the option to remove stopwords and to stem words.
	special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
	replace_numbers=re.compile(r'\d+',re.IGNORECASE)
	# Convert words to lower case and split them
	text = text.lower().split()
	# Optionally, remove stop words
	if remove_stopwords:
	    stops = set(stopwords.words("english"))
	    text = [w for w in text if not w in stops]

	text = " ".join(text)

	#Remove Special Characters
	text=special_character_removal.sub('',text)

	#Replace Numbers
	text=replace_numbers.sub('n',text)

	# Optionally, shorten words to their stems
	if stem_words:
	    text = text.split()
	    stemmer = SnowballStemmer('english')
	    stemmed_words = [stemmer.stem(word) for word in text]
	    text = " ".join(stemmed_words)

	# Return a list of words
	textlist = list(set(text.split()))
	y = [word for word in textlist if len(word) > 1]

	return(y)



def constructwordlist(traindf,copytrain):
	types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

	comments = traindf["comment_text"]

	commentlist = text_to_wordlist(comments[0],True)

	newdf = traindf[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
	zeros = np.ones(len(commentlist))

	d = {'words':commentlist,'toxic':zeros, 'severe_toxic':zeros, 'obscene':zeros, 'threat':zeros, 'insult':zeros, 'identity_hate':zeros,'clean':zeros}
	wordframe = pd.DataFrame(data=d,columns=['words','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','clean'])
	#print wordframe
	checklist = list(newdf.columns[(newdf > 0).iloc[0]])

	checklist = list(set(checklist) & set(types))
	#print newwords
	wordframe.loc[wordframe["words"].isin(commentlist),checklist] += 1
	#print wordframe
	#[checklist] += 1
	#print wordframe[wordframe["words"].isin(commentlist)][checklist]
	print traindf

	for i in range(len(comments)):
		newwordlist = text_to_wordlist(comments[i],True)
		newwords = [obj for obj in newwordlist if obj not in wordframe["words"].tolist()]
		zeros = np.ones(len(newwords))
		d = {'words':newwords,'toxic':zeros, 'severe_toxic':zeros, 'obscene':zeros, 'threat':zeros, 'insult':zeros, 'identity_hate':zeros,'clean':zeros}
		newwordframe = pd.DataFrame(data=d,columns=['words','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','clean'])
		wordframe = pd.concat([wordframe,newwordframe],axis=0)
		checklist = list(newdf.columns[(newdf > 0).iloc[i]])
		print i
		if len(checklist) < 1:
			wordframe.loc[wordframe["words"].isin(newwordlist),"clean"] += 1
		else:
			checklist = list(set(checklist) & set(types))
			wordframe.loc[wordframe["words"].isin(newwordlist),checklist] += 1
	return wordframe

#stuff = constructwordlist(train,copytrain)


#stuff.to_csv('csv/teststuff.csv', index=False)



