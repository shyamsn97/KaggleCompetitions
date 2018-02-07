
'''
@author Shyam Sudhakaran
This program utilizes a multichannelled convolutional neural network to predict what phone took a picture of the image
'''
# # Camera Identification



import numpy as np
import pandas as pd
import tensorflow as tf
from keras import *
from keras.layers import *
from keras import backend as K
from keras.utils.vis_utils import plot_model
import pydot
import graphviz
import PIL
import cv2
import glob
import os


# Read in data

# In[2]:


def read(img):
    return np.asfarray(PIL.Image.open(img))


# In[3]:


input_path = 'data'
train_path = input_path + '/' + 'train' + '/'
test_path = input_path + '/' + 'test' + '/'
labels = os.listdir(train_path)


# In[4]:


print labels


# In[33]:



def gettraining(labels,path):
    trainlabels = []
    for l in labels:
        for i in os.listdir(path + l + '/'):
            trainlabels.append((path + l + '/'+ str(i),l))

    train = pd.DataFrame(trainlabels, columns=['camera', 'name'])
    return train

def gettest(path):
    test = []
    ids = []
    for i in os.listdir(path):
        test.append(path + i)
        ids.append(i)
    return test, ids
train = gettraining(labels,train_path)
test, ids = gettest(test_path)


# In[6]:


train.head(5)


# In[7]:


test[0:5]


# In[8]:


image = train["camera"][0]
img = cv2.imread(str(image),0)


# In[9]:


print img


# In[10]:


img.shape


# In[11]:


resized_image = np.asfarray(cv2.resize(img, (32, 32)))
#cv2.imshow('image',img)


# Lets flatten the image and divide by the max value(255) and flatten it into a 1d vector, so we can input it into a model

# In[12]:


resized = resized_image.flatten()/255.0


# In[13]:


print resized.shape


# Do the rest for the other pictures

# In[14]:


Xtrain = train["camera"]
y = train["name"]


# In[15]:


def preprocess(path):
    img = cv2.imread(path,0)
    img = np.asfarray(cv2.resize(img,(32, 32)))
    return img.flatten()/255.0

def processbulk(paths):
    return np.asfarray([preprocess(filepath) for filepath in paths])

X_train = processbulk(Xtrain[0:5])

def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    label_index = labels.columns.values
    
    return labels, label_index


# In[16]:


X_train = processbulk(Xtrain)
X_test = processbulk(test)


# In[17]:


y = label_transform(y)


# In[18]:


y = np.array(y[0])


# In[40]:


print y


# ## Multi Channel Convolutional Neural Network
# Because the images probably have minor details, we want to use a convolutional neural network with multiple kernal sizes to account for local dependencies in the images.

# Here we will input the flattened version of grayscale images into a multichannel convolutional neural network, with kernal size of 12,24, and 36 and 32 filters each.

# In[39]:


class MultiChannelCNN:
    
    saved = 0
    
    def __init__(self, X,y,models=[]):
        self.X = np.expand_dims(X, axis=2) #need to add an extra column, 1d convolution needs to "slide" accross
        self.y = y
        self.models = models
        
    #channels is an integer, number of channels
    #kernel size is a list of dimensions for the kernels
    def createmodel(self,channels,kernel_size,num_filters):
        
        K.clear_session()

        inputlayers = {}
        layers = {}
        flats = {}
        length = self.X.shape[1]
        for i in range(channels):
            print i
            inputlayers["input"+ str(i)] = Input(shape=(length,1))
            print inputlayers["input"+str(i)]
            layers["conv" + str(i)] = Conv1D(filters=num_filters,input_shape=(length, 1), kernel_size=kernel_size[i], activation='relu')(inputlayers["input" + str(i)])
            layers["dropout" + str(i)] =  Dropout(0.5)(layers["conv" + str(i)])
            layers["pool" + str(i)] = MaxPooling1D(pool_size=4)(layers["dropout" + str(i)])
            flats["flat" + str(i)] = Flatten()(layers["pool" + str(i)])
        
        merge = concatenate(list(flats.values()))
        dense = Dense(10, activation='relu')(merge)
        outputs = Dense(10, activation='sigmoid')(dense)
        model = Model(inputs=list(inputlayers.values()), outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.models.append(model)
    
    #train using custom params
    def train(self,model,epochs,channels,batch_size):
        
        inp = []
        for i in range(channels):
            inp.append(self.X)
        
#         model.fit(inp, self.y,validation_split=0.1, epochs=epochs, batch_size=batch_size,verbose=1)
        model.fit(inp, self.y,validation_split=0.1, epochs=epochs,verbose=1)

        
        if MultiChannelCNN.saved < 1:
            model.save('multichannelcnn.h5')
        else:
            print("Already Saved")
        loss, acc = model.evaluate([self.X,self.X,self.X], self.y, verbose=0)
        print('Train Accuracy: %f' % (acc*100))
        
        return model
    
    #predict
    def predict(self,model,data):
        
        #model = load_model('multichannelcnn.h5')

        predicts = model.predict(data)

        return predicts 
        


# In[24]:


mcnn = MultiChannelCNN(X_train,y)
mcnn.createmodel(3,[12,24,36],32)


# In[25]:


mcnn.models[0] = mcnn.train(mcnn.models[0],3,3,1)


# In[26]:


print mcnn.models


# In[27]:


newtest = np.expand_dims(X_test, axis=2)
out = mcnn.predict(mcnn.models[0],[newtest,newtest,newtest])


# In[28]:


print X_test


# In[29]:


out


# In[30]:


out
predict = np.argmax(out, axis=1)
predict = [labels[p] for p in predict]


# In[31]:


print predict


# Submit to csv

# In[35]:


df = pd.DataFrame(columns=['fname', 'camera'])
df["fname"] = ids
df["camera"] = predict


# In[36]:


df


# In[38]:


df.to_csv("data/submissions/3channelconvnet.csv", index=False)

