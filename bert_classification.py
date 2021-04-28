"""
A simple example of BERT-based classifier

#Ref: https://swatimeena989.medium.com/bert-text-classification-using-keras-903671e0207d
"""
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import keras
from tqdm import tqdm
import pickle
from keras.models import Model
import keras.backend as K
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import itertools
from keras.models import load_model
from sklearn.utils import shuffle
from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig


data_dir = "/Users/dur-rbaral-m/projects/test_projects/data/"
data_file= os.path.join(data_dir, 'spam.csv')
log_dir = os.path.join(data_dir, "logs")
model_save_path=os.path.join(data_dir, 'bert_classification_model.h5')

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split()
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
    return " ".join(clean_words)

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    return w


data=pd.read_csv(data_file,encoding='ISO-8859-1')

print(data.head())

print(data.shape)

#format columns and removed unused columns
data = data.loc[:, ~data.columns.str.contains('Unnamed: 2', case=False)]
data = data.loc[:, ~data.columns.str.contains('Unnamed: 3', case=False)]
data = data.loc[:, ~data.columns.str.contains('Unnamed: 4', case=False)]

data=data.rename(columns = {'v1': 'label', 'v2': 'text'}, inplace = False)

print('File has {} rows and {} columns'.format(data.shape[0],data.shape[1]))

#clean the data
data=data.dropna()                                                           # Drop NaN valuues, if any
data=data.reset_index(drop=True)                                             # Reset index after dropping the columns/rows with NaN values
data = shuffle(data)                                                         # Shuffle the dataset
print('Available labels: ',data.label.unique())                              # Print all the unique labels in the dataset
data['text']=data['text'].map(preprocess_sentence)                           # Clean the text column using preprocess_sentence function defined above

num_classes=len(data.label.unique())
#create bert model
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=num_classes)

#sample tokenization
sent= 'how to train the model, lets look at how a trained model calculates its prediction.'
tokens=bert_tokenizer.tokenize(sent)
print(tokens)

tokenized_sequence= bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =30,pad_to_max_length = True,
return_attention_mask = True)

#decoding
bert_tokenizer.decode(tokenized_sequence['input_ids'])

#binarize labels
data['gt'] = data['label'].map({'ham':0,'spam':1})
sentences=data['text']
labels=data['gt']

#load the sentences into Bert-Tokenizer
input_ids=[]
attention_masks=[]

for sent in sentences:
    bert_inp=bert_tokenizer.encode_plus(sent, add_special_tokens = True,max_length =64,pad_to_max_length = True,return_attention_mask = True)
    input_ids.append(bert_inp['input_ids'])
    attention_masks.append(bert_inp['attention_mask'])

input_ids=np.asarray(input_ids)
attention_masks=np.array(attention_masks)
labels=np.array(labels)

#train-test split
train_inp,val_inp,train_label,val_label,train_mask,val_mask=train_test_split(input_ids,labels,attention_masks,test_size=0.2)

print('Train inp shape {} Val input shape {}\nTrain label shape {} Val label shape {}\nTrain attention mask shape {} Val attention mask shape {}'.format(train_inp.shape,val_inp.shape,train_label.shape,val_label.shape,train_mask.shape,val_mask.shape))

#configure the model

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True),keras.callbacks.TensorBoard(log_dir=log_dir)]

print('\nBert Model',bert_model.summary())

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

bert_model.compile(loss=loss,optimizer=optimizer,metrics=[metric])

#train the model
history=bert_model.fit([train_inp,train_mask],train_label,batch_size=32,epochs=4,validation_data=([val_inp,val_mask],val_label),callbacks=callbacks)


trained_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
trained_model.load_weights(model_save_path)

preds = trained_model.predict([val_inp,val_mask],batch_size=32)
print("preds are ",preds)
print("logits ",preds.logits)
pred_labels = preds.argmax(axis=1)
f1 = f1_score(val_label,pred_labels)
print('F1 score',f1)
print('Classification Report')
print(classification_report(val_label,pred_labels,target_names=list(set(labels))))

print('Training and saving built model.....')

#get the validation parameters
trained_model.evaluate([val_inp,val_mask],batch_size=32)