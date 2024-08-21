import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import fashion_mnist
(train_data,train_labels), (test_data,test_labels) = fashion_mnist.load_data()

def get_traindata():
    return train_data

#from tf.keras.datasets import fashion_mnist


#df_train = pd.read_csv('~/kaggle/fashionmnist/fashion-mnist_train.csv')
#df_test = pd.read_csv('~/kaggle/fashionmnist/fashion-mnist_test.csv')
#print(f"Training sample:\n{train_data[0]}\n")
print(f"Training label: {train_labels[0]}")

print(f'Shape of the train data:{train_data.shape}')
print(f'Shape of a train label: {train_labels[0].shape}')
print(f'Shape of a training sample: {train_data[0].shape}')


classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']


plt.imshow(train_data[6],cmap='gray')
plt.title(f"Encoded label: {train_labels[6]};  'Real' label: {classes[train_labels[6]]}") 
plt.show()