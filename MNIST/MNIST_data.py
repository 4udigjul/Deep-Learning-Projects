import numpy as np
import pandas as pd
from keras.utils import  np_utils
np.random.seed(10)
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("train_data =",len(x_train))
print("test_data =",len(x_test))
print("x_train:",x_train.shape)
print("y_train:",y_train.shape)

print("x_train[1] =",x_train[1])
print("y_train[1] =",y_train[1])

print("x_test:",x_test.shape)
print("y_test:",y_test.shape)

import  matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

plot_image(x_train[0])



