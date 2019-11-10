from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
(x_train, y_train,), ( x_test, y_test) = cifar10.load_data()
#print("x_train_len:", len(x_train))
#print("x_test_len:", len(x_test))
#print(x_train.shape)
#print(x_train[0])
#print(x_test[0])

dict = {0:"airplane", 1:"automobile", 2:'bird', 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}


import matplotlib.pyplot as plt
#def plot_prediction(images, labels, prediction, idx, num = 10):
#     fig = plt.gcf()
#     fig.set_size_inches(12, 14)
#    if num > 25: num = 25
#    for i in range (0, num):
#        ax = plt.subplot(5,5,1+i)
#        ax.imshow(images[idx], cmap = "binary")

#        title = str(idx)+ "," + dict[labels[idx][0]]
#        if len(prediction) > 0:
#            title+="=>"+dict[prediction[idx]]

#        ax.set_title(title, fontsize = 10)
#        ax.set_xticks([]);ax.set_yticks([])
#        idx += 1
#    plt.show()

#plot_prediction(x_train, y_train, [] ,20)

x_train_normalize = x_train.astype("float")/255
x_test_normalize = x_test.astype("float")/255
#print(x_train_normalize[0][0][0])

from keras.utils import np_utils
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)
#print(y_train_onehot.shape)
#print(y_test_onehot.shape)
#print(y_train_onehot[:5])
#print(y_test_onehot[:5])

#建立模型 #Dropout 模型結構改變?
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (32,32,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = (32,32,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(2500, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1500, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10,  activation = "softmax"))

#訓練方式 optimizer?
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
train_history = model.fit(x_train_normalize, y_train_onehot,validation_split = 0.2, epochs = 50, batch_size = 128,verbose = 1 )

scores = model.evaluate(x_test_normalize, y_test_onehot, verbose = 0)
print(scores[1])

def plot_image_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(14,14)
    if num>25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i+1)
        ax.imshow(images[idx], cmap= 'binary')
        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ", prediction=" + str(prediction[idx])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        idx += 1
    plt.show()

prediction = model.predict_classes(x_test_normalize)


plot_image_labels_prediction(x_test_normalize, y_test_onehot, prediction, 0)


import pandas as pd
print(dict)
print(pd.crosstab(y_test.reshape(-1), prediction, rownames=["labels"], colnames=["predictions"]))