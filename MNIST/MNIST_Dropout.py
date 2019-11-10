#匯入所需模組
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from keras.optimizers import SGD, Adam
from keras.layers import Dropout

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

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


#print("x_train:", x_train.shape)
#print("y_train:", y_train.shape)
#print("x_test:", x_test.shape)
#print("y_test:", y_test.shape)

x_train_reshape = x_train.reshape(60000, 784).astype(float)
x_test_reshape = x_test.reshape(10000, 784).astype(float)

x_train_normalize = x_train_reshape/255
x_test_normalize = x_test_reshape/255

y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer="normal",
                 activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=10, kernel_initializer="normal",
                 activation="softmax"))
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
train_history = model.fit(x_train_normalize, y_train_onehot, validation_split=0.2, epochs=30, batch_size=200, verbose=2)
results = model.evaluate(x_train_normalize, y_train_onehot)
print("Train Acc = ", results[1])
#Train Acc =  0.9960166666666667
score = model.evaluate(x_test_normalize, y_test_onehot)
print("Test Acc = ", score[1])
#Test Acc =  0.9831

prediction = model.predict_classes(x_test_reshape)
#print(prediction)

plot_image_labels_prediction(x_test,y_test,prediction,idx=1232)
plot_image_labels_prediction(x_test,y_test,prediction,idx=9009)

import pandas as pd
print(pd.crosstab(y_test, prediction, rownames=["label"], colnames=["predict"]))
df = pd.DataFrame({"label":y_test, "predict":prediction})
#print(df[:20])

label = 9
predict = 4
print(df[(df.label==label)&(df.predict==predict)])

label = 7
predict = 2
print(df[(df.label==label)&(df.predict==predict)])

label = 9
predict = 4
print(df[(df.label==label)&(df.predict==predict)])

#8339
#8522