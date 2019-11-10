#匯入所需模組
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
1
np.random.seed(10)

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap = "binary")
    plt.show()

#plot_image(x_train[0])
#plot_image(x_test[0])

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
model.add(Dense(units=10, kernel_initializer="normal",
                 activation="softmax"))

#print(model.summary())


model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])  #定義訓練模式
train_history = model.fit(x_train_normalize, y_train_onehot, validation_split=0.2, epochs=30, batch_size=200, verbose=2)  #訓練模型   #驗證資料過高降低準確度
results = model.evaluate(x_train_normalize, y_train_onehot) #評估訓練資料準確度
print("Train Acc = ", results[1])
#Train Acc =  0.9959666666666667
score = model.evaluate(x_test_normalize, y_test_onehot) #評估測試資料準確度
print("Test Acc = ", score[1])
#Test Acc =  0.978

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel("train")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

#show_train_history(train_history, "acc", "val_acc")

prediction = model.predict_classes(x_test_reshape)  #執行測試資料(reshape)預測
#print(prediction)

plot_image_labels_prediction(x_test,y_test,prediction,idx=9009)


import pandas as pd
print(pd.crosstab(y_test, prediction, rownames=["label"], colnames=["predict"]))

df = pd.DataFrame({"label":y_test, "predict":prediction})
#print(df[:20])

label = 7
predict = 2
print(df[(df.label==label)&(df.predict==predict)])

label = 2
predict = 8
print(df[(df.label==label)&(df.predict==predict)])

#8339
#8522