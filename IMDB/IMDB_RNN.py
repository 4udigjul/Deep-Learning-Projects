from keras.datasets import imdb
from keras.preprocessing import sequence

#maxlen = 100
#num_words=20000
#batch_size = 100
#epochs = 10
#scores = 0.82524

maxlen = 500
num_words=20000
batch_size = 100
epochs = 10
#scores = 0.86416



(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

#print(x_train[0])



x_train = sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = sequence.pad_sequences(x_test, maxlen = maxlen)


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from  keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN

model = Sequential()
model.add(Embedding(output_dim = 32, input_dim = num_words, input_length = maxlen))
model.add(Dropout(0.25))
model.add(SimpleRNN(units=16))
model.add(Dense(units=256,
                activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation= "sigmoid"))
model.summary()

model.compile(loss = "binary_crossentropy", optimizer="adam", metrics = ["accuracy"])
train_history = model.fit(x_train, y_train, batch_size = batch_size, epochs= epochs, verbose=2, validation_split=0.2)

scores = model.evaluate(x_test, y_test, verbose=1)
print("accuracy:",scores[1])
#accuracy: 0.8439