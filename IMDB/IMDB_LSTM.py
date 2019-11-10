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
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer

model = Sequential()
model.add(Embedding(output_dim = 32, input_dim = num_words, input_length = maxlen))
model.add(Dropout(0.25))
model.add(LSTM(32))
model.add(Dense(units=256,
                activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation= "sigmoid"))
model.summary()

model.compile(loss = "binary_crossentropy", optimizer="adam", metrics = ["accuracy"])
train_history = model.fit(x_train, y_train, batch_size = batch_size, epochs= epochs, verbose=2, validation_split=0.2)

scores = model.evaluate(x_test, y_test, verbose=1)
print("accuracy:",scores[1])
#accuracy: 0.85308

#進行預測
input_seq = "Prepare to be devastated by Joker. Not so much by the intense madness and blood-spewing violence that is sometimes hard to watch, or the overwhelming central performance " \
            "by Joaquin Phoenix in the title role, but by the vision and artistry of the film itself. Even if you hate it, it’s unlike anything you’ve ever seen before—like waking " \
            "up next to a poisonous snake nestled on your blanket, poised and ready to strike. You’re horrified but unable to move. Regardless of my mixed feelings, I think it’s the" \
            " best film about the psychological effect of violence as pop art since Stanley Kubrick’s A Clockwork Orange. Vigorously directed by Todd Phillips, who also co-wrote the " \
            "unique if uneven screenplay with Scott Silver, and beautifully shot by Lawrence Sher, Joker traces the history of the popular DC Comics villain and Batman’s arch enemy in" \
            " totally original terms. After award-winning performances by Jack Nicholson and Heath Ledger in the role, you may think you know the Joker, but who is he and where did he" \
            " come from?In this film, his roots are clearly and hair-raisingly defined. Born Arthur Fleck, he’s a mentally deranged social reject with a history of insanity, which he " \
            "shares with the weird mother he lives with who tried to burn him alive as a child (another unsettling, creepy triumph by Frances Conroy). In the past, both mother and son " \
            "have spent time in the same mental asylum. Now they share one common bond: a passion for watching Murray Franklin, a nightly TV talk-show host played by Robert De Niro.ck, " \
            "he is, needless to say, never the same. Director Phillips wastes no time getting straight to the chase. In fact, the movie opens with a dire premonition of things to come when " \
            "Arthur is smashed in the face with a wooden sign and nearly kicked to death by a gang of hoodlums. It gets worse from there. When he isn’t killing businessmen on the subway or " \
            "struggling to be a standup comic in empty clubs, Arthur becomes a vigilante, joining the underground forces in corrupted, criminal-infested Gotham City. One of his victims is " \
            "the wealthy politician running for mayor, Thomas Wayne, who Arthur’s delusional mother believes is the father who abandoned them both, prompting the Joker to stalk Wayne’s " \
            "son Bruce, who grows up to be Batman.Many homicidal acts of revenge ensue, including, at last, one that will knock you out of your socks when the infamous Joker finally gets " \
            "his big chance as a guest star on a “live” network broadcast of his hero Murray Franklin’s talk show. At the risk of revealing too much, I will say no more. This is one " \
            "movie you have to experience for yourself.  It’s comic-book fantasy is so feverishly close to today’s deranged tabloid news that I began to wonder if the Joker might be s" \
            "omewhere in the theater planning his next move. Every time you think no creature so vile could ever exist in real life, along comes another headline.I can’t tell you how it " \
            "ends, but what I can tell you is Frank Sinatra singing “Send In The Clowns” adds some badly needed humor, the cinematography is so incredible that the camera becomes an " \
            "important character in the middle of all the action, and the schizophrenic performance by Phoenix blazes like a bonfire." \
            "Joker is most definitely not a movie for everybody, but in the greatest performance of his career, Phoenix is electrifying." \
            " Weeping, shrieking, dragged screaming through police stations and mental asylums, then pausing after each evil slaughter to dance balletic tour jetés, he’s a cross between " \
            "Jacques D’Amboise’s Prince Siegfried in Swan Lake and James Cagney’s Cody Jarrett in White Heat. As a sick, twisted failure in life who takes his torment out on the rest of " \
            "the world, he reveals the soul of a monster in Hell, in a movie that borders on genius—repellant, dark, terrifying, disgusting, brilliant and unforgettable."

token = Tokenizer(num_words = num_words)

input_seq = token.texts_to_sequences([input_seq])
input_seq = sequence.pad_sequences(input_seq,maxlen = maxlen)
predict_results = model.predict_classes(input_seq)

print(predict_results[0][0])
