from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from  keras.layers.embeddings import Embedding

model = Sequential()
model.add(Embedding(output_dim = 32, input_dim = num_words, input_length = maxlen))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=256,
                activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation= "sigmoid"))
model.summary()

model.compile(loss = "binary_crossentropy", optimizer="adam", metrics = ["accuracy"])
train_history = model.fit(x_train, y_train, batch_size = batch_size, epochs= epochs, verbose=2, validation_split=0.2)

scores = model.evaluate(x_test, y_test, verbose=1)
print("accuracy:",scores[1])

token = Tokenizer(num_words = num_words)
SentimentDict = {1:"正面評價", 0:"負面評價"}

def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    input_seq = sequence.pad_sequences(input_seq, maxlen=maxlen)
    predict_results = model.predict_classes(input_seq)
    print(SentimentDict[predict_results[0][0]])

#https://observer.com/2019/10/joker-review-joaquin-phoenix-rex-reed/
predict_review("Prepare to be devastated by Joker. Not so much by the intense madness and blood-spewing violence that is sometimes hard to watch, or the overwhelming central performance " \
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
)

predict_review("Until now, the only review of this movie that I 100% agree with is the review from The Times. I stole the title of this review because it just hits the nail on the head.In the first "
               "impulse, I didn't even want to watch this movie, because I'm just fed up with the Joker. DC has many, much more interesting villains than Joker. It's time to tell stories about them"
               " and not constantly repeat the Joker's story, where there is no chance to say anything new, because in fact the Joker is not an interesting story at all. A rejected, angry man, who "
               "fails in life, a clown who becomes a killer - history as old as the world.Everything that could be said about Joker, said Nolan in the Dark Knight, plus various comics and yes - "
               "Batman and Joker cartoons too. Nolan also told the Joker story more subtly than this movie, Nolan's dialogues are much better, simply in their own league. People who, I already see, "
               "are beginning to claim in the reviews, that it is only this film that fully reflects the depth of the Joker character, or that it is such a successful '' artistic '' film - they "
               "simply try to cast a spell on reality, giving the film a meaning that doesn't exist .I know what Phoenix tried to do here - he chose the form of 'performance' as acting, that is, "
               "there is performance in the performance. The depth was supposed to be inside, between the lines, but unfortunately the effect is quite different - the whole film, like this role, is "
               "a Russian babushka: empty inside, you can open subsequent layers and there is nothing, no message, there is just another empty doll, which in the movie is just another empty, "
               "technical exercise in acting and filmmaking.I will quote the Times review again, because it is excellent at exposing the shallowness and pseudo-artistic nature of this film : ''"
               "Joker - which was written by Phillips and Scott Silver - doesn't have a plot; it's more like a bunch of reaction GIFs strung together. When Arthur gets fired from his clown job, "
               "he struts by the time-clock, deadpans, Oh no, I forgot to punch out and then, wait for it, socks it so hard it dangles from the wall. Make a note of the moment, because you'll "
               "be seeing it a lot in your Twitter and Facebook feeds.The movie's cracks - and it's practically all cracks - are stuffed with phony philosophy. Joker is dark only in a stupidly "
               "adolescent way, but it wants us to think it's imparting subtle political or cultural wisdom. Just before one of his more violent tirades, Arthur muses, Everybody just screams at"
               " each other. Nobody's civil anymore.Who doesn't feel that way in our terrible modern times? But Arthur's observation is one of those truisms that's so true it just slides off the wall, "
               "a message that both the left and the right can get behind and use for their own aims. It means nothing.''I could leave it here, but I'll add one more thing : what was great about "
               "Nolan was that Nolan was able to combine the features of commercial, mainstream and independent film, and what he said was authentic. This accusation, which Nolan haters often say - "
               "that Nolan is pretentious is not true, especially in Dark Knight trilogy (because I can agree that Inception is terribly pretentious). Whereas Joker by Todd Philiphs is the quintessence"
               " of pretentiousness. There is nothing authentic in this movie. This is a collection of cheap fanfiction, quotes from movies like Taxi Driver, poured with cheap sauce of pop culture "
               "tricks. Phoenix battered body twists and turns on the screen, but there is no soul in this creation. This role is empty, just like the whole movie. Time to say goodbye to Joker, DC. "
               "You have much better villains in your stable, whose story can be fascinated to tell. The Joker ceased to be interesting long time ago. Btw, Batman has always been a more interesting "
               "character to me, because although he is a superhero, he is also the only such a popular superhero who has so much darkness, killer impulse, but also humanity from his father inside "
               "(and Bale showed all of this masterfully in Nolan trilogy). The ambivalence is fascinating here. The Joker is a wounded clown, nihilist, murderer. This story is told in the cinema "
               "several times a year. There is nothing fresh or interesting about it.")