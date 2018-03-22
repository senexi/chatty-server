import nltk

sentence="Das ist ein toller Tag und ich glaube ich mache einen Spaziergang, draussen unten im Park!"

# NLTK's default German stopwords
default_stopwords = set(nltk.corpus.stopwords.words('german'))

revelant_words=["drinnen", "draussen" , "innen", "drin", "raus"]

all_stopwords = default_stopwords 

train_set = [("Das ist ein toller Tag und ich glaube ich mache einen Spaziergang, draussen unten im Park!", "outside"),
("Lass uns nach draussen gehen", "outside"),
("Draussen ist es total super", "outside"),
("Ich bleibe drinnen.", "inside"),
("Hier Drinnen ist es viel schöner.", "inside"),
("Wenn ich drinnen bin, geht es mir gut.", "inside"),
("Gehe in den Todesstern rein.", "inside")
]

def get_word_frequency(text):
    words = nltk.word_tokenize(sentence)
    # Remove single-character tokens (mostly punctuation)
    words = [word for word in words if len(word) > 1]
    # Remove numbers
    words = [word for word in words if not word.isnumeric()]
    # Lowercase all words (default_stopwords are lowercase too)
    words = [word.lower() for word in words]
    # Remove stopwords
    words = [word for word in words if word not in all_stopwords]
    # Calculate frequency distribution
    return nltk.FreqDist(words)

def word_features(text):
    features = {}
    #features["first_letter"] = name[0].lower()
    fdist = get_word_frequency(text)
    for word, count in fdist.items():
        print(word, count)
        if word in revelant_words:
            features["count({})".format(word)]= count
    return features

featuresets = [(word_features(text), category) for (text, category) in train_set]

classifier = nltk.NaiveBayesClassifier.train(featuresets)

classification = classifier.classify(word_features('Ich mache ein Tour zum Todesstern!'))
dist = classifier.prob_classify(word_features('Ich mache ein Tour zum autohaus!'))
classifier.show_most_informative_features(20)
print(classification)
print(dist.prob(dist.max()))
