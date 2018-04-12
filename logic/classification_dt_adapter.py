from __future__ import unicode_literals
from chatterbot.logic import LogicAdapter
from .classification_data import train_set, responses, keywords
from chatterbot.conversation import Statement
import difflib
import nltk



class ClassificationDTAdapter(LogicAdapter):
    """
    A logic adapter that returns a response based on known responses to
    the closest matches to the input statement.
    """
    default_stopwords = set(nltk.corpus.stopwords.words('german'))

    def __init__(self, **kwargs):
        super(ClassificationDTAdapter, self).__init__(**kwargs)
        featuresets = [(self.create_features(text), category) for (text, category) in train_set]
        print('start training')
        self.classifier = nltk.DecisionTreeClassifier.train(featuresets)
        print('training complete')

    def preprocess_bow_freq(self, input_statement):
        words = nltk.word_tokenize(input_statement)
        # Remove single-character tokens (mostly punctuation)
        words = [word for word in words if len(word) > 1]
        # Remove numbers
        words = [word for word in words if not word.isnumeric()]
        # Lowercase all words (default_stopwords are lowercase too)
        words = [word.lower() for word in words]
        # Remove stopwords
        words = [word for word in words if word not in self.default_stopwords]
        return words
    
    def create_features(self, input_statement):
        bow = self.preprocess_bow_freq(input_statement)
        features = {}
        for keyword in keywords:
            features["has({})".format(keyword)] = False
            for word in bow:
                kw = difflib.get_close_matches(word, keywords)
                if len(kw) >= 1 and kw[0] == keyword:
                    features["has({})".format(keyword)] = True
        return features

    def can_process(self, statement):
        return True

    def process(self, input_statement):
        self.response_statement = Statement(text="I don't know anything!")
        self.response_statement.confidence = 0.0
        input_set = self.create_features(input_statement.text)
        label = self.classifier.classify(input_set)
        print(input_set)
        confidence = nltk.classify.accuracy(self.classifier, [(input_set,label)])
        self.response_statement.confidence = confidence
        self.response_statement.text = responses[label]
        print("response statement is '{}' with a confidence of {} with Decision Tree classifier".format(self.response_statement, confidence))
        return self.response_statement