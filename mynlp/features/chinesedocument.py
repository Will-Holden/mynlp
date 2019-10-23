from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.sklearn_api import W2VTransformer
from mynlp.preprocess import chinese_sentence

class ChineseDocument:
    def __init__(self, text):
        self._document = chinese_sentence(text)

    def _tfidf(self):
        self._tfidf_vectorizer = TfidfVectorizer()
        self._tfidf_feature = self._tfidf_vectorizer.fit_transform(self._document.corpus)
        return self._tfidf_feature

    def _word2vec(self):
        # self._w2vmodel = 
        pass

    @property
    def tfidf_feature(self):
        try:
            if not self._tfidf_feature:
                self._tfidf()
        except AttributeError:
            self._tfidf()
        return self._tfidf_feature

    @property
    def tfidf_vectorizer(self):
        try:
            if not self._tfidf_vectorizer:
                self._tfidf()
        except AttributeError:
            self._tfidf()
        return self._tfidf_vectorizer