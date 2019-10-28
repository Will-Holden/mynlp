# from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.sklearn_api import W2VTransformer
from gensim import models
from mynlp.preprocess import sentence
from collections import defaultdict, Counter
from gensim import corpora
from gensim import matutils

class Document:
    def __init__(self, text, word2vec_size=10, word2vec_mint_count=1, word2vec_seed=1, num_topic=100):
        self._document = sentence(text)
        self.word2vec_size = word2vec_size
        self.word2vec_mint_count = word2vec_mint_count
        self.word2vec_seed = word2vec_seed
        self.num_topic = num_topic

    def _gen_corpus(self, min_count=1):
        words = self._document.words
        frequency = Counter([word for sentence in words for word in sentence])
        texts = [
            [token for token in text if frequency[token] > min_count]
            for text in words
        ]
        self._dictionary = corpora.Dictionary(texts)
        self._corpus = [self._dictionary.doc2bow(text) for text in texts]

    def _tfidf(self):
        self._tfidfmodel = models.TfidfModel(corpus=self.corpus)

    def _word2vec(self):
        self._w2vmodel = models.Word2Vec(self._document.words, size=self.word2vec_size, min_count=self.word2vec_mint_count, seed=self.word2vec_seed)

    def _lsi(self):
        self._lsimodel = models.LsiModel(self.corpus,id2word=self._dictionary, num_topics=self.num_topic)

    def _lda(self):
        self._ldamodel = models.LdaModel(self.corpus, id2word=self._dictionary, num_topics=self.num_topic)

    @property
    def corpus(self):
        try:
            if not self._corpus:
                self._gen_corpus()
        except AttributeError:
            self._gen_corpus()
        return self._corpus

    @property
    def dictionary(self):
        try:
            if not self._dictionary:
                self._gen_corpus()
        except AttributeError:
            self._gen_corpus()
        return self._dictionary

    @property
    def word2vec_model(self):
        """ word2vec model
        """
        try: 
            if not self._w2vmodel:
                self._word2vec()
        except AttributeError:
            self._word2vec()
        return self._w2vmodel

    @property
    def tfidf_model(self):
        try:
            if not self._tfidfmodel:
                self._tfidf()
        except AttributeError:
            self._tfidf()
        return self._tfidfmodel

    @property
    def lsi_model(self):
        try:
            if not self._lsimodel:
                self._lsi()
        except AttributeError:
            self._lsi()
        return self._lsimodel

    @property
    def lda_model(self):
        try:
            if not self._ldamodel:
                self._lda()
        except AttributeError:
            self._lda()
        return self._ldamodel

    def word2vec(self, words):
        """ word2vec 转换函数
        """
        try:
            if not self._w2vmodel:
                self._word2vec()
        except AttributeError:
            self._word2vec()
        return self._w2vmodel[words]

    def save_word2vec_model(self, path):
        self.word2vec_model.save(path)

    def load_word2vec_model(self, path):
        self.word2vec_model.load(path)

    def bow(self, text):
        """covert text to bow
        """
        _document = sentence(text)
        return [self.dictionary.doc2bow(sentence) for sentence in _document]
    
    def tfidf(self, text):
        """cover text to tfidf feature
        """
        result = self.tfidfmodel[self.bow(text)]
        return self.to_numpy(result)

    def save_tfidf_model(self, path):
        self.tfidf_model.save(path)

    def load_tfidf_model(self, path):
        self._tfidfmodel = models.TfidfModel.load(path)

    def lsi(self, text):
        result = self.lsi_model[self.bow(text)]
        return self.to_numpy(result)

    def save_lsi_model(self, path):
        self.lsi_model.save(path)
    
    def load_lsi_model(self, path):
        self._lsimodel = models.LsiModel.load(path)

    def lda(self, text):
        result = self.lda_model[self.bow(text)]
        return self.to_numpy(result)

    def save_lda_model(self, path):
        self.lda_model.save(path)
    
    def load_lda_model(self, path):
        self._ldamodel = models.LdaModel.load(path)

    def to_numpy(self, array, terms_num=None):
        if not terms_num:
            return matutils.corpus2dense(array, self.dictionary.__len__()).T
        else:
            return matutils.corpus2dense(array, terms_num)