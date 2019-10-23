import nltk
from mynlp.preprocess.englishprocess import EnglishSentence
from mynlp.preprocess.chineseprocess import ChineseSetence

__all__ = ['english_sentence', 'chinese_sentence']

def english_sentence(text):
    return EnglishSentence(text)

def chinese_sentence(text):
    return ChineseSetence(text)