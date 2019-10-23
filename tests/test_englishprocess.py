from mynlp.preprocess import chinese_sentence
from mynlp.preprocess import english_sentence
import unittest
from unittest import main as unittest_main

class TestChineseSentence(unittest.TestCase):
    def setUp(self):
        text = """
        10月21-22日，湖南省农学会组织中国农业科学院、福建省农业科学院、中国水稻研究所、江西农业大学、湖北省农业科学院、湖南农业大学、湖南师范大学、湖南省农业农村厅、湖南省水稻研究所等单位专家，对国家杂交水稻工程技术研究中心(湖南杂交水稻研究中心)在湖南省衡南县云集镇、湘潭市雨湖区及长沙市芙蓉区示范展示的第三代杂交晚稻系列组合进行了现场考察与测产。此次测产验收是我国第三代杂交水稻首次专家测产验收，对于评估第三代杂交水稻产量具有重要意义。22日上午的专家评议会形成以下评议意见
        """
        self.chinese = chinese_sentence(text)
        return super().setUp()

    def test_words(self):
        print(self.chinese.words)
        print(self.chinese.sentences)

    def tearDown(self):
        del self.chinese
        return super().tearDown()

class TestEnglishSentence(unittest.TestCase):

    def setUp(self):
        text = """
        this is a test.
        """
        self.english = english_sentence(text)
        return super().setUp()

    def test_words(self):
        print(self.english.words)
        print(self.english.stems)
        print(self.english.sentences)

    # def tearDown(self):
    #     del self.english
    #     return super().tearDown()

if __name__ == "__main__":
    # test_chinese = TestChineseSentence()
    # test_chinese.__main__()
    test_english = TestEnglishSentence()
    test_english.___main__()