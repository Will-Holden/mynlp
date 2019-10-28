from mynlp.utils import *
import unittest


class TestUtils(unittest.TestCase):
    def test_is_english_char(self):
        _char = 's'
        self.assertTrue(is_english_char(_char))

    def test_is_chinese_char(self):
        _char = '我'
        self.assertTrue(is_chinese_char(_char))

    def test_check_lang(self):
        _en_text = 'this is a english text'
        _cn_text = '这是一个中文串。'
        self.assertEqual(check_lang(_en_text), 'en')
        self.assertEqual(check_lang(_cn_text), 'cn')
