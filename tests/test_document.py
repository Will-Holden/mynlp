from mynlp.features import Document
import unittest

class TestDocument(unittest.TestCase):
    def setUp(self):
        text = """
        10月21-22日，湖南省农学会组织中国农业科学院、福建省农业科学院、中国水稻研究所、江西农业大学、湖北省农业科学院、湖南农业大学、湖南师范大学、湖南省农业农村厅、湖南省水稻研究所等单位专家，对国家杂交水稻工程技术研究中心(湖南杂交水稻研究中心)在湖南省衡南县云集镇、湘潭市雨湖区及长沙市芙蓉区示范展示的第三代杂交晚稻系列组合进行了现场考察与测产。此次测产验收是我国第三代杂交水稻首次专家测产验收，对于评估第三代杂交水稻产量具有重要意义。22日上午的专家评议会形成以下评议意见
        """
        self.chinese = text
        return super().setUp()

    def test_tfidf(self):
        document = Document(self.chinese)
        tfidf_vectors = [doc for doc in document.tfidf_model[document.corpus]]
        print(document.to_numpy(tfidf_vectors))

    def test_lsi(self):
        document = Document(self.chinese)
        lsi_vectors = [doc for doc in document.lsi_model[document.corpus]]
        # print(lsi_vectors)
        print(document.to_numpy(lsi_vectors, 100))

    def test_lda(self):
        document = Document(self.chinese)
        lda_vectors = [doc for doc in document.lda_model[document.corpus]]
        # print(lda_vectors)
        print(document.to_numpy(lda_vectors, 100))

    def test_word2vec(self):
        document = Document(self.chinese)
        print(document.word2vec(['中国', '水稻']))