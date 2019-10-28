from mynlp.models import CSVDedupe
import unittest
from mynlp.settings import BASE_DIR
import os


class TestCSVDedupe(unittest.TestCase):
    def setUp(self):
        self.input_file = os.path.join(BASE_DIR, 'data', 'dedupe_input.csv')
        self.settings_file = os.path.join(BASE_DIR, 'data', 'settings')
        self.output_file = os.path.join(BASE_DIR, 'data', 'output_file')
        self.training_file = os.path.join(BASE_DIR, 'data', 'training_file')
        return super().setUp()

    def test_dedupe(self):
        csvdedupe = CSVDedupe(input_file=self.input_file, 
                fields=[{'field': 'name', 'type': 'String'}],
                output_file=self.output_file,
                settings_file= self.settings_file,
                training_file= self.training_file
                )
        csvdedupe.run()

    def tearDown(self):
        os.remove(self.settings_file)
        os.remove(self.output_file)
        return super().tearDown()
