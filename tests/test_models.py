from mynlp.models import CSVDistinct, CSVLinker
import unittest
from mynlp.settings import BASE_DIR
import os


class TestModel(unittest.TestCase):
    def setUp(self):
        self.input_file = os.path.join(BASE_DIR, 'data', 'dedupe_input.csv')
        self.input_file_base = os.path.join(BASE_DIR, 'data', 'linker_input.csv')
        self.settings_file = os.path.join(BASE_DIR, 'data', 'settings')
        self.output_file = os.path.join(BASE_DIR, 'data', 'output_file')
        self.training_file = os.path.join(BASE_DIR, 'data', 'training_file')
        self.linker_training_file = os.path.join(BASE_DIR, 'data', 'linker_training_file')
        self.linker_settings_file = os.path.join(BASE_DIR, 'data', 'linker_settings')
        self.linker_output_file = os.path.join(BASE_DIR, 'data', 'linker_output')
        return super().setUp()

    def test_dedupe(self):
        csvdedupe = CSVDistinct(input_file=self.input_file, 
                fields=[{'field': 'name', 'type': 'String'}],
                output_file=self.output_file,
                settings_file= self.settings_file,
                training_file= self.training_file
                )
        csvdedupe.run()

    def test_linker(self):
        linker = CSVLinker(
            fields=[{'field': 'name', 'type': 'String'}],
            input_file=self.input_file,
            input_file_base=self.input_file_base,
            output_file=self.linker_output_file,
            settings_file=self.linker_settings_file,
            training_file=self.linker_training_file
        )
        linker.run()
