"""
Common dedupe run 
"""
from future.builtins import next

import os
import csv
import re
import logging
import optparse


import dedupe
from unidecode import unidecode
from ..dedupe import BaseModel

class CommonDistinct(BaseModel):
    def __init__(self, log_level=logging.WARNING,
    fields = []):
        # ## Logging
        logging.getLogger().setLevel(log_level)
        self.fields = fields
        self.data_d = None

    # to be done
    def train(self, data, training_file):
        fields = self.fields
        deduper = dedupe.Dedupe(fields)
        # ## training data
        if os.path.exists(training_file):
            logging.info('reading labeled examples from ', training_file)
            with open(training_file, 'rb') as f:
                deduper.prepare_training(data, f)
        else:
            raise Exception('no training data')
        
        deduper.train()
        self.deduper = deduper
        return self

    def active_train(self, input_data ,training_file):
        data_d = input_data
        fields = self.fields
        deduper = dedupe.Dedupe(fields)
        # ## training data
        if os.path.exists(training_file):
            logging.info('reading labeled examples from ', training_file)
            with open(training_file, 'rb') as f:
                deduper.prepare_training(data_d, f)
        else:
            deduper.prepare_training(data_d)
        
        print('starting active labeling...')
        dedupe.consoleLabel(deduper)
        deduper.train()

        with open(training_file, 'w') as tf:
            deduper.writeTraining(tf)
        self.deduper = deduper
        return self

    def cluster(self, data):
        """
        输入是字典 id 是 key，字段值组成的字典是value
        输出是[((id1, id2), array([score1, score2])), ...]
        """
        deduper = self.deduper
        threshold = deduper.threshold(data, recall_weight=1)

        # ## clustering
        print('clustering...')
        clustered_dupes = deduper.match(data, threshold)

        print('# duplicate sets', len(clustered_dupes))
        return clustered_dupes

class CommonLink(BaseModel):
    def __init__(self, log_level=logging.WARNING,
    fields=[]):
        logging.getLogger().setLevel(log_level)
        self.fields = fields

    def load_model(self, settings_file):
        if os.path.exists(settings_file):
            logging.info('reading settings from', settings_file)
            with open(settings_file, 'rb') as f:
                self.deduper = dedupe.StaticRecordLink(f)
        else:
            raise Exception("file not found")

    def train(self, data_1, data_2, training_file):
        fields = self.fields
        deduper = dedupe.RecordLink(fields)
        # ## training data
        if os.path.exists(training_file):
            logging.info('reading labeled examples from ', training_file)
            with open(training_file, 'rb') as f:
                deduper.prepare_training(data_1, data_2, f)
        else:
            # deduper.prepare_training(data_1, data_2)
            raise Exception("no traing data")

        # print('starting active labeling...')
        # dedupe.consoleLabel(deduper)
        deduper.train(recall=1)
        self.deduper = deduper
        return self

    def active_train(self, data_1, data_2, training_file):
        fields = self.fields
        deduper = dedupe.RecordLink(fields)

        # ## training data
        if os.path.exists(training_file):
            logging.info('reading labeled examples from ', training_file)
            with open(training_file, 'rb') as f:
                deduper.prepare_training(data_1, data_2, f)
        else:
            deduper.prepare_training(data_1, data_2)

        print('starting active labeling...')
        dedupe.consoleLabel(deduper)
        deduper.train(recall=1)

        with open(training_file, 'w') as tf:
            deduper.writeTraining(tf)

        self.deduper = deduper
        return self

    def cluster(self, data_1, data_2):
        """输入
        输出 [((id1, id2), score1)]
        """
        linker = self.deduper
        threshold = linker.threshold(data_1, data_2, recall_weight=2)

        linked_records = linker.match(data_1, data_2, threshold=threshold)
        return linked_records

