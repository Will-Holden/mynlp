import dedupe
import os
import logging
import csv


class BaseModel:

    def __init__(self):
        self.settings_file = None

    def load_model(self, settings_file):
        if os.path.exists(settings_file):
            logging.info('reading settings from', settings_file)
            with open(settings_file, 'rb') as f:
                self.deduper = dedupe.StaticDedupe(f)
        else:
            raise Exception("file not found")

    def save_model(self, settings_file):
        with open(settings_file, 'wb') as f:
            self.deduper.writeSettings(f)

    def _read_data(self, filename):
        """
        read in out data from a csv  file and create a dictionary of records.
        where the key is a unique record ID and each value is dict
        """
        
        data_d = {}
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = [(k, v) for (k, v) in row.items()]
                # row_id = int(row['Id'])
                row_id = row['Id']
                data_d[row_id] = dict(clean_row)

        return data_d

    def train(self):
        pass

    def cluster(self):
        pass

    def run(self):
        if os.path.exists(self.settings_file):
            self.load_model(self.settings_file)
        else:
            print('starting run csv dedupe')
            self.train()
        print('starting cluster data')
        self.cluster()
        print('starting saving model')
        self.save_model(self.settings_file)