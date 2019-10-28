"""
Csv dedupe run
"""
from future.builtins import next

import os
import csv
import re
import logging
import optparse


import dedupe
from unidecode import unidecode


class CSVDedupe:
    def __init__(self, log_level=logging.WARNING,
                 input_file='input.csv',
                 output_file='output.csv',
                 settings_file='settings',
                 training_file='training.json',
                 fields = {}):
        # ## Logging
        logging.getLogger().setLevel(log_level)
        self.input_file = input_file
        self.output_file = output_file
        self.settings_file = settings_file
        self.training_file = training_file
        self.fields = fields
        self.data_d = None

    def run(self):
        print('starting run csv dedupe')
        self.train()
        print('starting cluster data')
        self.cluster()
        print('starting saving model')
        self.save_model(self.settings_file)

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
                row_id = int(row['Id'])
                data_d[row_id] = dict(clean_row)

        return data_d

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

    def train(self):
        data_d = self._read_data(self.input_file)
        self.data_d = data_d

        fields = self.fields
        deduper = dedupe.Dedupe(fields)
        training_file = self.training_file

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

    def cluster(self):
        deduper = self.deduper
        if not self.data_d:
            self.data_d = self._read_data(self.input_file)
        data_d = self.data_d
        threshold = deduper.threshold(data_d, recall_weight=1)

        # ## clustering
        print('clustering...')
        clustered_dupes = deduper.match(data_d, threshold)

        print('# duplicate sets', len(clustered_dupes))
        cluster_membership = {}
        cluster_id = 0
        for (cluster_id, cluster) in enumerate(clustered_dupes):
            id_set, scores = cluster
            cluster_d = [data_d[c] for c in id_set]
            canonical_rep = dedupe.canonicalize(cluster_d)
            for record_id, score in zip(id_set, scores):
                cluster_membership[record_id] = {
                    'cluster id': cluster_id,
                    'canonical representation' : canonical_rep,
                    'confidence': score
                }

        singleton_id = cluster_id + 1

        output_file = self.output_file
        input_file = self.input_file
        with open(output_file, 'w') as f_output, open(input_file) as f_input:
            writer = csv.writer(f_output)
            reader = csv.reader(f_input)
        
            heading_row = next(reader)
            heading_row.insert(0, 'confidence_score')
            heading_row.insert(0, 'Cluster ID')
            canonical_keys = canonical_rep.keys()
            for key in canonical_keys:
                heading_row.append('canonical_' + key)
        
            writer.writerow(heading_row)
        
            for row in reader:
                row_id = int(row[0])
                if row_id in cluster_membership:
                    cluster_id = cluster_membership[row_id]["cluster id"]
                    canonical_rep = cluster_membership[row_id]["canonical representation"]
                    row.insert(0, cluster_membership[row_id]['confidence'])
                    row.insert(0, cluster_id)
                    for key in canonical_keys:
                        row.append(canonical_rep[key].encode('utf8'))
                else:
                    row.insert(0, None)
                    row.insert(0, singleton_id)
                    singleton_id += 1
                    for key in canonical_keys:
                        row.append(None)
                writer.writerow(row)