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
from ..dedupe import BaseModel

class CSVDistinct(BaseModel):

    def __init__(self, log_level=logging.WARNING,
                 input_file='input.csv',
                 output_file='output.csv',
                 settings_file='settings',
                 training_file='training.json',
                 fields = []):
        # ## Logging
        logging.getLogger().setLevel(log_level)
        self.input_file = input_file
        self.output_file = output_file
        self.settings_file = settings_file
        self.training_file = training_file
        self.fields = fields
        self.data_d = None

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
                # row_id = int(row[0])
                row_id = row[0]
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

class CSVLinker(BaseModel):
    def __init__(self, log_level=logging.WARNING,
                input_file_base = 'input_base.csv',
                input_file='input.csv',
                output_file='output.csv',
                settings_file='settings',
                training_file='training.json',
                fields=[]):
        logging.getLogger().setLevel(log_level)
        self.input_file_base = input_file_base
        self.input_file = input_file
        self.output_file = output_file
        self.settings_file= settings_file
        self.training_file = training_file
        self.fields = fields
        self.data_1 = None
        self.data_2 = None

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
                data_d[str(filename) + str(row_id)] = dict(clean_row)

        return data_d


    def train(self):
        # data_d = self._read_data(self.input_file)
        # self.data_d = data_d
        data_1 = self._read_data(self.input_file_base)
        data_2 = self._read_data(self.input_file)
        self.data_1 = data_1
        self.data_2 = data_2

        fields = self.fields
        deduper = dedupe.RecordLink(fields)
        training_file = self.training_file

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

    def cluster(self):
        linker = self.deduper
        data_1 = self.data_1 if self.data_1 else self._read_data(self.input_file_base)
        data_2 = self.data_2 if self.data_2 else self._read_data(self.input_file)
        output_file = self.output_file

        # threshold1 = linker.threshold(data_1, data_2, recall_weight=2)

        linked_records = linker.match(data_1, data_2, threshold=0.1)
        # linked_records = [data for data in linked_records]
        print(linked_records)
        # import pdb;pdb.set_trace()
        print('# duplicate sets', len(linked_records))

        # ## Writing Results
        
        # Write our original data back out to a CSV with a new column called 
        # 'Cluster ID' which indicates which records refer to each other.
        
        # import pdb; pdb.set_trace()
        cluster_membership = {}
        cluster_id = None
        for cluster_id, (cluster, score) in enumerate(linked_records):
            for record_id in cluster:
                cluster_membership[record_id] = (cluster_id, score)

        print(cluster_membership)
        
        if cluster_id :
            unique_id = cluster_id + 1
        else :
            unique_id =0
            
        
        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            
            header_unwritten = True
        
            for fileno, filename in enumerate((self.input_file_base, self.input_file)) :
                with open(filename) as f_input :
                    reader = csv.reader(f_input)
        
                    if header_unwritten :
                        heading_row = next(reader)
                        heading_row.insert(0, 'source file')
                        heading_row.insert(0, 'Link Score')
                        heading_row.insert(0, 'Cluster ID')
                        writer.writerow(heading_row)
                        header_unwritten = False
                    else :
                        next(reader)
        
                    for row_id, row in enumerate(reader):
                        # print(str(filename)+ str(row_id))
                        # import pdb;pdb.set_trace()
                        cluster_details = cluster_membership.get(str(filename) + str(row[0]))
                        if cluster_details is None :
                            # cluster_id = unique_id
                            # unique_id += 1
                            # score = None
                            continue
                        else :
                            cluster_id, score = cluster_details
                        row.insert(0, fileno)
                        row.insert(0, score)
                        row.insert(0, cluster_id)
                        writer.writerow(row)
        