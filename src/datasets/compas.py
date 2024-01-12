import csv
import math
import json
from torch.utils.data import Subset, Dataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from typing import Any, Callable, List, Optional, Tuple, Union

import os
import warnings

class COMPAS_Dataset(Dataset):
    def __init__(self, csv_filename: str, subset=1):
        # subset 0 => High decile score, reoffended
        # subset 1 => High decile score, did not reoffend
        # subset 2 => Low decile score, reoffended
        # subset 3 => Low decile score, did not reoffend

        compas_full = COMPAS(csv_filename)
        indices = []
        for i in range(len(compas_full)):
            risk_cat = compas_full[i][1]
            recidivated = compas_full[i][2]

            if subset // 2 == 0:
                if risk_cat == 'Low':
                    continue
            else:
                if risk_cat == 'High':
                    continue

            if subset % 2 == 0:
                if recidivated == '0':
                    continue
            else:
                if recidivated == '1':
                    continue
            
            indices.append(i)

        self.train_set = Subset(compas_full, [indices[i] for i in range(math.floor(0.8 * len(indices)))])
        self.test_set = Subset(compas_full, [indices[i] for i in range(math.floor(0.8 * len(indices)), len(indices))])

class COMPAS(Dataset):
    def __init__(self, csv_filename):
        tag_cats = ['sex', 'age_cat', 'race', 'c_charge_degree', 'c_charge_desc']
        self.compas_json = {}

        with open(csv_filename, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
            headers = next(csv_reader)[0].split(',')
            for row in csv_reader:
                row = row[0].split(',')
                self.compas_json[row[0]] = {}
                for i in range(len(headers) - 1):
                    self.compas_json[row[0]][headers[i + 1]] = row[i + 1]

        self.tag_map = {}
        self.index_map = []
        self.num_tags = 0
        for person in self.compas_json:
            for tag_cat in tag_cats:
                tag = self.compas_json[person][tag_cat]
                if tag not in self.tag_map:
                    self.tag_map[tag] = self.num_tags
                    self.index_map.append(tag)
                    self.num_tags += 1

        self.idx_map = []
        self.instances = {}
        for person in self.compas_json:
            tags = [0 for i in range(self.num_tags)]
            for tag_cat in tag_cats:
                tag = self.compas_json[person][tag_cat]
                tags[self.tag_map[tag]] = 1
            self.instances[person] = tags
            self.idx_map.append(person)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        true_idx = self.idx_map[idx]
        return self.instances[true_idx], self.compas_json[true_idx]['score_text'], self.compas_json[true_idx]['two_year_recid']
