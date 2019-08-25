import numpy as np
import os
from info import DataKeys, Paths, ModelParameters as Params, Constants
from dataset_generator import read_json
import json
import jsonpickle
from keras.utils import Sequence
import random


class Dataloader(Sequence):
    mode_train = 0
    mode_validation = 1

    # Date, Open, High, Low, Close, Volume, OpenInt
    def __init__(self, loader_type, sectors=None, split_size=100):
        """initializes dataloader

        Args:
            loader_type (int): 0 - train; 1 - validation
            sectors (list): names of sectors (found in DataKeys); None for all
        """
        self.loader_type = loader_type
        self.sectors = sectors
        self.split_size = split_size

        self.entry_num_dict, self.sector_dict = self.load_dictionaries()
        self.sample_names = self.load_sample_names()

    def load_sample_names(self):
        sample_names = []

        if self.sectors is not None:
            for sector in self.sectors:
                sample_names.extend(self.sector_dict[sector])
        else:
            for tickers in self.sector_dict.values():
                sample_names.extend(tickers)

        sample_names = [name for name in sample_names if
                        name in self.entry_num_dict and self.entry_num_dict[name] > Params.min_data_size]

        split = []
        for i in range(self.split_size):
            split.append(random.choice(sample_names))

        return split

    @staticmethod
    def load_dictionaries():
        with open(Paths.ticker_num_entries) as file:
            entry_num_dict = json.loads(file.read())
        with open(Paths.sector_stock_dictionary) as file:
            sector_dict = json.loads(file.read())
        return entry_num_dict, sector_dict

    def __getitem__(self, index):
        start_idx = index * Params.batch_size
        end_idx = min(start_idx + Params.batch_size, self.split_size)
        return self.load_batch(start_idx, end_idx)

    def load_batch(self, start_sample, end_sample):

        input_data = np.empty((Params.batch_size, Params.input_days, Params.info_dimension))
        labels = [np.empty((Params.batch_size, 1)) for _ in Params.days_to_predict]

        for sample in range(start_sample, end_sample):
            zero_sample = sample - start_sample
            name = self.sample_names[sample]
            path = self.path_to_stock(name)
            data = read_json(path)

            data_size = data[DataKeys.data_size]

            while data_size < Params.min_data_size:
                new_name = random.choice(self.sample_names)
                print(f"Not enough data for {name}. Attempting to replace with {new_name}.")
                path = self.path_to_stock(new_name)
                data = read_json(path)
                name = new_name

                data_size = data[DataKeys.data_size]

            # get frame scope for data
            farthest_prediction = Params.days_to_predict[-1]
            if self.loader_type == self.mode_train:
                rand_int_start = 0
                rand_int_end = data_size - Params.input_days - Params.validation_interval - farthest_prediction
            else:
                rand_int_start = data_size - Params.input_days - Params.validation_interval
                rand_int_end = data_size - Params.input_days - farthest_prediction

            start = random.randint(rand_int_start, rand_int_end)
            end = start + Params.input_days

            # extract input
            i = 0
            pivot_point = None
            for name, data_type in Params.data.items():
                if not data_type.used:
                    continue

                dim = data_type.dimension
                extracted = data[name][start:end]

                if Params.predicted_key == name:
                    pivot_point = float(extracted[-1])

                # get day-to-day change
                if not data_type.is_categorical:
                    extracted[1:] = extracted[1:] / (extracted[:-1] + Constants.epsilon)
                    extracted[0] = 1

                input_data[zero_sample, ..., i:i + dim] = extracted

                i += 1

            # get label
            for i, pred_day in enumerate(Params.days_to_predict):
                if Params.predict_direction:
                    labels[i][zero_sample] = data[Params.predicted_key][end + pred_day - 1] > pivot_point
                else:
                    labels[i][zero_sample] = data[Params.predicted_key][end + pred_day - 1] / pivot_point

        return input_data, labels

    def on_epoch_end(self):
        return random.shuffle(self.sample_names)

    def __len__(self):
        return self.split_size // Params.batch_size

    @staticmethod
    def path_to_stock(name):
        return os.path.join(Paths.extracted_stocks, name + ".json")
