import numpy as np
from random import shuffle
import sys


class TSeries(object):

    def __init__(self, key, data_df, stride=1):
        self.key = key
        self.data_df = data_df
        self.stride = stride
        self.num_items = self.data_df.shape[0]
        self.current_idx = 0

    def get_batch(self, seq_len):

        if self.current_idx + seq_len > self.num_items:
            return None

        start = self.current_idx
        end = start + seq_len
        self.current_idx += self.stride

        return self.data_df.iloc[start:end, :]

    def max_iter(self, window):
        return (self.num_items-window)//self.stride + 1

    def reset(self):
        self.current_idx = 0


class TSDataGenerator(object):

    def __init__(self, data_df, x_cols, y_cols, batch_size, num_steps=50, stride=1, randomize=False, loop=False):
        self.data_df = data_df
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.stride = stride
        self.randomize = randomize
        self.loop = loop

        self.data = {}
        self.num_ids = 0
        self.first_id = 1
        self.current_id = 0
        self.num_iterations = 0

        self.build()

    def build(self):
        ids = self.data_df['id'].unique()
        self.num_ids = ids.shape[0]

        for i in ids:
            data = self.data_df[self.data_df['id'] == i]
            self.data[i] = TSeries(i, data, self.stride)

        self.first_id = ids[0]
        self.current_id = self.first_id

    def summary(self):
        stats = {}
        stats['items'] = len(self.data)
        stats['shape'] = self.data_df.shape
        stats['batch_size'] = self.batch_size

        num_iterations = 0
        for k, v in self.data.items():
            num_iterations += v.max_iter(self.batch_size)
        stats['max_iter'] = num_iterations
        stats['max_steps'] = num_iterations//self.batch_size

        return stats

    def print_summary(self, verbose=False):
        stats = self.summary()
        print("Number of items: ", stats['items'])
        print("Data shape: ", stats['shape'])
        print("Max Iterations: ", stats['max_iter'])
        print("Max steps: {} @ {}".format( stats['max_steps'], stats['batch_size']))

        if verbose:
            for k, v in self.data.items():
                print(k, v.num_items)

    def generate(self):

        x_data = []
        y_data = []
        num_in_batch = 0

        while True:
            self.num_iterations += 1

            sample_keys = list(self.data.keys())

            if self.randomize:
                shuffle(sample_keys)

            for k_id in sample_keys:
                data = self.data[k_id]

                if data.num_items < self.batch_size:
                    continue

                data_array = data.get_batch(self.num_steps)
                while data_array is not None:

                    x_seq = data_array[self.x_cols].values
                    x_data.append(x_seq)

                    y_seq = data_array[self.y_cols].values[0]
                    y_data.append(y_seq)

                    num_in_batch += 1

                    np_x_data = np.asarray(x_data)
                    np_y_data = np.asarray(y_data)

                    if num_in_batch == self.batch_size:
                        if np_x_data.shape[0] != self.num_steps:
                            np_x_data.shape

                        assert np_x_data.shape[0] == np_y_data.shape[0]

                        x_data = []
                        y_data = []
                        num_in_batch = 0

                        yield np_x_data, np_y_data

                    data_array = data.get_batch(self.num_steps)
                    if data_array is None and self.loop:
                        data.reset()

            if not self.loop:
                return


