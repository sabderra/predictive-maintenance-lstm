import numpy as np
import pandas as pd
from random import shuffle


def split_data(df, randomize=True, train_pct=0.8):
    """Given a pandas DataFrame of time series data that is organized with a unique 'id' column,
    split the data into a training and a validation set.

    Args:
        df (DataFrame) - Data to split
        randomize (bool) -- Shuffle the order of the engines while retaining the times series order (default True)
        train_pct (float) -- Percentage as float to allocate as training data.

    Returns:
        DataFrame: Training data DataFrame
        DataFrame: Validation data DataFrame
    """

    all_ids = df['id'].unique()

    if randomize:
        shuffle(all_ids)

    num_train_samples = int(all_ids.shape[0] * train_pct)
    train_sample_ids = all_ids[:num_train_samples]
    test_sample_ids = all_ids[num_train_samples:]

    train_data_df = df[df['id'].isin(train_sample_ids)]
    val_data_df = df[df['id'].isin(test_sample_ids)]

    assert df['id'].unique().shape[0] == train_data_df['id'].unique().shape[0] + val_data_df['id'].unique().shape[0]

    return train_data_df, val_data_df


def create_generators(train_data_df, val_data_df, x_cols, y_cols, batch_size=128, sequence_length=25, stride=1,
                      randomize=False, loop=False, pad=False, verbose=False):
    """
    Given training and validation DataFrames, create Generators for each. The columns that will be used for the features
    and labels is specified by the x_cols and y_cols. Only these columns will be returned by the generators.

    Args:
        train_data_df (DataFrame): Training data
        val_data_df (DataFrame): Validation data
        x_cols (list): List of column names for the feature (X) data.
        y_cols (list): List of column names for the label (Y) data.
        batch_size (int): Size of the batch to be returned by the generators
        sequence_length (int): The number of time steps returned from each series.
        stride (int): Steps between time series events (default 1)
        randomize (bool): If true the engines are shuffled (default False)
        loop (bool): If true, continuously loop when the end of the data is reached (default False)
        pad (bool): If True, will add zero value rows to ensure all data can be returned,
                    i.e. rows % seq_length == 0.
        verbose (bool): If true, statistics on the generates is printed out.
    Returns:
        TSDataGenerator: Training generator
        TSDataGenerator: Validation generator
    """

    train_data_generator = TSDataGenerator(train_data_df,
                                           x_cols,
                                           y_cols,
                                           batch_size=batch_size,
                                           seq_length=sequence_length,
                                           stride=stride,
                                           randomize=randomize,
                                           loop=loop,
                                           pad=pad)

    val_data_generator = TSDataGenerator(val_data_df,
                                         x_cols,
                                         y_cols,
                                         batch_size=batch_size,
                                         seq_length=sequence_length,
                                         stride=stride,
                                         randomize=randomize,
                                         loop=loop,
                                         pad=pad)

    if verbose:
        num_t_engines = train_data_df['id'].unique().shape[0]
        num_v_engines = val_data_df['id'].unique().shape[0]
        num_t_cycles = train_data_df.shape[0]
        num_v_cycles = val_data_df.shape[0]

        print("Engine split: Training={:.2f}, Validation={:.2f}"
              .format(num_t_engines / (num_t_engines + num_v_engines),
                      num_v_engines / (num_t_engines + num_v_engines)))

        print("Cycle split:  Training={:.2f}, Validation={:.2f}"
              .format(num_t_cycles / (num_t_cycles + num_v_cycles),
                      num_v_cycles / (num_t_cycles + num_v_cycles)))

        train_data_generator.print_summary()
        val_data_generator.print_summary()

    return train_data_generator, val_data_generator


class TSeries(object):
    """ Single engine's time series data.
    """

    def __init__(self, key, data_df, seq_length, stride=1, pad=False):
        """
        Args:
            key (str): Id of the engine
            data_df (DataFrame): DataFrame containing all engine series events.
            seq_length (int): Amount of events to be returned.
            stride (int): Steps between events to return.
            pad (bool): If True, will add zero value rows to ensure all data can be returned,
                        i.e. rows % seq_length == 0.
        """
        self.current_idx = 0
        self.key = key
        self.seq_length = seq_length
        self.stride = stride

        self.data_df = data_df

        if pad:
            gap = data_df.shape[0] % seq_length
            if gap > 0:
                num_pad_rows = seq_length - gap
                pad_df = pd.DataFrame(np.zeros((num_pad_rows, data_df.shape[1])),
                                      columns=data_df.columns)
                pad_df['id'] = pad_df['id'].astype(int)
                self.data_df = pd.concat([pad_df, data_df])
                self.data_df.reset_index()

                assert (self.data_df.shape[0] % seq_length) == 0

        self.num_items = self.data_df.shape[0]

    def get_sequence(self):
        """ Returns:
                A seq_length entries and increments the index to the next window."""

        if self.current_idx + self.seq_length > self.num_items:
            return None

        start = self.current_idx
        end = start + self.seq_length
        self.current_idx += self.stride

        return self.data_df.iloc[start:end, :]

    def max_steps(self, window):
        return max(0, (self.num_items-window)//self.stride + 1)

    def reset(self):
        self.current_idx = 0


class TSDataGenerator(object):
    """ Generator that will return time series data for use by an RNN such as LSTM or GRU.
    For each iteration, data is returned with dimensions (batch_size, sequence_length, num_columns).
    """

    def __init__(self, data_df, x_cols, y_cols, batch_size, seq_length=50, stride=1,
                 randomize=False, loop=False, pad=False):
        """
        Args:
            data_df (DataFrame): Data
            x_cols (list): List of column names for the feature (X) data.
            y_cols (list): List of column names for the label (Y) data.
            batch_size (int): Size of the batch to be returned by the generators
            seq_length (int): The number of time steps returned from each series. (Default 50)
            stride (int): Steps between time series events (default 1)
            randomize (bool): If true the engines are shuffled (default False)
            loop (bool): If true, continuously loop when the end of the data is reached (default False)
            pad (bool): If True, will add zero value rows to ensure all data can be returned,
                        i.e. rows % seq_length == 0.
        """

        self.data_df = data_df
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.stride = stride
        self.randomize = randomize
        self.loop = loop

        self.data = {}
        self.num_ids = 0
        self.first_id = 1
        self.current_id = 0
        self.num_steps = 0

        self.build(pad)

    def build(self, pad=False):
        ids = self.data_df['id'].unique()
        self.num_ids = ids.shape[0]

        for i in ids:
            data = self.data_df[self.data_df['id'] == i]
            self.data[i] = TSeries(i, data, self.seq_length, stride=self.stride, pad=pad)

        self.first_id = ids[0]
        self.current_id = self.first_id

    def summary(self):
        """ Stats that estimate what this generator can produce.
        Returns:
           dict of stats
        """

        stats = dict()
        stats['shape'] = self.data_df.shape
        stats['batch_size'] = self.batch_size
        stats['seq_length'] = self.seq_length

        num_items = num_total_items = num_steps = num_total_steps = 0
        for k, v in self.data.items():
            num_total_items += 1
            num_total_steps += v.max_steps(self.seq_length)

            # Only count if the data can fit in a batch. Refer to padding.
            if v.num_items >= self.seq_length:
                num_items += 1
                num_steps += v.max_steps(self.seq_length)

        stats['items'] = num_items
        stats['total_items'] = num_total_items
        stats['max_steps'] = num_steps
        stats['max_total_steps'] = num_total_steps

        stats['max_iterations'] = num_steps//self.batch_size

        return stats

    def print_summary(self, verbose=False):
        stats = self.summary()
        print("Number of items: ", stats['items'])

        # The sequence length will be relevant to the cycle lengths. Engines will have to
        # have minimum number of cycles no less then the sequence length else it would ignored or
        # need to be padded. Those that will be dropped are counted as undersized.

        print("Undersized items: ", stats['total_items'] - stats['items'])
        print("Data shape: ", stats['shape'])
        print("Max steps: ", stats['max_steps'])
        print("Max iterations: {} @ {}".format(stats['max_iterations'], stats['batch_size']))

        if verbose:
            for k, v in self.data.items():
                print(k, v.num_items)

    def generate(self):

        x_data = []
        y_data = []
        num_in_batch = 0

        while True:
            self.num_steps += 1

            sample_keys = list(self.data.keys())

            if self.randomize:
                shuffle(sample_keys)

            for k_id in sample_keys:
                data = self.data[k_id]

                # Only use this engine if it has enough data to fit in a sequence. Refer to padding
                if data.num_items < self.seq_length:
                    continue

                data_array = data.get_sequence()
                while data_array is not None:

                    x_seq = data_array[self.x_cols].values
                    x_data.append(x_seq)

                    y_seq = data_array[self.y_cols].values[-1]
                    y_data.append(y_seq)

                    num_in_batch += 1

                    np_x_data = np.asarray(x_data)
                    np_y_data = np.asarray(y_data)

                    if num_in_batch == self.batch_size:
                        if np_x_data.shape[0] != self.seq_length:
                            np_x_data.shape

                        assert np_x_data.shape[0] == np_y_data.shape[0]

                        x_data = []
                        y_data = []
                        num_in_batch = 0

                        yield np_x_data, np_y_data

                    data_array = data.get_sequence()
                    if data_array is None and self.loop:
                        data.reset()

            if not self.loop:
                return


