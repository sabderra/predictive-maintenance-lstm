import os
import pandas as pd


def path_to_id_base(path):
    fn = os.path.splitext(os.path.basename(path))[0]
    return fn_id_map[fn]


# Filename is mapped to a separate range to ensure unique engine ids
fn_id_map = {
    "train_FD001": 1000,
    "train_FD002": 2000,
    "train_FD003": 3000,
    "train_FD004": 4000,
    "test_FD001":  5000,
    "test_FD002":  6000,
    "test_FD003":  7000,
    "test_FD004":  8000,
    "RUL_FD001":  5000,
    "RUL_FD002":  6000,
    "RUL_FD003":  7000,
    "RUL_FD004":  8000,
}


# Filename is mapped to a condition. Map:
#       ONE (Sea Level) to 0
#       SIX to 1
fn_condition_map = {
    "train_FD001": 1,
    "train_FD002": 2,
    "train_FD003": 1,
    "train_FD004": 2,
    "test_FD001":  1,
    "test_FD002":  2,
    "test_FD003":  1,
    "test_FD004":  2,
}


def load_data(paths, col_names, sort_cols):
    # read data
    df = pd.DataFrame()
    for p in paths:
        instance_df = pd.read_csv(p, sep=" ", header=None)
        instance_df.drop(instance_df.columns[[26, 27]], axis=1, inplace=True)
        instance_df.columns = col_names
        instance_df['filename'] = os.path.splitext(os.path.basename(p))[0]

        df = pd.concat((df, instance_df), sort=False)

    df['condition'] = df['filename'].apply(lambda f: fn_condition_map[f])
    df['id'] = df['id'] + df['filename'].apply(lambda f: fn_id_map[f])
    df.drop(['filename'], axis=1, inplace=True)
    df = df.sort_values(sort_cols)
    return df


def calc_training_rul(df):
    # Data Labeling - generate column RUL
    rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    df = df.merge(rul, on=['id'], how='left')
    df['RUL'] = df['max'] - df['cycle']
    df.drop('max', axis=1, inplace=True)
    return df


def load_rul_data(paths, col_names):
    # Filename is used to determine the condition
    col_names.append('filename')

    # read data
    df = pd.DataFrame()
    for p in paths:
        instance_df = pd.read_csv(p, sep=" ", header=None)
        instance_df.drop(instance_df.columns[[1]], axis=1, inplace=True)
        instance_df['filename'] = os.path.splitext(os.path.basename(p))[0]
        instance_df = instance_df.reset_index()
        instance_df.columns = col_names

        df = pd.concat((df, instance_df), sort=False)

    df['id'] = df['id'] + df['filename'].apply(lambda f: fn_id_map[f]) + 1
    df.drop(['filename'], axis=1, inplace=True)
    return df


def calc_test_rul(feature_df, label_df):
    # If index is not reset there will be int/str type issues when attempting the merge.
    cycle_count_df = feature_df.groupby('id').count().reset_index()[['id', 'cycle']].rename(index=str, columns={
        "cycle": "cycles"}).reset_index(drop=True)

    # Join cycle and RUL dataframes
    assert cycle_count_df.shape[0] == label_df.shape[0]
    tmp_df = cycle_count_df.merge(label_df, on="id", how='left')

    # The RUL actual column contains the value for the last cycle.
    # Adding the cycles column will give us the RUL for the first cycle.
    tmp_df['RUL_actual'] = tmp_df['cycles'] + tmp_df['RUL_actual']
    tmp_df.drop('cycles', axis=1, inplace=True)

    # Join the two data frames
    feature_df = feature_df.merge(tmp_df, on='id', how='left')

    # Use the cycle to decrement the RUL until the ground truth is reached.
    feature_df['RUL'] = feature_df['RUL_actual'] - feature_df['cycle']
    feature_df.drop('RUL_actual', axis=1, inplace=True)

    return feature_df


def transform(df, pipeline):
    # Set up the columns that will be scaled
    df['cycle_norm'] = df['cycle']

    # Transform all columns except id, cycle, and RUL
    cols_transform = df.columns.difference(['id', 'cycle', 'RUL'])

    xform_df = pd.DataFrame(pipeline.transform(df[cols_transform]),
                            columns=cols_transform,
                            index=df.index)
    join_df = df[df.columns.difference(cols_transform)].join(xform_df)
    df = join_df.reindex(columns=df.columns)
    return df
