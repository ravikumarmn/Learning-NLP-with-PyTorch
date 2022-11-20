import config
from sklearn.model_selection import train_test_split


def split_dataset(dataframe):
    train_dataset,valid_dataset= train_test_split(
        dataframe,
        test_size=0.2,
        shuffle=True
    )
    return train_dataset,valid_dataset