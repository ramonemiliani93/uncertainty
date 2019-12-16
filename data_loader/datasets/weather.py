import os

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sklearn.model_selection import train_test_split
import torch
from torchvision.datasets.utils import download_url, makedir_exist_ok

from .base import UncertaintyDataset


class WeatherDataset(UncertaintyDataset):
    """Weather dataset from the https://mrcc.illinois.edu/CLIMATE portal.

    Args:
        root (string): Root directory of dataset where ``WeatherDataset/processed/training.pt``
            and  ``WeatherDataset/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an day
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    urls = [
        'https://drive.google.com/uc?id=1ueQ4MYGqI2bOY5gwlpxn8wGZtMqbvDS3'
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    variables = {
        'precipitation': 'PRCP',
        'snow': 'SNOW',
        'snow-depth': 'SNWD',
        'max-temp': 'TMAX',
        'min-temp': 'TMIN',
        'mean-temp': 'MEAN',
    }

    @property
    def mean(self):
        return self.mean_per_day[self.variables.get(self.variable)].tolist()

    @property
    def std(self):
        return self.std_per_day[self.variables.get(self.variable)].tolist()

    def __init__(self, root='data',
                 variable='max-temp',
                 train=True,
                 num_years_train=1,
                 transform=None,
                 target_transform=None,
                 download=True):
        super(WeatherDataset, self).__init__()
        assert variable in self.variables.keys(), "Use a variable from {}".format(list(self.variables.keys()))
        self.root = root
        self.variable = variable
        self.train = train
        self.num_years_train = num_years_train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        df_dict, self.mean_per_day, self.std_per_day = torch.load(os.path.join(self.processed_folder, data_file))
        self.data = df_dict.get(self.variable)
        self.data = self.data[[self.variables.get(self.variable)]]  # Get variable of interest

    def __getitem__(self, sample):
        """
        Args:
            sample (int): sample number to be retrieved

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        day, target = self.data.index[sample].dayofyear, self.data.iat[sample, 0]

        if self.transform is not None:
            day = self.transform(day)
        day = torch.tensor([day], dtype=torch.float32)

        if self.target_transform is not None:
            target = self.target_transform(target)
        target = torch.tensor([target], dtype=torch.float32)

        if self.probabilities is None:
            probability = torch.tensor([1], dtype=torch.float32)
        else:
            probability = torch.tensor(self.probabilities[sample])

        return day, target, probability

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__.lower(), 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__.lower(), 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the weather data if it doesn't exist in data folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = self.__class__.__name__.lower() + '.csv'
            download_url(url, root=self.raw_folder, filename=filename)

        # process and save as torch files
        print('Processing...')
        filepath = os.path.join(self.raw_folder, filename)
        df = self.process(filepath)

        # Get dataset statistics
        mean_per_day = df.groupby(df.index.dayofyear).mean()
        std_per_day = df.groupby(df.index.dayofyear).std()

        # Split into training and testing
        train, test = {}, {}
        for variable, column in self.variables.items():
            df_variable = df[[column, 'dayofyear', 'year']].dropna()
            train_var = df_variable.groupby('dayofyear', as_index=False).apply(
                lambda x: x.sample(min(self.num_years_train, len(x)))
            ).droplevel(0)
            test_var = df_variable.drop(train_var.index)
            train[variable] = train_var
            test[variable] = test_var

        # Save data
        training_set = (
            train,
            mean_per_day,
            std_per_day
        )
        test_set = (
            test,
            mean_per_day,
            std_per_day
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    @staticmethod
    def process(filepath):
        df = pd.read_csv(filepath,
                         header=3,
                         usecols=range(7),
                         skipfooter=17,
                         na_values='M',
                         parse_dates=[0],
                         infer_datetime_format=True,
                         memory_map=True,
                         engine='python',
                         converters={  # Use trace as 0
                             'PRCP': lambda x: 0 if x is 'T' else x,
                             'SNOW': lambda x: 0 if x is 'T' else x,
                             'SNWD': lambda x: 0 if x is 'T' else x,
                         })
        df.set_index('Date', inplace=True)
        df['dayofyear'] = df.index.dayofyear
        df['year'] = df.index.year

        return df

    def generate_neighbors(self, neighbors: int, **kwargs) -> np.ndarray:
        # Extract parameters if provided in kwargs.
        dimension = 1  # Length of item vector that will be indexed
        metric = kwargs.get('metric', 'euclidean')
        num_trees = kwargs.get('num_trees', 10)

        # Build tree with the given data.
        t = AnnoyIndex(dimension, metric)
        for i in range(len(self)):
            t.add_item(i, [self[i][0].item()])
        t.build(num_trees)

        # Generate neighbor map array.
        neighbor_map = np.zeros((len(self), neighbors))
        for i in range(len(self)):
            nearest_neighbors = t.get_nns_by_item(i, neighbors)
            neighbor_map[i, :] = nearest_neighbors

        self.neighbor_map = neighbor_map.astype(int)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = WeatherDataset()

    print("Done!")