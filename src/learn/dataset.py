import csv
import numpy as np
import tensorflow as tf
import utils
import warnings

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

class DataSet(object):
    """Class for creating and maintaining a data set"""

    def __init__(self, examples, labels, train_percent=0.6, test_percent=0.2, batch_size=None,
                 normalize=True, pca=True, retained_variance=0.99, **config):
        if len(examples) != len(labels):
            raise ValueError('Must provide same number of examples and labels')

        self.examples = np.array(examples)
        self.labels = np.array(labels, ndmin=2).T
        self.train_percent = train_percent
        self.test_percent = test_percent

        shuffle(self.examples, self.labels)
        self.split_data()

        if pca:
            self.perform_pca(normalize, retained_variance)
        elif normalize:
            self.normalize_data()

        self.backup_train_data()

        self.batch_size = batch_size
        self.cur_batch = 0

        print 'Examples: {}, Features: {}'.format(self.n_examples, self.n_features)

    @staticmethod
    def create(filename, **config):
        """Factory method for creating a new dataset

        Arguments:
            filename {string} -- Path of file to save the new dataset to
            **config {dictionary} -- Configuration settings, MUST contain transform_fn
        """
        transform_fn = config.pop('transform_fn')

        print 'Creating new dataset...'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            examples, labels = utils.create_img_data(transform_fn, **config)

        dataset = DataSet(examples, labels, **config)
        print 'Created {} examples'.format(dataset.n_examples)

        print 'Saving data to CSV...'
        dataset.save_to_csv(filename)

        return dataset

    @staticmethod
    def read(filename, **config):
        """Factory method for reading a dataset from an existing CSV file"""
        print 'Reading data from CSV...'

        dataset = tf.contrib.learn.datasets.base.load_csv_without_header(
            filename=filename,
            features_dtype=np.float32,
            target_dtype=np.float32)

        print 'Read {} examples'.format(len(dataset.data))

        return DataSet(dataset.data, dataset.target, **config)

    def split_data(self):
        train_split = int(self.n_examples * self.train_percent)
        test_split = self.n_examples - int(self.n_examples * self.test_percent)

        self.X_train = self.examples[:train_split]
        self.y_train = self.labels[:train_split]
        self.X_cval = self.examples[train_split:test_split]
        self.y_cval = self.labels[train_split:test_split]
        self.X_test = self.examples[test_split:]
        self.y_test = self.labels[test_split:]

    def normalize_data(self):
        self.X_scaler = StandardScaler().fit(self.X_train)
        self.X_train = self.X_scaler.transform(self.X_train)
        self.X_cval = self.X_scaler.transform(self.X_cval) if len(self.X_cval) > 0 else self.X_cval
        self.X_test = self.X_scaler.transform(self.X_test)

    def perform_pca(self, normalize, retained_variance):
        self.split_data()

        if normalize:
            self.normalize_data()

        # Choose number of components k
        full_svd = PCA(n_components=None).fit(self.X_train)
        total_variance = 0.
        for k, variance in enumerate(full_svd.explained_variance_ratio_):
            total_variance += variance
            if total_variance > retained_variance:
                break

        self.pca = PCA(n_components=k+1).fit(self.X_train)
        self.X_train = self.pca.transform(self.X_train)
        self.X_cval = self.pca.transform(self.X_cval) if len(self.X_cval) > 0 else self.X_cval
        self.X_test = self.pca.transform(self.X_test)

    def shuffle_train_data(self):
        shuffle(self.X_train, self.y_train)

    def limit_training_data(self, limit):
        """Limit the training data to only the given number of examples

        Artificially limits the training set to the given number of examlpes, typically used to
        plot learning curves (as a function of the number of training examples)

        Arguments:
            limit {int | None} -- Number of examples to limit the training set to. If None is
                                  given, restores the training set to its original size
        """
        self.restore_train_data()
        self.X_train = self.X_train[:limit]
        self.y_train = self.y_train[:limit]

    def next_batch(self):
        if (self.cur_batch < self.n_batches - 1):
            self.cur_batch = self.cur_batch + 1
        else:
            self.cur_batch = 0
            self.shuffle_train_data()

    def save_to_csv(self, filename):
        data = [np.append(e, l) for e, l in zip(self.examples, self.labels.flatten())]

        with open(filename, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(data)

    def backup_train_data(self):
        self.X_train_backup = self.X_train.copy()
        self.y_train_backup = self.y_train.copy()

    def restore_train_data(self):
        self.X_train = self.X_train_backup.copy()
        self.y_train = self.y_train_backup.copy()

    @property
    def n_examples(self):
        return self.examples.shape[0]

    @property
    def n_train_examples(self):
        return self.X_train.shape[0]

    @property
    def n_features(self):
        return self.X_train.shape[1]

    @property
    def _batch_start(self):
        return self.cur_batch * self.batch_size

    @property
    def _batch_end(self):
        return min((self.cur_batch + 1) * self.batch_size, self.n_train_examples)

    @property
    def X_batch(self):
        if self.batch_size is None:
            return self.X_train

        return self.X_train[self._batch_start:self._batch_end]

    @property
    def y_batch(self):
        if self.batch_size is None:
            return self.y_train

        return self.y_train[self._batch_start:self._batch_end]

    @property
    def n_batches(self):
        if self.batch_size is None:
            return 1

        return int(np.ceil(self.n_train_examples / float(self.batch_size)))
