import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from contextlib import contextmanager
from dataset import DataSet

COST_GRANULARITY = 100
DATA_DIRECTORY = '../../data'
MODEL_DIRECTORY = '../../saved_models'

class TransformLearner(object):
    """Class for learning and identifying a specific image transformation"""

    def __init__(self, **config):
        create = config.pop('create_data', False)

        self.name = config.get('name', 'default')
        self.regularize = config.get('regularize', True)
        self.ideal_rs = config.get('ideal_rs', None)

        self.data_file = '{}/{}_data.csv'.format(DATA_DIRECTORY, self.name)
        self.model_file = '{}/{}_model.csv'.format(MODEL_DIRECTORY, self.name)

        # Read or create data
        data_fn = DataSet.create if create else DataSet.read
        self.data = data_fn(self.data_file, **config)

        self.n_epochs = config.get('n_epochs', 1000)
        self.sess = None

    def run(self, start_rs=0.01):
        """Trains and evaluates the algorithm

        Keyword Arguments:
            start_rs {float} -- Seed value for _find_ideal_rs (default: {0.01})
        """
        ideal_rs = self.ideal_rs

        if self.regularize and self.ideal_rs is None:
            ideal_rs = self._find_ideal_rs(start_value=start_rs)

        self._init_tensors(random=True, reg_strength=ideal_rs)

        with self._session():
            self._train(plot_cost=True)
            self._evaluate()

    def plot_learning_curves(self, granularity=15):
        """Plots the training set and cross-validation set errors as a function of
        the number of training examples

        Convenience method for debugging algorithms. Observing the trends in
        training and cross-validaiton error as the number of training examples
        increases can provide insight into whether an algorithm is suffereing from
        high bias or high variance.

        Keyword Arguments:
            granularity {int} -- Level of granularity (default: {15})
        """
        n_examples = self.data.n_train_examples / granularity
        limits = [n_examples * (i + 1) for i in range(granularity)]
        errors = []

        for lim in limits:
            self.data.limit_training_data(lim)
            self._init_tensors(reg_strength=self.ideal_rs)

            with self._session():
                errors.append(self._train())

        errors = [list(t) for t in zip(*errors)]
        train_error = errors[0]
        cv_error = errors[1]

        plt.plot(limits, train_error, 'r-', limits, cv_error, 'b-')
        plt.axis([0, self.data.n_train_examples, 0, np.max(cv_error)])
        plt.show()

        self.data.limit_training_data(None)

    def _init_tensors(self, reg_strength=0.01, random=False):
        """Initializes the Tensors

        Keyword Arguments:
            reg_strength {float} -- Regularization strength (default: {0.01})
            random {bool} -- If True, initializes the weights and bias term
                             to normally distributed random values (default: {False})
        """
        X = tf.placeholder(tf.float32, [None, self.data.n_features])
        Y = tf.placeholder(tf.float32, [None, 1])

        if random:
            W = tf.Variable(tf.random_normal([self.data.n_features, 1]), name='weight', dtype=tf.float32)
            b = tf.Variable(tf.random_normal([1]), name='bias', dtype=tf.float32)
        else:
            W = tf.Variable(tf.ones([self.data.n_features, 1]), name='weight', dtype=tf.float32)
            b = tf.Variable(tf.ones([1]), name='bias', dtype=tf.float32)

        y_ = tf.add(tf.matmul(X, W), b)
        cost_no_reg = tf.reduce_mean(tf.square(y_ - Y))

        reg_term = reg_strength * tf.reduce_mean(tf.square(W))
        cost = cost_no_reg + reg_term if self.regularize else cost_no_reg

        training_step = tf.train.AdamOptimizer().minimize(cost)
        self.tensors = (X, Y, W, b, y_, cost, cost_no_reg, training_step)

    def _train(self, plot_cost=False):
        """Trains the algorithm

        Keyword Arguments:
            plot_cost {bool} -- If True, plots the training error as a function of the
                                number of epochs that have occured (default: {False})

        Returns:
            tuple -- (training error, cross-validation error)

        Raises:
            ValueError -- If a session has not already been initialized
        """
        X, Y, W, b, y_, cost, cost_no_reg, training_step = self.tensors

        if self.sess is None:
            raise ValueError('Session has not been initialized')

        # Train
        print 'Running gradient descent with {} batch(es)...'.format(self.data.n_batches)
        cost_history = []

        for epoch in range(self.n_epochs):
            for batch in range(self.data.n_batches):
                self.sess.run(training_step, feed_dict={X: self.data.X_batch, Y: self.data.y_batch})
                self.data.next_batch()

            if epoch % COST_GRANULARITY == 0:
                c = self.sess.run(cost, feed_dict={X: self.data.X_train, Y: self.data.y_train})
                cost_history.append(c)
                print('Epoch: {}  Cost: %.2f'.format(epoch) % c)

        if plot_cost:
            self.plot_cost(cost_history)

        train_error = self.sess.run(cost_no_reg, feed_dict={X: self.data.X_train, Y: self.data.y_train})
        cv_error = self.sess.run(cost_no_reg, feed_dict={X: self.data.X_cval, Y: self.data.y_cval})

        print 'Training error: %.3f' % train_error
        print 'Cross-validation error: %.3f' % cv_error

        return train_error, cv_error

    def _evaluate(self):
        """Evalutes the performance of the algorithm

        Raises:
            ValueError -- If a session has not already been initialized
        """
        X, Y, W, b, y_, cost, cost_no_reg, training_step = self.tensors

        if self.sess is None:
            raise ValueError('Session has not been initialized')

        pred_y = self.sess.run(y_, feed_dict={X: self.data.X_test})
        self.print_percent_within(pred_y, [0.01, 0.05, 0.1])
        self.plot_predictions(pred_y)

        test_error = self.sess.run(cost_no_reg, feed_dict={X: self.data.X_test, Y: self.data.y_test})
        print 'Test error: %.3f' % test_error

    def _find_ideal_rs(self, start_value=0.01, n_iters=7, precision=3):
        """Determines the ideal regularization strength value using cross-validation

        Computes the regularization strength value that minimizes cross-validation
        set error by iteratively checking the error produced by exponentially increasing
        values.

        Values are computed in batches of 7. If the value that minimizes the cross-val
        error is the first or last in the list, the method is called recursivley with
        a new start_value to determine if the cross-val error continues to decrease
        in that direction.

        Keyword Arguments:
            start_value {float} -- Regularization strength to start at (default: {0.01})
            n_iters {int} -- Number of iterations (default: {7})
            precision {int} -- Number of decimal places to round cv_error to before
                               comparing (default: {3})

        Returns:
            int -- Ideal regularization strength value
        """
        reg_strengths = [start_value * (2 ** i) for i in range(n_iters)]
        cv_errors = []

        for rs in reg_strengths:
            print 'Evaluating with regularization strength: {}'.format(rs)
            self._init_tensors(reg_strength=rs)

            with self._session():
                cv_errors.append(self._train()[1])

        np.round(cv_errors, decimals=precision)
        idx = np.argmin(cv_errors)
        all_equal = np.all(cv_errors == cv_errors[0])

        if idx == 0 and not all_equal:
            # Subtracting 2 here avoids a potential infinite loop
            new_start = start_value / (2 ** (n_iters - 2))
            return self._find_ideal_rs(start_value=new_start, n_iters=n_iters)
        elif idx == n_iters - 1:
            new_start = start_value * (2 ** (n_iters - 1))
            return self._find_ideal_rs(start_value=new_start, n_iters=n_iters)
        else:
            ideal_rs = reg_strengths[idx]
            print 'Ideal regularization strength: {}'.format(ideal_rs)
            return ideal_rs

    @contextmanager
    def _session(self):
        self._init_session()
        try:
            yield
        finally:
            self._close_session()

    def _init_session(self):
        if self.sess is not None:
            raise ValueError('Session is already initialized')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _close_session(self):
        if self.sess is None:
            raise ValueError('No session to close')

        self.sess.close()
        self.sess = None

    def print_percent_within(self, predicted, given_pcts):
        value_range = self.data.y_test.max() - self.data.y_test.min()

        for pct in given_pcts:
            within_range = np.abs(predicted - self.data.y_test) / value_range < pct
            percent_in_range = np.mean(within_range)
            print('Predictions within {}%% of actual: %.2f%%'.format(int(pct * 100)) % (percent_in_range * 100))

    def plot_cost(self, cost_history):
        steps = [step * COST_GRANULARITY for step in range(len(cost_history))]
        plt.plot(steps, cost_history)
        plt.axis([0, self.n_epochs, 0, np.max(cost_history)])
        plt.show()

    def plot_predictions(self, pred_y):
        y_test = self.data.y_test
        fig, ax = plt.subplots()
        ax.scatter(y_test, pred_y)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()

