import copy
import numpy as np

from transformlearner import TransformLearner

class FilterLearner(object):
    """Class for learning to identify individual components of a complex image filter"""

    base_config = {
        'name' : 'default',

        # Data initalization
        'create_data'   : False,
        'max_examples'  : None,
        'preprocess_fn' : None,
        'data_batch'    : 100,
        'img_height'    : 100,
        'img_width'     : 100,

        # DataSet parameters
        'batch_size'    : 100,
        'train_percent' : 0.6,
        'test_percent'  : 0.2,

        # Training parameters
        'n_epochs'          : 1000,
        'regularize'        : True,
        'ideal_rs'          : None,
        'normalize'         : True,
        'pca'               : True,
        'retained_variance' : 0.99
    }

    def __init__(self, transform_configs):
        self.transform_learners = [self.new_transform_learner(tl_config) for tl_config in transform_configs]
        self.transform_fns = [tl.transform_fn for tl in self.transform_learners]

    def new_transform_learner(self, tl_config):
        new_config = copy.deepcopy(self.base_config)
        new_config.update(tl_config)
        return TransformLearner(**new_config)

    def train(self, **kwargs):
        map(lambda tl: tl.train(**kwargs), self.transform_learners)

    def evaluate(self, eval_tls=False):
        if eval_tls:
            map(lambda tl: tl.evaluate(), self.transform_learners)
        else:
            for tl, i in enumerate(self.transform_learners):
                y_test = np.array(self.data.y_test[:,i], ndmin=2).T
                tl.evaluate(test_set=(self.data.X_test, y_test))

    def close_session(self):
        map(lambda tl: tl.close_session(), self.transform_learners)
