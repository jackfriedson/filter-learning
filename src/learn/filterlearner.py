import copy

from transformlearner import TransformLearner

class FilterLearner(object):
    """Class for learning to identify individual components of a complex image filter"""

    base_config = {
        'name' : 'default',

        # Data initalization
        'create_data'   : False,
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
        self.transform_learners = [self.new_transform_learner(config) for config in transform_configs]

    def new_transform_learner(self, config):
        new_config = copy.deepcopy(self.base_config)
        new_config.update(config)
        return TransformLearner(**new_config)

    def train(self, **kwargs):
        map(lambda tl: tl.train(**kwargs), self.transform_learners)

    def evaluate(self):
        map(lambda tl: tl.evaluate(), self.transform_learners)

    def close_session(self):
        map(lambda tl: tl.close_session(), self.transform_learners)
