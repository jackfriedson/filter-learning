import copy
import sys
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import utils

from transformlearner import TransformLearner

base_config = {
    'name' : 'default',

    # Data creation
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

def main():
    hue_config = copy.deepcopy(base_config)
    saturation_config = copy.deepcopy(base_config)
    bright_config = copy.deepcopy(base_config)
    contrast_config = copy.deepcopy(base_config)

    hue_config.update({
        'name'          : 'hue',
        'transform_fn'  : utils.adjust_hue,
        'preprocess_fn' : utils.hue_only,
        'ideal_rs'      : 335544.32
    })
    saturation_config.update({
        'name'          : 'saturation',
        'transform_fn'  : utils.adjust_saturation,
        'preprocess_fn' : utils.saturation_only,
        'ideal_rs'      : 2621.44
    })
    bright_config.update({
        'name'          : 'bright',
        'transform_fn'  : utils.adjust_brightness,
        'preprocess_fn' : utils.value_only,
        'ideal_rs'      : 20971.52
    })
    contrast_config.update({
        'name'          : 'contrast',
        'transform_fn'  : utils.adjust_contrast,
        'preprocess_fn' : utils.value_only,
        'ideal_rs'      : 327.68
    })

    # hue_learner = TransformLearner(**hue_config)
    sat_learner = TransformLearner(**saturation_config)
    # bright_learner = TransformLearner(**bright_config)
    # contrast_learner = TransformLearner(**contrast_config)

    # hue_learner.run()
    sat_learner.run()
    # bright_learner.run()
    # contrast_learner.run()

main()


