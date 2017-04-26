import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import utils

from filterlearner import FilterLearner

transform_configs = [{
    'name'          : 'hue',
    'transform_fn'  : utils.adjust_hue,
    'preprocess_fn' : utils.hue_only,
    'ideal_rs'      : 335544.32
},
{
    'name'          : 'saturation',
    'transform_fn'  : utils.adjust_saturation,
    'preprocess_fn' : utils.saturation_only,
    'ideal_rs'      : 2621.44
},
{
    'name'          : 'bright',
    'transform_fn'  : utils.adjust_brightness,
    'preprocess_fn' : utils.value_only,
    'ideal_rs'      : 20971.52
},
{
    'name'          : 'contrast',
    'transform_fn'  : utils.adjust_contrast,
    'preprocess_fn' : utils.value_only,
    'ideal_rs'      : 327.68
}]

def main():
    filter_learner = FilterLearner(transform_configs)
    filter_learner.train(persist_session=True)
    filter_learner.evaluate()
    filter_learner.close_session()

main()


