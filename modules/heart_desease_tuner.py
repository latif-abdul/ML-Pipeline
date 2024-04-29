"""Module Tuner"""

import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tfx.v1.components import TunerFnResult
from modules.heart_desease_trainer import (
    input_fn,
    get_model
)

def tuner_fn(fn_args):
    """Build the tuner using the KerasTuner API.
    Args:
      fn_args: Holds args used to tune models as name/value pairs.

    Returns:
      A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                      model , e.g., the training and validation dataset. Required
                      args depend on the above tuner's implementation.
    """
    # Memuat training dan validation dataset yang telah di-preprocessing
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    # Mendefinisikan strategi hyperparameter tuning
    tuner = kt.Hyperband(get_model,
                         objective='val_binary_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='TunerResult',
                         project_name='kt_hyperband')

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy",
        patience=5,
        min_delta=0.97,
    )

    fit_kwargs = {
        "callbacks": [stop_early],
        'x': train_set,
        'epochs': 10,
        'validation_data': val_set,
        'steps_per_epoch': fn_args.train_steps,
        'validation_steps': fn_args.eval_steps
    }

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs=fit_kwargs
    )
