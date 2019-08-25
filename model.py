from keras.layers import Conv1D, Dense, Dropout, ZeroPadding1D, Input, MaxPool1D, BatchNormalization, Concatenate, \
    Flatten
from keras.activations import relu, softmax
from info import ModelParameters as Params, Constants
from math import log
from keras import Model
import numpy as np


def create_model():
    reps = int(log(Params.input_days, 2))

    num_filters = 8
    inputs = Input(shape=(Params.input_days, Params.info_dimension))
    x = Conv1D(num_filters, 1, padding='same')(inputs)
    x = BatchNormalization()(x)

    for i in range(reps):
        branch_filters = num_filters // 4

        branch_a = Conv1D(branch_filters, kernel_size=1, strides=1, padding='same', activation=relu)(x)

        branch_b = Conv1D(branch_filters, kernel_size=1, strides=1, padding='same', activation=relu)(x)
        branch_b = Conv1D(branch_filters, kernel_size=3, strides=1, padding='same', activation=relu)(branch_b)
        branch_b = Conv1D(branch_filters, kernel_size=3, strides=1, padding='same', activation=relu)(branch_b)

        branch_c = Conv1D(branch_filters, kernel_size=1, strides=1, padding='same', activation=relu)(x)
        branch_c = Conv1D(branch_filters, kernel_size=3, strides=1, padding='same', activation=relu)(branch_c)

        branch_d = MaxPool1D(3, strides=1, padding='same')(x)
        branch_d = Conv1D(branch_filters, kernel_size=1, strides=1, padding='same', activation=relu)(branch_d)

        x = Concatenate()([branch_a, branch_b, branch_c, branch_d])
        x = BatchNormalization()(x)

        num_filters = num_filters * 2
        x = ZeroPadding1D(1)(x)
        x = Conv1D(num_filters, kernel_size=2, strides=2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=.2)(x)

    x = Flatten()(x)
    x = Dense(64, activation=relu)(x)
    x = Dropout(rate=.2)(x)

    outputs = []
    for day in Params.days_to_predict:
        outputs.append(Dense(1, activation="sigmoid" if Params.predict_direction else None,
                             name=f"out_day_{day}")(x))

    return Model([inputs], outputs, name=Params.model_name)
