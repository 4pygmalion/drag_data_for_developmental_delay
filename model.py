import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l1

def _conv_block(input_tensor, filters, dropout=False, kernel_size=3, L1_PENALTY=0.0001):
    ''' Create Conv+MaxPool block

    Parameters
    ----------
    input_tensor: tf.Tensor
    fitlers: int
    kernel_size: int

    Return
    ------
    tf.Tensor

    '''
    conv_layer = Conv1D(filters,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same',
                        activation='selu',
                        activity_regularizer=l1(L1_PENALTY))(input_tensor)
    x = MaxPooling1D(pool_size=2)(conv_layer)

    if dropout == True:
        x = Dropout(0.4)(x)
    return x


def build_1DCNN(timestamp, n_feature_ts, n_feature_aux):
    '''

    Returns
    -------
    tf.keras.model.Model
    '''
    x_main = Input(shape=(timestamp, n_feature_ts), dtype='float32', name='main')
    x = _conv_block(x_main, filters=32)
    x = _conv_block(x, filters=64, kernel_size=3, dropout=False)
    x = _conv_block(x, filters=32, kernel_size=3, dropout=False)
    x = _conv_block(x, filters=16, kernel_size=3, dropout=False)
    x = _conv_block(x, filters=8, kernel_size=2, dropout=True)
    x = _conv_block(x, filters=4, kernel_size=2, dropout=False)
    x = GlobalAvgPool1D()(x)

    aux_input = Input(shape=(n_feature_aux,))
    aux_x = RepeatVector(150)(aux_input)
    aux_x = _conv_block(aux_x, filters=64, kernel_size=3, dropout=False)
    aux_x = _conv_block(aux_x, filters=32, kernel_size=2, dropout=True)
    aux_x = _conv_block(aux_x, filters=16, kernel_size=2, dropout=False)
    aux_x = _conv_block(aux_x, filters=8, kernel_size=2, dropout=True)
    aux_x = _conv_block(aux_x, filters=4, kernel_size=2, dropout=False)
    aux_x = GlobalAveragePooling1D()(aux_x)

    con_x = concatenate([x, aux_x])
    con_x = Dense(2, activation='sigmoid')(con_x)

    model = tf.keras.models.Model([x_main, aux_input], con_x)

    return model
