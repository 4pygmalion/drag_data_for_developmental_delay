# -*- coding: utf-8 -*-
# written by Ho Heon kim 
# last update : 2020.07.09

from tensorflow.keras import backend as K
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, InputLayer, Input, Bidirectional, Softmax, Multiply, Lambda, concatenate, Activation
from tensorflow.keras.optimizers import RMSprop

class RETAIN(object):
    ''' default retain model

    '''

def build_RETAIN(x_time_vect_size = 40,
           x_aux_vect_size = 7,
           alpha_lstm_unit = 10, 
           beta_lstm_unit = 10,
           timestamp = 91,
           use_mid_dense=False,
           l2_penalty=0.01,
           beta_activation='tanh',
           Bidirectional_activation = 'relu',
           LSTM_initializer = 'glorot_uniform',
           FC_initializer = 'random_uniform',
           predict='regression',
           use_extra_dense=False,
           ):
    
    '''
    Parameters
    ----------
    input_vector_size: int.
    
    
    LSTM_initializer: str
    
        (default: glorot_uniform)
        - 'he_normal'
        - 'random_uniform'
        - 'Constant'
        - 'Zeros'
        - 'Ones'
        - 'RandomNormal'
        
        see. https://keras.io/initializers/
        
        
    FC_initializer: str
    
        (default: glorot_uniform)
        - 'he_normal'
        - 'random_uniform'
        - 'Constant'
        - 'Zeros'
        - 'Ones'
        - 'RandomNormal'
        
        see. https://keras.io/initializers/
        
    beta_activation: str.
        tanh
        tan
        linear
    
    l2_penalty: float
    
    predict: str.
        (default: regession)
        - 'classification': for binary classification
        - 'regression': for regression model
    '''
    

    def reshape(data):
        '''Reshape the context vectors to 3D vector''' # 
        return K.reshape(x=data, shape=(-1, x_time_vect_size)) # backend.shape(data)[0]


    alpha = Bidirectional(LSTM(alpha_lstm_unit,
                               activation=Bidirectional_activation, 
                               implementation=2, 
                               return_sequences=True,
                               kernel_initializer=LSTM_initializer,
                               activity_regularizer=regularizers.l2(l2_penalty)), name='alpha') 
    
    alpha_dense = Dense(1, activity_regularizer=regularizers.l2(l2_penalty))
    
    beta = Bidirectional(LSTM(beta_lstm_unit, 
                              activation=Bidirectional_activation, 
                              implementation=2, 
                              return_sequences=True,
                              kernel_initializer=LSTM_initializer,
                              activity_regularizer=regularizers.l2(l2_penalty)), name='beta') 
    
    beta_dense = Dense(x_time_vect_size, activation=beta_activation)
    
    mid_dense = Dense(8, kernel_regularizer=regularizers.l2(l2_penalty), kernel_initializer=FC_initializer, name='mid_output')
    
    # Regression:
    if predict == 'regression':
        output_layer = Dense(1, kernel_regularizer=regularizers.l2(l2_penalty), kernel_initializer=FC_initializer, name='output')
    else:
        output_layer = Dense(1, activation='sigmoid', name='output')

    
    # Model define
    x_input = Input(shape=(timestamp, x_time_vect_size), name='X') # feature
    x_input_fixed = Input(shape=(x_aux_vect_size,), name='x_input_fixed')

    # 2-1. alpha
    alpha_out = alpha(x_input)
    alpha_out = TimeDistributed(alpha_dense, name='alpha_dense')(alpha_out) 
    alpha_out = Softmax(axis=1, name='alpha_softmax')(alpha_out) # 논문 본문에 alpha1, alph2, alph3..을 의미

    # 2-2. beta
    beta_out = beta(x_input)
    beta_out = TimeDistributed(beta_dense, name='beta_dense')(beta_out) # 논문 내 beta1 ,beta2, beta3을 의미.

    # 3. Context vector
    c_t = Multiply()([alpha_out, beta_out, x_input])
    context = Lambda(lambda x: K.sum(x, axis=1) , name='lamdaSum')(c_t) 
    
    # Output layer
    c_concat = concatenate([context, x_input_fixed])
    
    if use_extra_dense:
        dense1 = Dense(100, activation=None, name='fc1')
        dense2 = Dense(20, activation=None, name='fc2')
        dense3 = Dense(10, activation=None, name='fc3')
        
        dense1_out = dense1(c_concat)
        dense2_out = dense2(dense1_out)
        c_concat   = dense3(dense2_out)


    output_final = output_layer(c_concat)     

    # Model
    model = Model([x_input, x_input_fixed] , output_final)
    
    return model
    
    
