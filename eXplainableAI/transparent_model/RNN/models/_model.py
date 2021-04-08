# -*- coding: utf-8 -*-
# written by Ho Heon kim
# last update : 2020.08.26

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras import Model, regularizers


class RETAIN(object):

    def __init__(self, config):
        '''
        Parameters
        ----------
        config: Dict
            key: 'n_features', 'steps', 'hidden_units'

        '''
        self.n_features = config['n_features']
        self.steps = config['steps']
        self.hidden_units = config['hidden_units']
        

    def build_model(self,
                    name='base_model',
                    l2_penalty=0.01,
                    beta_activation='tanh',
                    Bidirectional_activation='relu',
                    LSTM_initializer='glorot_uniform',
                    kernel_initializer='random_uniform',
                    predict='regression',
                    ):
        '''
        Build model with tensorflow.keras framework
        
        Parameters
        ----------
        use_x_aux: bool. whether use conditional RNN



        LSTM_initializer: str

            (default: glorot_uniform)
            - 'he_normal'
            - 'random_uniform'
            - 'Constant'
            - 'Zeros'
            - 'Ones'
            - 'RandomNormal'

            see. https://keras.io/initializers/


        kernel_initializer: str

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

        Returns
        -------
        model: tensorflow.keras.Model
        '''


        # Definition
        def reshape(data):
            '''Reshape the context vectors to 3D vector'''  #
            return K.reshape(x=data, shape=(-1, self.n_features))  # backend.shape(data)[0]

        alpha = Bidirectional(LSTM(self.hidden_units ,
                                   activation=Bidirectional_activation,
                                   implementation=2,
                                   return_sequences=True,
                                   kernel_initializer=LSTM_initializer,
                                   activity_regularizer=regularizers.l2(l2_penalty)), name='alpha')

        alpha_dense = Dense(1, activity_regularizer=regularizers.l2(l2_penalty))

        beta = Bidirectional(LSTM(self.hidden_units ,
                                  activation=Bidirectional_activation,
                                  implementation=2,
                                  return_sequences=True,
                                  kernel_initializer=LSTM_initializer,
                                  activity_regularizer=regularizers.l2(l2_penalty)), name='beta')

        beta_dense = Dense(self.n_features, activation=beta_activation)

        # Regression:
        if predict == 'regression':
            output_layer = Dense(1,
                                 kernel_regularizer=regularizers.l2(l2_penalty),
                                 kernel_initializer=kernel_initializer, name='output')
        # with classification
        else:
            output_layer = Dense(1, activation='sigmoid', name='output')


        # -- main : operation --
        x_input = Input(shape=(self.steps, self.n_features), name='X')  # feature

        # 2-1. alpha
        alpha_out = alpha(x_input)
        alpha_out = TimeDistributed(alpha_dense, name='alpha_dense')(alpha_out)
        alpha_out = Softmax(axis=1, name='alpha_softmax')(alpha_out)  # 논문 본문에 alpha1, alph2, alph3..을 의미

        # 2-2. beta
        beta_out = beta(x_input)
        beta_out = TimeDistributed(beta_dense, name='beta_dense')(beta_out)  # 논문 내 beta1 ,beta2, beta3을 의미.

        # 3. Context vector
        c_t = Multiply()([alpha_out, beta_out, x_input])
        c_t = Lambda(lambda x: K.sum(x, axis=1), name='lamdaSum')(c_t)

        # Output layer
        output_final = output_layer(c_t)

        # Model
        model = Model(x_input, output_final, name=name)

        return model



class ConditionalRETAIN(RETAIN):
    '''
    Conditional RNN type of RETAIN
    '''

    def __init__(self, config):
        ''' Conditional RNN type for RETAIN

        Parameters
        ----------
        config: Dict
            key: 'n_features', 'steps', 'hidden_units', 'n_auxs'
            
            
        Attribution
        -----------
        self.n_features = config['n_features']
        self.steps = config['steps']
        self.hidden_units = config['hidden_units']

        '''
        RETAIN.__init__(self, config)
        self.n_axus = config['n_auxs']
        
        
    def build_model(self,
                    name='base_model',
                    problem='regression',
                    l2_penalty=0.25,
                    beta_activation='tanh',
                    Bidirectional_activation='relu',
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='random_uniform',
                    other_initializer='zeros'
                    ):
        
        
        
        def reshape(data):
            '''Reshape the context vectors to 3D vector''' # 
            return K.reshape(x=data, shape=(-1, x_time_vect_size)) # backend.shape(data)[0]

        alpha = Bidirectional(LSTM(self.hidden_units,
                                   activation=Bidirectional_activation, 
                                   implementation=2, 
                                   return_sequences=True,
                                   kernel_initializer=kernel_initializer,
                                   activity_regularizer=regularizers.l2(l2_penalty)), name='alpha') 

        alpha_dense = Dense(1, activity_regularizer=regularizers.l2(l2_penalty))

        beta = Bidirectional(LSTM(self.hidden_units,
                                  activation=Bidirectional_activation, 
                                  implementation=2, 
                                  return_sequences=True,
                                  kernel_initializer=kernel_initializer,
                                  activity_regularizer=regularizers.l2(l2_penalty)), name='beta') 

        beta_dense = Dense(self.n_features, activation=beta_activation)

        # Regression:
        if problem == 'regression':
            output_layer = Dense(1, kernel_regularizer=regularizers.l2(l2_penalty), kernel_initializer=other_initializer, name='output')
        else:
            output_layer = Dense(1, activation='sigmoid', name='output')

        # Model define
        x_input = Input(shape=(self.steps, self.n_features), name='X') # feature
        x_input_fixed = Input(shape=(self.n_axus), name='x_input_fixed')

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
        output_final = output_layer(c_concat)     

        # Model
        model = Model([x_input, x_input_fixed] , output_final, name=name)

        return model

    
class VariableWiseRETAIN(RETAIN):

    def __init__(self, config):
        ''' Class for Mixed Uni-variate RETAIN

        Parameters
        ----------
        config: Dict
            key: 'n_features', 'steps', 'hidden_unit'

        '''
        RETAIN.__init__(self, config)


    def make_listed_x(self, X):
        '''
        to make list including each variable to fit model

        Parameters
        ----------
        X: np.array, pd.DataFrame


        Return
        ------
        variable_list: list including each col_idx

        Example
        -------
        # model configuration
        >>> config = {'n_features':5, 'step': 20, 'hidden_units':10}

        # model build
        >>> vw_retain = VariableWiseRETAIN(config)
        >>> model = vw_retain.build_model()

        # compile & fit
        >>> model.compile(RMSprop(lr=0.0005), loss='binary_crossentropy')
        >>> X = vw_retain.make_listed_x(X)
        >>> model.fit(X, y)

        '''

        # for input validity
        if np.ndim(X) != 3:
            raise ValueError('Numpy X is not 3 dimensional')

        # --main --
        variable_list = []

        for col_idx in range(self.n_features):
            variable = X[:, :, col_idx]
            variable = variable[:, :, np.newaxis]   # (none, timestamp, 1)
            variable_list.append(variable)

        return variable_list



    def build_model(self):

        input_layer_list = []
        context_list=[]

        # Create uni-variate retain for i-th variable
        for col_idx in range(self.n_features):

            # -- layer definition --
            # input layer
            input_layer = Input(shape=(self.steps, 1), name='input_variable_{}'.format(col_idx+1))

            # lstm layer
            alpha_lstm = Bidirectional(LSTM(units=self.hidden_units,
                                            return_sequences=True),
                                            name='alpha_lstm_variable_{}'.format(col_idx+1))
            beta_lstm = Bidirectional(LSTM(units=self.hidden_units,
                                           return_sequences=True),
                                           name='beta_lstm_variable_{}'.format(col_idx+1))

            # attention layer
            alpha_dense = Dense(1, name='alpha_dense_variable_{}'.format(col_idx+1))
            beta_dense = Dense(1, activation='tanh', name='alpha_dense_variable_{}'.format(col_idx+1))

            # output layer
            output_layer = Dense(1, activation='sigmoid', name='output_layer_variable_{}'.format(col_idx+1))

            # -- operation --
            # compute alpha attention
            g = alpha_lstm(input_layer)
            alpha_out = TimeDistributed(alpha_dense)(g)
            alpha_out = Softmax(axis=1, name='alpha_variable_{}'.format(col_idx+1))(alpha_out)

            # compute beta attention
            h = beta_lstm(input_layer)
            beta_out = TimeDistributed(beta_dense, name='beta_variable_{}'.format(col_idx+1))(h)

            # compute context vector
            c_t = Multiply(name='univariate_context_vec_{}'.format(col_idx+1))([alpha_out, beta_out, input_layer])


            input_layer_list.append(input_layer)
            context_list.append(c_t)


        # Context Matrix
        matrix_c_t = concatenate(context_list, name='concated_c_t')     # shape: (none, time, variable)
        sum_c_t = Lambda(lambda x: K.sum(x, axis=1), name='context_reduce_sum_1')(matrix_c_t)
        vw_output = Dense(units=1, activation='sigmoid')(sum_c_t)

        model = Model(input_layer_list, vw_output)
        return model


class MultiHeadRETAIN(RETAIN):

    def __init__(self, config):
        RETAIN.__init__(self, config)









