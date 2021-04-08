# from RETAIN.utils import activate


# from tensorflow.keras.utils.generic_utils import get_custom_objects

def predict_clip(y):
    return K.clip(y, 40, 500)
    
# get_custom_objects().update({'clipping': Activation(predict_clip)})

# ativation function
def clipping_parameter(ab,init_alpha=1.0, max_beta_value=1000, max_alpha_value=1000):
    
    from keras import backend as k
    a = ab[:, 0]
    b = ab[:, 1]

    # Implicitly initialize alpha:
    if max_alpha_value is None:
        a = init_alpha * k.exp(a)
    else:
        a = init_alpha * k.clip(x=a, min_value=k.epsilon(),max_value=max_alpha_value)

    m = max_beta_value
    if m > 1.05:  # some value >>1.0
        # shift to start around 1.0
        # assuming input is around 0.0
        _shift = np.log(m - 1.0)

        b = k.sigmoid(b - _shift)
    else:
        b = k.sigmoid(b)

    # Clipped sigmoid : has zero gradient at 0,1
    # Reduces the small tendency of instability after long training
    # by zeroing gradient.
    b = m * k.clip(x=b, min_value=k.epsilon(), max_value=1. - k.epsilon())

    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))

    return k.concatenate((a, b), axis=1)



def W_RETAIN(input_vector_size, 
             time_steps, 
             alpha_lstm_unit, 
             beta_lstm_unit, 
             reshape_size, 
             alpha_activation='relu', 
             beta_activation='relu', 
             kernel_regularizer=0.01, 
             embedding=False,
             return_seq=True):
    
    ''' Build w-Retain model 

    Parameters
    ----------
    input_vector_size: Int. size of vector in input variables. (=shape of shape 0)
    time_steps: Int. windowing
    embedding: Bool. whether you want to use embeddign layer at first layer 
    alpha_lstm_unit: Int. Positive integer, dimensionality of the output space.
    alpha_activation: Activation function to use.
        Default: 'relu'
    kernel_regularizer: Positive float. 
        Defualt: 0.01
    reshape_size: 


    Return
    ---------
    Model: keras model object
    
    
    Example
    ---------
    model = W_RETAIN(input_vector_size=45, 
                 time_steps=48, 
                 alpha_lstm_unit=5, 
                 beta_lstm_unit=5, 
                 reshape_size=45)
    model.summary()
    model.compile(optimizer=RMSprop(lr=1e-35), loss=weibull_loglik_discrete)
    '''
    
    
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Activation, Dense, Bidirectional, Input, Lambda, LSTM, TimeDistributed, Softmax, Multiply
    from tensorflow.keras.models import Model
    
    
    
    # Parameters check
    if type(input_vector_size) is not int:
        raise ValueError('Input size was not int')

    
    # Function define
    def reshape(data):
        '''Reshape the context vectors to 3D vector''' # 
        return K.reshape(x=data, shape=(-1, K.shape(data)[0], reshape_size)) # backend.shape(data)[0]

    
    # Reset graph
    K.clear_session()
    
    
    # Main
    if embedding == False:
        pass

    
    # Alpha(time-level weight) 
    alpha = Bidirectional(LSTM(alpha_lstm_unit, activation=alpha_activation, implementation=2, return_sequences=True), name='alpha') 
    alpha_dense = Dense(1, kernel_regularizer=regularizers.l2(kernel_regularizer))

    # Beta (variable level weight)
    beta = Bidirectional(LSTM(beta_lstm_unit, activation=beta_activation, 
                                            implementation=2, return_sequences=True), name='beta') 
    
    beta_dense = Dense(input_vector_size, activation='sigmoid', 
                              kernel_regularizer=regularizers.l2(kernel_regularizer))

    # Output layer
    output_layer = Dense(2, activation='sigmoid', kernel_regularizer=regularizers.l2(kernel_regularizer),  name='output')
    
    # Output layer2 - pdf sequence
    output_pdf_seq = Dense(time_steps, name='pdf_seq')
    
    # Model define
    x_input = Input(shape=(time_steps, input_vector_size), name='X') # feature
        
    # 2-1. alpha
    alpha_out = alpha(x_input)
    alpha_out = TimeDistributed(alpha_dense, name='alpha_dense')(alpha_out) 
    alpha_out = Softmax(axis=1, name='alpha_softmax')(alpha_out) # 논문 본문에 alpha1, alph2, alph3..을 의미

    # 2-2. beta
    beta_out = beta(x_input)
    beta_out = TimeDistributed(beta_dense, name='beta_dense')(beta_out) # 논문 내 beta1 ,beta2, beta3을 의미.

    # 3. Context vector
    c_t = Multiply()([alpha_out, beta_out, x_input])
    c_t = Lambda(lambda x: K.sum(x, axis=1) , name='Sum_Context')(c_t) 

    # 4. Fully connected layer
    if return_seq == True:
        output_final = output_pdf_seq(context) # generated weibull distribution as time size
        output_final = Lambda(lambda x: K.reshape(x, shape=(-1, time_steps)), name='output_lambda')(output_final)
    else:
        output_final = Dense(2, name='dense')(c_t)
        output_final = Activation(clipping_parameter, name='Activation')(output_final)


        
    model = Model(x_input , output_final)
    return model




def get_noom_retain(input_vector_size, time_size, alpha_lstm_unit, beta_lstm_unit, 
             reshape_size, 
             alpha_activation='relu', 
             beta_activation='relu', 
             kernel_regularizer=0.01, 
             embedding=False,
             return_seq=True):
    
    ''' Build w-Retain model 

    Parameters
    ----------
    input_vector_size: Int. size of vector in input variables. (=shape of shape 0)
    time_size: Int. windowing
    embedding: Bool. whether you want to use embeddign layer at first layer 
    alpha_lstm_unit: Int. Positive integer, dimensionality of the output space.
    alpha_activation: Activation function to use.
        Default: 'relu'
    kernel_regularizer: Positive float. 
        Defualt: 0.01
    reshape_size: 


    Return
    ---------
    Model: keras model object
    
    
    Example
    ---------
    model = W_RETAIN(input_vector_size=45, 
                 time_size=48, 
                 alpha_lstm_unit=5, 
                 beta_lstm_unit=5, 
                 reshape_size=45)
    model.summary()
    model.compile(optimizer=RMSprop(lr=1e-35), loss=weibull_loglik_discrete)
    '''
    
    
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Activation
    from tensorflow.keras import layers, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers # To use regularization at each layer
    
    
    # Parameters check
    if type(input_vector_size) is not int:
        raise ValueError('Input size was not int')

    
    # Function define
    def reshape(data):
        '''Reshape the context vectors to 3D vector''' # 
        return K.reshape(x=data, shape=(K.shape(data)[0], 1, reshape_size)) # backend.shape(data)[0]

    
    # Reset graph
    K.clear_session()
    
    
    # Main
    if embedding == False:
        pass

    
    # Alpha(time-level weight) 
    alpha = layers.Bidirectional(layers.LSTM(alpha_lstm_unit, activation=alpha_activation, implementation=2, return_sequences=True), name='alpha') 
    alpha_dense = layers.Dense(1, kernel_regularizer=regularizers.l2(kernel_regularizer))

    # Beta (variable level weight)
    beta = layers.Bidirectional(layers.LSTM(beta_lstm_unit, activation=beta_activation, implementation=2, return_sequences=True), name='beta') 
    beta_dense = layers.Dense(input_vector_size, activation='sigmoid', kernel_regularizer=regularizers.l2(kernel_regularizer))

    
    # Model define
    x_input = Input(shape=(time_size, input_vector_size), name='X') # feature
    time_inv_input = Input(shape=(1,5), name='time_inv')
    
    # 2-1. alpha
    alpha_out = alpha(x_input)
    alpha_out = layers.TimeDistributed(alpha_dense, name='alpha_dense')(alpha_out) # bidrection 층의 출력에 한번에 filter size 1짜리 FC을 함. 
    alpha_out = layers.Softmax(axis=1, name='alpha_softmax')(alpha_out) # 논문 본문에 alpha1, alph2, alph3..을 의미

    # 2-2. beta
    beta_out = beta(x_input)
    beta_out = layers.TimeDistributed(beta_dense, name='beta_dense')(beta_out) # 논문 내 beta1 ,beta2, beta3을 의미.

    # 3. Context vector
    c_t = layers.Multiply()([alpha_out, beta_out, x_input])
    context = layers.Lambda(lambda x: K.sum(x, axis=1) , name='Context_lambda_sum')(c_t) 
    context = layers.Lambda(reshape, name='contextReshaped')(context) # Reshape to 3d vector for consistency between Many to Many and Many to One 
    
    
    # concat 
    c_concat = layers.concatenate([context, time_inv_input])
    output = layers.Dense(1, name='output')(c_concat)
    output = layers.Lambda(lambda x: K.reshape(x, shape=(-1,1)))(output)

    model = Model([x_input, time_inv_input] , output)
    return model