import numpy as np
import pandas as pd
from random import sample
from math import floor

class TimeSeriesTrimming(object):

    def __init__(self, data):
        '''
        Basic class for Timeseries Trimming

        Parameters
        ----------
        data : pd.DataFrame with index (person, time)

        '''

        self.data = data.sort_index()
        self._validing_index()  # Validity


    def _validing_index(self):
        if (self.data.index.name is None) and (self.data.index.names is None):
            raise IndexError('The DataFrame did not have index')

    def unique_by_date(self, col):
        '''
        Parameters
        ----------
        col : cycle
        :return:
        '''
        return self.data.groupby(col).first()



    def backward_trimming(self, event_dates):
        '''
        Parameters
        ----------
        event_dates: array-like
            require same order with dataframe

        Return
        ------
        pd.DataFrame
        '''

        # validity
        person = sorted(list(set(self.data.index.get_level_values(0))))
        n_person = len(person)
        if n_person != len(event_dates):
            raise ValueError('length of event_date not match with n of person')

        # loop
        histories = []
        for pt, last_date in zip(person, event_dates):
            person_history = self.data.loc[[pt]]
            rows = person_history.loc[person_history.index.get_level_values(1) <= last_date]
            histories.append(rows)

        return pd.concat(histories)



class BackwardTrimming(TimeSeriesTrimming):

    def __init__(self, data):
        super().__init__(data)
        super()._validing_index()





















def oversampling_with_idx(train_idx, Y):
    '''Oversampling with index (Binary classification)
    
    Parameters
    ----------
    train_idx : array-like (list)

    Y0: pd.DataFrame
        ---columns---
        * index 
        * Y label
        
    Return
    ------
    Balanced list
    
    '''
    
    # indexing
    if Y.index is None:
        Y_cols = Y.columns
        Y.set_index(Y_cols[0])
        
    # Y0, Y1
    Y = Y.loc[train_idx]
    Y0 = list(Y.loc[Y == 0].index)
    Y1 = list(Y.loc[Y == 1].index)
    
    # Multiple
    n_Y0 = len(Y0)
    n_Y1 = len(Y1)
    if n_Y0 > n_Y1:
        multiple = floor(n_Y0 / n_Y1)
        remain = n_Y0 - (multiple * n_Y1)
        Y1 = (multiple * Y1) + Y1[remain:]
        
    return Y0 + Y1
    



def over_sampling(train_idx, Y0, Y1, use_generator=True):
    '''make Train_idx balanced with oversampling
    Y_0: list of y0
    
    
    Parameters
    ----------
    train_idx : array-like (list)
    
    Y0: array-like (list)
        ---columns---
        * index 
        * Y label
    
    Y0: pd.DataFrame 
    
    returns
    ------
    list. train idx with balanced Y0, Y1 data
    
    '''

    
    Y0, Y1 = list(Y0), list(Y1)
    train_Y0 = [i for i in train_idx if i in Y0]
    train_Y1 = [i for i in train_idx if i in Y1]
    n_Y0, n_Y1 = len(train_Y0), len(train_Y1)
    
    if n_Y0 >= n_Y1:
        n_oversampling =  max(n_Y0, n_Y1) - n_Y1
        
        if n_Y1 >= n_oversampling:
            added =  sample(train_Y1, n_oversampling)  # 추가적으로 뽑은 인덱스
            overed_list = list(train_idx) + list(added)
        else:
            n_duplicated = floor(n_Y0/n_Y1) # 150: 30이어서 30보다 더 많이 뽑아야 하는 경우
            n_oversampling = n_Y0 - (n_Y1 * n_duplicated)
            
            added =  train_Y1*(n_duplicated-1) + list(sample(train_Y1, n_oversampling)  )
            overed_list = list(train_idx)+ list(added)
    else:
        # Y1이 많은 경우
        n_oversampling =  max(n_Y0, n_Y1) - n_Y0
        
        if n_Y0 >= n_oversampling:
            added =  sample(train_Y0, n_oversampling)  # 추가적으로 뽑은 인덱스
            overed_list = list(train_idx) + list(added)
        else:
            n_duplicated = floor(n_Y1/n_Y0) # 150: 30이어서 30보다 더 많이 뽑아야 하는 경우
            n_oversampling = n_Y1 - (n_Y0) * n_duplicated
            added =  train_Y0*(n_duplicated-1) + list(sample(train_Y0, n_oversampling))
            overed_list = list(train_idx) + list(added)
            
    return overed_list
    

class StratifiedKfold(object):
    
    def __init__(self, n_split):
        self.n_split = n_split

    def _shuffle(self, data):
        ''' Shuffle data

        Parameters
        ----------
        data : pd.DataFrame

        Return
        -----
        shuffled pd.DataFrame
        '''

        import pandas as pd
        return data.iloc[np.random.RandomState(seed=42).permutation(len(data))]

    def index_split(self, Y):
        ''' 나머지가 남는 경우는 마지막 폴드에서 n 수를 조금더 가져감

        Parameters
        ----------
        Y : pd.DataFrame


        Returns
        ------
        list including indexes of Y1 , Y0
        '''

        from random import sample
        from math import floor

        # stratified
        Y1 = Y.loc[Y == 1]
        Y1 = list(self._shuffle(Y1).index)
        Y0 = Y.loc[Y == 0]
        Y0 = list(self._shuffle(Y0).index)

        # Sampling
        n_sample_y0 = floor(len(Y0) / self.n_split)
        n_sample_y1 = floor(len(Y1) / self.n_split)

        # K-fold

        while True:
            for k in range(self.n_split):

                # test set

                if k != self.n_split-1:   # for last fold
                    test_y0 = set(Y0[k * n_sample_y0 : (k+1) * n_sample_y0])
                    test_y1 = set(Y1[k * n_sample_y1 : (k+1) * n_sample_y1])
                    test = test_y0.union(test_y1)
                else:
                    test_y0 = set(Y0[k * n_sample_y0: ])
                    test_y1 = set(Y1[k * n_sample_y1: ])
                    test = test_y0.union(test_y1)
                
                # Train
                train_y0 = set(Y0) - test_y0
                train_y1 = set(Y1) - test_y1
                train = train_y0.union(train_y1)
                
                yield train, test


def data_generator(instnace_idx, x_time, x_aux, y_data):
    '''Train generation for Keras.Model.fit_generator method in Keras

    Parameters
    ----------
    train_idx: array-like, list

    x_data: pd.DataFrame. time-variant data
        * index: patient_id, date
    x_fixed: pd.DataFrame. time-fixed data.
        * index : patient_id
        (1D data)
    y_data: pd.DataFrame.
        * index : patient_id


    Returns
    -------
    Generator
    '''
    while True:
        for instance in instnace_idx:
            # 1. time variant
            x_instance = np.array(x_time.loc[instance])
            time_x = x_instance.reshape(1, -1, x_instance.shape[1])

            # 2. time fixed
            x_fixed_arr = np.array(x_aux.loc[instance])
            x_fixed_arr = x_fixed_arr.reshape(1, -1)

            # 3. Y label
            y_ = np.array(y_data.loc[instance]).reshape(1, -1)

            yield [time_x, x_fixed_arr], y_


def train_generator(train_idx, x_time, x_aux, y_data, use_balance=False, Y0=None, Y1=None):
    '''Train generation for Keras.Model.fit_generator method in Keras
    
    Parameters
    ----------
    train_idx: array-like, list
    
    x_data: pd.DataFrame. time-variant data 
        * index: patient_id, date
    x_fixed: pd.DataFrame. time-fixed data. 
        * index : patient_id
        (1D data)
    y_data: pd.DataFrame.
        * index : patient_id
        
        
        
    Returns
    -------
    Generator 
    '''
    
    assert (x_time.index.name is not None) or (x_time.index.names is not None), 'x_data must be indexed'
    assert (x_aux.index.name is not None) or (x_aux.index.names is not None), 'x_data_fixed must be indexed'
    
       
    # Generator: main (sapmling)
    if use_balance == False:
        while True:
            for instance in train_idx:
                # print(instance)
                # 1. time variant
                x_instance = np.array(x_time.loc[instance])
                time_x = x_instance.reshape(1, -1, x_instance.shape[1])

                # 2. time fixed
                x_fixed_arr = np.array(x_aux.loc[instance])
                x_fixed_arr = x_fixed_arr.reshape(1, -1)
                
                # 3. Y label
                y_ = np.array(y_data.loc[instance]).reshape(1, -1)
                
                yield [time_x, x_fixed_arr], y_, 
    else:
        train_idx = idx_over_sampling(train_idx, Y0, Y1)
        
        while True:
            for instance in train_idx:

                # 1. time variant
                x_instance = x_time.loc[instance].fillna(method='backfill') # x_data imputation
                x_instance = x_instance.fillna(0)
                time_x = np.array(x_instance)
                time_x = time_x.reshape(1, -1, time_x.shape[1])



                # 2. time fixed
                x_fixed_arr = np.array(x_aux.loc[instance])
                x_fixed_arr = x_fixed_arr.reshape(1, -1)
                
                # 3. Y label
                y_ = np.array(y_data.loc[instance]).reshape(-1, 1)

                yield [time_x, x_fixed_arr], np.array(y_)






def mean_absolute_percentage_error(y_true, y_pred): 
    '''
    See Also
    --------
    https://stackoverflow.com/questions/47266383/save-and-load-weights-in-keras
    '''
    from sklearn.utils import check_array
    import numpy as np

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def make_realtime_seq(accessCode, dataframe, var, window_size=16, moving_obs=16):
    '''To make reatime seq
    
    
    Parameters
    ----------
    dataframe: pd.DataFrame with indexing + (None, 16 columns)
        ----- columns -----
        * accessCode: index
        * week	wt_adherence	meal_adherence	meal_color	calories	steps	n_exercise	n_drink	n_overcal
        * 1 ~ 16 week
        
    var: list. including variable name in DataFrame
    
    
    
    
    Returns
    ------
    np.array
    '''
    
    assert dataframe.index.names is not None, print('DataFrame must have index')
    

    # 데이터 전처리
    dataframe_ = dataframe.copy()
    dataframe_ = dataframe_.replace('week', '')
    user_log = dataframe_.loc[accessCode]
    
    
    
    # Real-time array
    empty_core = np.zeros((window_size, len(var)))
    empty = np.zeros((window_size, len(var)))
    
    
    # Loop
    for i in range(0, moving_obs): 
        if i < window_size:
            empty[window_size-1-i:, :] = user_log[var].iloc[0:i+1]  # max(i) = 47
            empty_core = np.append(empty_core, empty, axis=0)
        else:
            try:
                empty[:,:] = np.array(user_log[var].iloc[i-self.window_size:i])
                empty_core = np.append(empty_core, empty, axis=0)
            except:
                pass
    empty_core = empty_core[window_size:,:]

    return empty_core.reshape(-1, window_size, len(var))


def make_dup(arr, shape=(1, 16)):
    '''1-d array -> shape array로 broadcast'''
    empty = np.ones(shape=shape)
    
    for i in range(len(arr)):
        ones = np.ones(shape=(1, 16))
        target = ones * arr[i]
        empty = np.vstack([empty, target])
    empty = empty[1:,:]
    return empty



def weibull_loglik_continuous(y_true, ab_pred, name=None):
    ''' negtaive weibull log-likelihood 
    
    Parameters 
    ----------
    y_true: tuple. 
    ab_pred: tuple
    
    Returns
    -------
    float
    
    '''
    from keras import backend as K
    y_ = y_true[:, 0]  # actual event time
    u_ = y_true[:, 1]  # censor or uncensor indicator
    a_ = ab_pred[:, 0] # predicted alpha parameter (shape)
    b_ = ab_pred[:, 1] + 1e-35 # predicted beta parameter (scale)
    
    ya = (y_ + 1e-35) / a_  # elementwise divided. ya = Y_t / alpha  
    
    element1 = u_ * (b_ * K.log(b_+ 1e-35 ) + b_ * K.log(ya+ 1e-35 ))
    element2 = K.pow(ya, b_)
    return - 1 * K.mean(element1 - element2)  # minimize: negative log-likelihood 



def weibull_loglik_discrete(y_true, ab_pred, name=None):
    ''' negtaive weibull log-likelihood for discrete timeseries
    
    Parameters 
    ----------
    y_true: tuple. 
    ab_pred: tuple
    
    Returns
    -------
    float
    
    '''
    from keras import backend as K
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0] + 1e-35
    b_ = ab_pred[:, 1] 

    hazard0 = K.pow((y_ + 1e-35) / a_, b_)
    hazard1 = K.pow((y_ + 1) / a_, b_)

    return -1 * K.mean(u_ * K.log(K.exp(hazard1 - hazard0) - 1.0) - hazard1)

def activate(ab_pred):
    '''
     Keras doesn't support applying different activation functions to the individual neurons. 
     Thankfully, a custom activation function takes care of this...
    '''
    from keras import backend as K
    
    a = K.exp(ab_pred[:, 0])
    b = K.softplus(ab_pred[:, 1])

    a = K.reshape(a, (K.shape(a)[0], 1))
    b = K.reshape(b, (K.shape(b)[0], 1))

    return K.concatenate((a, b), axis=1)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          fontsize=15):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=fontsize,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=fontsize,
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    


def getPDF(result):
    idset = []
    for i in range(0, int(result.shape[0]/48)):
        for j in range(0, 48):
            idset.append(i+1)
    idset = pd.DataFrame(idset)
    result = pd.concat([idset, result], axis=1)
    result.columns = ["id" ,"tau_tte","censored","scale","shape"]
    x= result.ix[:,1]
    shape = result.ix[:,4]
    scale = result.ix[:,3]
    result["pdf"]= (shape / scale) * (x / scale)**(shape - 1) * np.exp(-(x / scale)**shape)
    return result