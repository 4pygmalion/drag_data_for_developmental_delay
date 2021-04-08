#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Written by Jae il Ahn.
# Co-worker: Ho Heon Kim.  hoheon0509@gmail.com 



import pandas as pd
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D, concatenate, Dropout, Dense, Flatten, Bidirectional, LSTM, TimeDistributed, Softmax, Multiply, RepeatVector
from tensorflow.keras.regularizers import l2, l1


def load_data(normal=True, data_path='/home/hoheon/Jupyter/2020_01/DD_classification/cache/'):
    
    
    
    if normal:
        main_result_normal_1_2 = pd.read_csv(data_path+'main_input_normal_0216_1_2.csv')
        sub_result_normal_1_2 = pd.read_csv(data_path+'sub_input_normal_0216_1_2.csv')

        main_result_normal_1_3 = pd.read_csv(data_path+'main_input_normal_0216_1_3.csv')
        sub_result_normal_1_3 = pd.read_csv(data_path+'sub_input_normal_0216_1_3.csv')

        main_result_normal_1_6 = pd.read_csv(data_path+'main_input_normal_0216_1_6.csv')
        sub_result_normal_1_6 = pd.read_csv(data_path+'sub_input_normal_0216_1_6.csv')

        main_result_normal_1_7 = pd.read_csv(data_path+'main_input_normal_0216_1_7.csv')
        sub_result_normal_1_7 = pd.read_csv(data_path+'sub_input_normal_0216_1_7.csv')
        
        returns = [main_result_normal_1_2, sub_result_normal_1_2, 
                   main_result_normal_1_3, sub_result_normal_1_3,
                   main_result_normal_1_6, sub_result_normal_1_6,
                   main_result_normal_1_7, sub_result_normal_1_7]
        
        return returns
    else:
        main_result_abnormal_1_2 = pd.read_csv(data_path+'main_input_abnormal_0216_1_2.csv')
        sub_result_abnormal_1_2 = pd.read_csv(data_path+'sub_input_abnormal_0216_1_2.csv')

        main_result_abnormal_1_3 = pd.read_csv(data_path+'main_input_abnormal_0216_1_3.csv')
        sub_result_abnormal_1_3 = pd.read_csv(data_path+'sub_input_abnormal_0216_1_3.csv')

        main_result_abnormal_1_6 = pd.read_csv(data_path+'main_input_abnormal_0216_1_6.csv')
        sub_result_abnormal_1_6 = pd.read_csv(data_path+'sub_input_abnormal_0216_1_6.csv')

        main_result_abnormal_1_7 = pd.read_csv(data_path+'main_input_abnormal_0216_1_7.csv')
        sub_result_abnormal_1_7 = pd.read_csv(data_path+'sub_input_abnormal_0216_1_7.csv')
        
        returns = [main_result_abnormal_1_2, sub_result_abnormal_1_2, 
                   main_result_abnormal_1_3, sub_result_abnormal_1_3,
                   main_result_abnormal_1_6, sub_result_abnormal_1_6,
                   main_result_abnormal_1_7, sub_result_abnormal_1_7]
        
        return returns
    




def build_CNN_model(time_resampler = 100):
     #Time variant input layer
    main_input = Input(shape=(time_resampler,12), dtype = 'float32', name='main')
    encoded_main = Conv1D(32, kernel_size=3, strides=1, padding='same', activation='selu', 
                             activity_regularizer=l1(0.00000001)
                                )(main_input)
   # pooled_main = layers.Dropout(0.4)(encoded_main)
    pooled_main = MaxPooling1D(pool_size=2)(encoded_main)
    
    encoded_main = Conv1D(64, kernel_size=3, strides=1, padding='same', activation='selu', 
                             activity_regularizer=l1(0.0000001)
                                )(pooled_main)
    #pooled_main = layers.Dropout(0.4)(encoded_main)
    pooled_main = MaxPooling1D(pool_size=2)(encoded_main)
    
    encoded_main = Conv1D(32,kernel_size=3, strides=1, padding='same', activation='selu', 
                             activity_regularizer=l1(0.0000001)#, l2 = 0.00000000001 )
                                )(pooled_main)
    pooled_main = Dropout(0.4)(encoded_main)
    pooled_main = MaxPooling1D(pool_size=2)(pooled_main)
    
    encoded_main = Conv1D(16,kernel_size=2, strides=1, padding='same', activation='selu', 
                           activity_regularizer=l1(0.0000001)
                                )(pooled_main)
  #  pooled_main = layers.Dropout(0.4)(encoded_main)
    pooled_main = MaxPooling1D(pool_size=2)(encoded_main)
    
    encoded_main = Conv1D(8,kernel_size=2, strides=1, padding='same', activation='selu', 
                            activity_regularizer=l1(0.0000001) 
                                )(pooled_main)
    pooled_main = Dropout(0.4)(encoded_main)
    pooled_main = MaxPooling1D(pool_size=2)(pooled_main)
    
    encoded_main = Conv1D(4,kernel_size=2, strides=1, padding='same', activation='selu', 
                           activity_regularizer=l1(0.0000001), name='test'
                                )(pooled_main)


    pooled_main = GlobalAveragePooling1D()(encoded_main)

    #Time Fixed input layer
    sub_input = Input(shape=(25,), dtype = 'float32', name='sub')
    #embedded_sub = layers.Reshape((-1,1))(sub_input)
    embedded_sub = RepeatVector(150)(sub_input)
    encoded_sub = Conv1D(32,kernel_size=3,strides=1,padding='same', activation='selu',
                                                  activity_regularizer=l1(0.0000001)
                               )(embedded_sub)
   # pooled_sub = layers.Dropout(0.4)(encoded_sub)
    pooled_sub = MaxPooling1D(pool_size=2)(encoded_sub)
    encoded_sub = Conv1D(64,kernel_size=3,strides=1,padding='same', activation='selu',
                                                  activity_regularizer=l1(0.0000001)
                               )(pooled_sub)
    #pooled_sub = layers.Dropout(0.4)(encoded_sub)
    pooled_sub = MaxPooling1D(pool_size=2)(encoded_sub)
    encoded_sub = Conv1D(32,kernel_size=3, strides=1, padding='same', activation='selu', 
                             activity_regularizer=l1(0.0000001)
                               )(pooled_sub)
    pooled_sub = Dropout(0.4)(encoded_sub)
    pooled_sub = MaxPooling1D(pool_size=2)(pooled_sub)
    encoded_sub = Conv1D(16,kernel_size=2, strides=1, padding='same', activation='selu', 
                             activity_regularizer=l1(0.0000001)
                               )(pooled_sub)
   #pooled_sub = layers.Dropout(0.4)(encoded_sub)
    pooled_sub = MaxPooling1D(pool_size=2)(encoded_sub)
    encoded_sub = Conv1D(8,kernel_size=2, strides=1, padding='same', activation='selu', 
                              activity_regularizer=l1(0.0000001)
                                )(pooled_sub)
    pooled_sub = Dropout(0.4)(encoded_sub)
    pooled_sub = MaxPooling1D(pool_size=2)(pooled_sub)
    encoded_sub = Conv1D(4,kernel_size=2, strides=1, padding='same', activation='selu', 
                             activity_regularizer=l1(0.0000001)
                               )(pooled_sub)
    pooled_sub = GlobalAveragePooling1D()(encoded_sub)
    
    concatenated = concatenate([pooled_main, pooled_sub],axis=-1)
    answer = Dense(2, activation='softmax')(concatenated)
    model = Model([ main_input,sub_input], answer)
    
    return model


def build_HH_model(n_time_var,
                   n_fix_var,
                   dot_frequency=300,
                   kernel_regularizer=0.25):
    '''
    Parameters
    ----------
    dot_frequency: int
        (default: 300)
    
    kernel_regularizer: float
    
    Returns
    -------
    keras.Model
    '''
    # Time variant input layer
    input_x = Input(shape=(dot_frequency, n_time_var), dtype = 'float32')
    
    encoded_main = Conv1D(8, kernel_size=1, strides=1, activation='selu', kernel_regularizer=l2(kernel_regularizer))(input_x)
    pooled_main = MaxPooling1D(pool_size=5, )(encoded_main)
    encoded_main = Conv1D(3, kernel_size=1, strides=1,activation='selu', kernel_regularizer=l2(kernel_regularizer))(pooled_main)
    pooled_main = MaxPooling1D(pool_size=5,)(encoded_main)
    encoded_main = Conv1D(2, kernel_size=1, strides=1, activation='selu', kernel_regularizer=l2(kernel_regularizer))(pooled_main)
    flatten = Flatten()(encoded_main)

    # Time Fixed input layer
    input_aux = Input(shape=(n_fix_var,), dtype = 'float32')
    
    concatenated = concatenate([flatten, input_aux], axis=-1)
    answer = Dense(1, activation='relu', kernel_regularizer=l2(kernel_regularizer))(concatenated)
    
    model = Model([input_x, input_aux], answer)
    
    return model





class DragGame(object):
    def __init__(self, user_id, display_data):
        self.user_id = user_id
        self.display_size = np.array(display_data.loc[user_id][['screenWidth', 'screenHeight']], dtype='float32')
        

    def coordiate_norm(self, coordinate):
        '''사용자의 한 시점에 대한 X좌표 Y좌표를 기입하였을때, 사용자의 디바이스를 고려하여 Ratio비율을 구합니다.

        Parameters
        ----------
        coordiates : array(X, Y)
            * x : float16
            * y : float16

        Return
        ------
        tuple. (X, Y) 
            * X <= 1
            * Y <= 1


        Example
        -------

        >>> coordinate = pd.DataFrame[['PosX', 'PosY']].iloc[0]
        >>> disp_size = np.array(pd.DataFrame[['screen_width', 'screen_height']])

        >>> coordiate_norm(coordinate, 'dhs72yj-23huda7-23bjd7a-23bd6', disp_size)
        (0.3, 0,2)
        '''

        assert type(coordinate) is np.ndarray, print('coordinate type must be np.array')
        
        # dtype
        coordinate = coordinate.astype('float32')
        div = 1 / self.display_size

        return coordinate * div
        
        
    def set_margin_ratio(self, margin_rate):
        '''Optimal path에 디스플레이 사이즈의 얼마만큼 고려해서 흔들림을 마진으로 잡을 것인지 '''
        self.margin_rate = margin_rate
        
    
    def _get_pathway(self, game_num):
        '''Game number에 따른 start point, endpoint구하기
        
        Parameters
        ----------
        game_num: int
        
        Return
        --------
        tuple: np.array (start point),  np.array (start point)
        '''
        if game_num == 2:
            start_point = np.array([1600, 625])
            end_point = (np.array([1285, 439]) + np.array([970, 122]))/2
        elif game_num == 3:
            start_point = np.array([1545, 280])
            end_point = np.array([912, 328])
        elif game_num == 6:
            start_point = np.array([1680, 425])
            end_point = np.array([1350, 555])
        return start_point, end_point
    
    
    def optimal_path(self, game_num):

        ''' 각 게임의 Optimal pathway 구하기
        해당 Optimal path은 한 오브젝트에서 다른 오브젝트의 선형회귀 선과 같다.
        따라서, 시작포인트(startpoint)와 끝점(Endpoint)을 선형회귀하여 해당좌표의 linear regression 객체를 반환한다.


        Parameters
        ----------
        game: int


        Return
        ------
        numpy.poly1d:  표준화 회귀선형식
        
        '''
        
        # Start & End point
        startpoint, endpoint = self._get_pathway(game_num)
        startpoint_norm = self.coordiate_norm(startpoint)
        endpoint_norm = self.coordiate_norm(endpoint)

        # Get linear model
        dots = np.vstack([startpoint_norm, endpoint_norm])

        xs = dots[:, 0]
        ys = dots[:, 1]

        fp1 = np.polyfit(xs, ys, 1)
        f1 = np.poly1d(fp1)   

        return f1

    
    def parellel_translation(self, opt_path, direction='up'):
        '''
        optimal_path을 Y축으로 X만큼 평행이동 함
        설계좀 부탁드립니다.
        Parameters
        ----------
        path: optimal_path 함수의 결과물.
            * numpy.poly1d

        direction: margin의 방향

        margin_rate: margin의 비율
        
        Example
        -------
        >>> opt_path = drag_game.optimal_path(2)
        >>> parellel_translation(opt_path)
        '''

        
        if direction == 'up':
            # Upper marginal line
            opt_path = opt_path + self.margin_rate
        else:
            # Lower marginal line
            opt_path = opt_path - self.margin_rate

        return opt_path
    
    
    def check_margin(self, coordinate, game_num=2):
        '''
        좌표가 Upper marginal line과 lower marginal line에 사이에 있는지 검토

        Parameters
        ----------
        coordinate: np.array


        Return
        ------
        bool: 
            *True: Outside beyond margina
            *False: Inside margin


        Example
        -------
        from DoBrain.DragDataModel.DragAndDrop import DragGame

        # input
        id_ = '484b896a9065768ab6b64e291227f0df'

        drag_game = DragGame(user_id=id_, display_data=displays)
        drag_game.set_margin_ratio(0.2)

        # Uppler lower margin
        poly = drag_game.optimal_path(2)
        upper_margin = drag_game.parellel_translation(poly, 'up')
        lower_margin = drag_game.parellel_translation(poly, 'lower')

        print(poly)
        print(upper_margin)
        print(lower_margin)


        dots = np.array(main_result_1_2.loc[main_result_1_2.ID == id_][['PosX','PosY']], dtype='float32')
        norm_dot = drag_game.coordiate_norm(dot)
        dots_norm = np.apply_along_axis(func1d=drag_game.coordiate_norm,
                                        axis=1,
                                        arr=dots)

        opt_path = [poly(i) for i in np.linspace(0, 1, 50)]
        y_up = [upper_margin(i) for i in np.linspace(0, 1, 50)]
        y_lo = [lower_margin(i) for i in np.linspace(0, 1, 50)]
        
        # Plot
        plt.plot(np.linspace(0, 1, 50), y_up)
        plt.plot(np.linspace(0, 1, 50), y_lo)
        plt.plot(np.linspace(0, 1, 50), opt_path)
        for dot in dots_norm:
            plt.plot(dot[0], dot[1], 'bo')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        '''
        coordinate= coordinate.astype('float32')
        
        
        opt_path = self.optimal_path(game_num=2)
        
        upper_margin = self.parellel_translation(opt_path, 'up')
        lower_margin = self.parellel_translation(opt_path, 'lower')
        
        
        upper_value = upper_margin(coordinate[0])
        lower_value = lower_margin(coordinate[0])

        if upper_value > coordinate[1] and lower_value < coordinate[1]:
            return True
        else:
            return False    