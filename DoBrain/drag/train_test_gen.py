# UTF8

import numpy as np
import pandas as pd
import collections
from tslearn.preprocessing import TimeSeriesResampler


def data_preprocess(time_vari_DL_input_ab, time_fixed_DL_input_ab, time_vari_DL_input_nor, time_fixed_DL_input_nor):
    '''
    각 DataFrame에서 필요한 columns 선택 후 정규화 진행.
    정상발달이랑 비정상아동 데이터를 한 데이터프레임에 저장
    
    Parameters
    -------------
    time_vari_DL_input_ab : pd.DataFrame
        발달 장애 아동들의 time-vari input
    time_fixed_DL_input_ab : pd.DataFrame
        발달 장애 아동들의 time-fixed input
    time_vari_DL_input_nor : pd.DataFrame
        정상 발달 아동들의 time-vari input
    time_fixed_DL_input_nor : pd.DataFrame
        정상 발달 아동들의 time-fixed input
        
    Return
    -------------
    time_vari_DL_input : pd.DataFrame
    time_fixed_DL_input : pd.DataFrame
    '''
    vari_col = ['ID', 'PosX', 'PosY','sin','cos', 'velocity', 'velocity_X',
       'velocity_Y', 'accelerate', 'accelerate_X', 'accelerate_Y', 'Fly_time','Play_time', 'State']
    
    fixed_col = ['ID', 'Line_Count', 'Flytime_cal', 'Height_Max','Height_Mean', 'Height_Std', 'Width_Max', 'Width_Mean', 'Width_Std',
       'Play_time_cal', 'Line_Length', 'vx_cnt','vy_cnt', 'a_cnt', 'ax_cnt','ay_cnt','vx_per_line', 'vy_per_line', 
        'a_per_line', 'ax_per_line', 'ay_per_line','vx_per_time', 'vy_per_time', 'a_per_time', 'ax_per_time', 'ay_per_time']
    

    time_vari_DL_input_ab = time_vari_DL_input_ab.loc[:, vari_col]  
    time_fixed_DL_input_ab = time_fixed_DL_input_ab.loc[:, fixed_col]
    
    time_vari_DL_input_nor = time_vari_DL_input_nor.loc[:, vari_col]
    time_fixed_DL_input_nor = time_fixed_DL_input_nor.loc[:, fixed_col]
    
    time_vari_DL_input = pd.concat([time_vari_DL_input_nor, time_vari_DL_input_ab])
    time_fixed_DL_input = pd.concat([time_fixed_DL_input_nor, time_fixed_DL_input_ab])
    
    vari_col.remove('ID')
    vari_col.remove('State')
    fixed_col.remove('ID')

    for _ in vari_col:
        x = time_vari_DL_input[_]
        normalized_df=(x-x.min())/(x.max()-x.min())
        time_vari_DL_input[_] =  list(normalized_df)
    
    for _ in fixed_col:
        x = time_fixed_DL_input[_]
        normalized_df=(x-x.min())/(x.max()-x.min())
        time_fixed_DL_input[_] =  list(normalized_df)
    time_vari_DL_input = pd.get_dummies(time_vari_DL_input, columns=['State'])
        
    return time_vari_DL_input, time_fixed_DL_input


def time_vari_gen(time_vari_DL_input):
    '''ID 기준으로 학습이 필요한 열을 ...............
    
    Parameter
    ----------
    time_vari_DL_input : pd.DataFrame
        data_process의 첫번째 반환 데이터
        
    Return
    ----------
    mapPositionDataByID : dict.
    mapStateByID : dict
    '''

    vari_col = ['PosX', 'PosY','sin','cos', 'velocity', 'velocity_X', 'velocity_Y', 'accelerate', 'accelerate_X',
                'accelerate_Y', 'Fly_time','Play_time']
    vari_label_col = ['State_Abnormal','State_Normal']
    mapPositionDataByID = collections.OrderedDict()
    mapStateByID = collections.OrderedDict()
    data_dic=dict(tuple(time_vari_DL_input.groupby('ID')))

    
    for k in data_dic.keys():
        mapPositionDataByID[k]=data_dic[k].loc[:,vari_col].values
        mapStateByID[k]=data_dic[k].loc[:,vari_label_col].values
    return mapPositionDataByID, mapStateByID


def train_test_split_JJ(time_vari_DL_input, time_fixed_DL_input, train_id, val_id, resampler=100 ):
    '''
    train_id와 vali_id에 맞게 train, test data split
    
    Parameter
    -----------
    time_vari_DL_input : pd.DataFrame
    time_fixed_DL_input : pd.DataFrame
    train_id : list
        학습에 사용할 아동들의 id
    test_id : list
        검증에 사용할 아동들의 id
    resampler : int.
        time-vari data 재구성에 사용할 데이터 숫자
        
    
    '''
    mapPositionDataByID, mapStateByID = time_vari_gen(time_vari_DL_input)
  
    x_train, y_train, x_test, y_test, x_train_sub, x_test_sub  = [], [], [] ,[],[],[]
   
    fixed_col = ['Line_Count', 'Flytime_cal', 'Height_Max','Height_Mean', 'Height_Std', 'Width_Max',
                 'Width_Mean', 'Width_Std', 'Play_time_cal', 'Line_Length', 'vx_cnt','vy_cnt', 'a_cnt', 
                 'ax_cnt','ay_cnt', 'vx_per_line', 'vy_per_line', 'a_per_line', 'ax_per_line',
                 'ay_per_line','vx_per_time', 'vy_per_time', 'a_per_time', 'ax_per_time', 'ay_per_time']
    
    for key in train_id:
        x_train.append(mapPositionDataByID[key])
        y_train.append(mapStateByID[key][0])
        x_train_sub.append(np.array(time_fixed_DL_input.set_index('ID').loc[key,fixed_col].values))
        
    for key in val_id:
        x_test.append(mapPositionDataByID[key])
        y_test.append(mapStateByID[key][0])
        x_test_sub.append(np.array(time_fixed_DL_input.set_index('ID').loc[key,fixed_col].values))


    x_train_input = TimeSeriesResampler(sz=resampler).fit_transform(x_train)
    x_test_input = TimeSeriesResampler(sz=resampler).fit_transform(x_test)
    y_train_input = np.array(y_train)
    y_test_input = np.array(y_test)
    x_train_sub_input = np.array(x_train_sub)
    x_test_sub_input = np.array(x_test_sub)
    
    
    # append -> stack
    
    x_train_sub_input = np.vstack(x_train_sub_input)
    x_test_sub_input = np.vstack(x_test_sub_input)
    
    return x_train_input, x_train_sub_input, y_train_input, x_test_input, x_test_sub_input, y_test_input
        