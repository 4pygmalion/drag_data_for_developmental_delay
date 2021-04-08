#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Written by Jae il Ahn.
# Co-worker: Ho Heon Kim.  hoheon0509@gmail.com 

import numpy as np



import numpy as np
import pandas as pd
import seaborn as sns

def blancing_training(x_train, y_train):
    '''
    
    Return
    -----
    Oversampled x_train, y_train
    '''
    
    from random import sample
    
    Y0 = np.where(y_train == 0)[0]
    Y1 = np.where(y_train == 1)[0]

    if len(Y0) > len(Y1):
        multiple = round(len(Y0) / len(Y1))
        k = len(Y0) - (len(Y1) * multiple)
        
        remain = sample(list(Y1), k)
        Y1 = list(Y1) * (multiple - 1) + remain
        add_x = x_train[Y1]
        add_y = y_train[Y1] 
        return np.vstack([x_train, add_x]), np.hstack([y_train, add_y])
        



def display_info(id_, display_data):
    '''
    Parameters
    ----------
    ids
    
    display_data: pd.DataFrame
        * not index : but, name of index columns must be '
    
    Returns
    -------
    display info: tuple. (Height, Width)
    
    
    Example
    -------
    height, width = display_info(id_, displays)
    
    '''
    
    assert display_data.index.name is not None, print('Display dataframe must be index')
    
    height, width = display_data.loc[id_][['screenHeight', 'screenWidth']]
    
    return height, width


def infer_display(id_, gamedata, game_num):
    '''
    Parameters
    ----------
    ids_: 
    gamedata
    game_num
    
    
    Return
    ------
    display info: tuple. (Height, Width)
    
    Example
    -------
    
    '''
    # Get last dot
    user_data = gamedata.loc[gamedata.ID == id_]
    x, y = user_data.iloc[-1][['PosX', 'PosY']]
    
    # Target object
    
    # Target path area
    if game_num == 2:
        x_ratio = (1285+970)/2/2436
        y_ratio = (439+122)/2/1125
    elif game_num == 3:
        x_ratio = (640+1179)/2/2436
        y_ratio = (423+441)/2/1125
        
    x_infer = x * (1/x_ratio)
    y_infer = y * (1/y_ratio)
        
    return (x_infer, y_infer)
    
    