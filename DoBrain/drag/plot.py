#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Written by Ho Heon Kim. hoheon0509@gmail.com
# Coworker: Jae il Ahn 

import numpy as np
import pandas as pd
import seaborn as sns

def plot_drag(id_, DataFrame, display_height, display_width, hue=True):
    ''' Plot user's drag trace
    
    Parameters
    ----------
    id: str. user id
    DataFrame: pd.DataFrame including drag and drop data
        * not including index
        * columns
            PosX: X coordinate 
            PosY: Y coordinate
            
    Return 
    ------
    None
    
    
    Example
    -------
    from DoBrain.DragDataModel.utils import display_info
    
    id_ = '484b896a9065768ab6b64e291227f0df'
    
    height, width = display_info(id_, displays)
    
    plot_drag(id_, main_result_1_2, height, width)
    
    print('display size', width, height)
    '''
    
    # Parameters valification
    assert type(id_) is str, print('id_ is not string')
    assert DataFrame.index.name is None, print('The DataFrame must not be indexed')
    
    user_trace = DataFrame.loc[DataFrame.ID == id_].reset_index(drop=True)
    
    pos_x = np.array(user_trace['PosX'])
    pos_y = np.array(user_trace['PosY'])
    
    if hue:
        plot = sns.scatterplot(pos_x, pos_y, hue=user_trace.index)
    
    ax = plot.axes
    ax.set_xlim(0, display_width)
    ax.set_ylim(0, display_height)
        
    
    
    