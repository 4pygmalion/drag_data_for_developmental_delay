# UTF-8

# Packages
import numpy as np
import pandas as pd

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

class DragToImage(object):
    '''
    Parameters
    ----------
    data: pd.DataFrame. (single game log)
    resolution: list
        default = [height, width]
    
    '''


    def __init__(self, dataframe, resolution=[768, 1204]):
        self.data = dataframe
        self.resolution = resolution
               
        
    def get_display_info(self, w='ScreenWidth', h='ScreenHeight'):
        
        # if display info exist
        w = int(self.data[w][0])
        h = int(self.data[h][0])
        
        # display info
        self.h = h
        self.w = w
        
        # ratio 
        self.w_ratio = self.resolution[1] / w
        self.h_ratio = self.resolution[0] / h
        

    def _get_display_info_error(self, w='ScreenWidth', h='ScreenHeight', x_col='posX', y_col='posY'):

        '''
        whether posX, posY is beyond display size

        Return
        ------
        bool
        '''

        self.get_display_info()

        max_posX = max(self.data[x_col])
        max_posY = max(self.data[y_col])
        cond1 = self.w < max_posX
        cond2 = self.h < max_posY

        if cond1 or cond2:
            return True
        else:
            return False

        
        
    def make_channel(self, feature, x_col='posX', y_col='posY', pad=0):
        '''
        Parameters
        ----------
        feature: str. the column name. feature channel
        
        x_col: str
            the column name. original coordinate
            
             
        Return
        ------
        3D shape np.array
       
        '''
        
        # to HD resolution

        if self._get_display_info_error():   # posX > display width
            max_posX = max(self.data[x_col])
            max_posY = max(self.data[y_col])
            self.w = max_posX
            self.h = max_posY
            self.w_ratio = self.resolution[0] / self.w
            self.h_ratio = self.resolution[1] / self.h

        r_X = self.data[x_col] * self.w_ratio
        r_Y = self.data[y_col] * self.h_ratio
        xy = list(zip(r_X, r_Y))
        xy = np.array(xy).astype('int16')



        # feature channel
        empty_img = np.zeros(shape=self.resolution, dtype=np.float16)  # 336Mb -> 88 Mb
        
        for xy, feature in zip(xy, self.data[feature]):
            x, y = xy[0]-1, xy[1]-1

            try:
                empty_img[y][x] = feature
            except:
                pass

        img = empty_img.T
        return np.rot90(img)
            
        
        
        
        
        
        