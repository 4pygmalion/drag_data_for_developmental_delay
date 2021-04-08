# UTF-8


# Packages
import pymysql
import logging
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

class DragVariablesGenerator(object):
    '''
    Parameters
    ----------
    data: single game log

    '''


    def __init__(self, data):
        self.data = data
    
    
    def to_coordinates_ratio(self, width_col='ScreenWidth', height_col='ScreenHeight', x='posY', y='posY'):
        '''
        Parameters
        ----------
        
        Return
        ------
        (x, y): pd.DataFrame
        '''
        
        x = self.data[x] / self.data[width_col]
        y = self.data[y] / self.data[height_col]
        
        norm_coordinates = pd.concat([x,y], axis=1)
        norm_coordinates.columns = ['rposX', 'rposY']
        
        return norm_coordinates
    

    def to_timestamp(self, colname='CreationDateTime'):
        '''
        Converting: YYYY-MM-DD HH.MM.SS -> YYYY-MM-DD HH:MM:SS.mmmmmm

        Parameter
        ----------
        col : str.


        Return
        ----------
        date: None


        '''

        def _to_datetime(date):
           
            datetime = date.split()[0]
            
            try:
                timestamp = date.split()[1]
            except:
                # to handle CreationDatetime_M is '0'
                return np.nan

            # '08-19-59' -> '08:19:59'
            hhmmss = timestamp[:8].replace('.',':')
            hhmmss = timestamp[:8].replace('-',':')

            # Miliseconds (ms)
            milisec = timestamp[8:].replace(':', '.')
            milisec = timestamp[8:].replace('-', '.')

            # merge
            date = datetime + ' ' + hhmmss + milisec

            if len(date) == 19:
                date = date + '.000'
            elif len(date) < 23:
                date = date + '0'*(23-len(date))
            elif len(date) == 23:
                date = date[:19] + '.' + date[20:]


            # 0000-00-00 00:00:00:000 -> 0000-00-00 00:00:00:00
            # date = date[:-1]
            return date
        
        # dates = self.data[colname].apply(lambda x: pd.to_datetime(x))
        dates = self.data[colname].apply(lambda x: pd.to_datetime(_to_datetime(x)))
        self.data.loc[:, colname] = dates

        
    def get_timediff(self, colname='CreationDateTime'):
        '''
        Return
        ------
        None
        '''
        time_diff = self.data[colname] - self.data[colname].shift()
        time_diff = time_diff.fillna('0 days 00:00:00.000000')
        self.data.loc[:, 'timediff'] = time_diff
        self.data.loc[:, 'timediff'] = self.data['timediff'].apply(lambda x: pd.to_timedelta(x).total_seconds())



    def cnt_fly_time(self,):
        '''
        게임 플레이 중간, 기기와 손이 떨어지는 Fly time을 계산

        Parameter
        ----------
        DataFrame : pandas.DataFrame.
            아동 플레이 데이터
        level_ : str.
            분석에 사용할 레벨

        Return
        ----------
        pandas.DataFrame.
        '''


    def get_sin(self, xcol='posX', ycol='posY'):
        '''
        게임을 하면서 생기는 좌표의 sin과 cos을 계산하여 변수로 생성

        Parameter
        ----------
        None

        Return
        ----------
        list of cosine in each vector
        '''

        def _sin(x, y):
            return y / np.sqrt(x**2 + y**2)

        return list(self.data.apply(lambda row: _sin(row[xcol], row[ycol]), axis=1))


    def get_cos(self, xcol='posX', ycol='posY'):
        '''
        게임을 하면서 생기는 좌표의 cos을 계산하여 변수로 생성

        Parameter
        ----------
        None

        Return
        ----------
        list of cosine in each vector
        '''

        def _cox(x, y):
            return x / np.sqrt(x**2 + y**2)

        return list(self.data.apply(lambda row: _cox(row[xcol], row[ycol]), axis=1))

    
    def cal_velocity(self, timediff='timediff', xcol='posX', ycol='posY'):
        ''' 게임을 하면서 생기는 좌표의 순간속도를 계산하여 변수로 생성

        :param timecol: str. name of column
        :param xcol: str. name of column
        :param ycol: str. name of column

        return
        ------
        list

        Example
        self._to
        '''

        # to float
        self.data.loc[:, xcol] = self.data[xcol].astype('float32')
        self.data.loc[:, ycol] = self.data[ycol].astype('float32')

        # length
        x_delta = self.data[xcol].shift() - self.data[xcol]
        y_delta = self.data[ycol].shift() - self.data[ycol]

        x_square = x_delta.apply(lambda x: x**2).fillna(0)  # x**2
        y_square = y_delta.apply(lambda x: x**2).fillna(0)  # y**2

        xy = x_square + y_square
        length = xy.apply(lambda x: np.sqrt(x))

        # Velocity
        velocity = length / self.data[timediff]
        velocity = velocity.astype('float16')
        velocity = velocity.fillna(0)  #
        velocity = velocity.replace([np.inf, -np.inf], 0)

        return velocity


    def cal_velocity_along_axis(self, axis='X', timediff='timediff', xcol='posX', ycol='posY'):
        ''' Velocity along the x axis
        
        Parameters
        ---------
            timecol: str. name of column
            xcol: str. name of column
            ycol: str. name of column

        Return
        ------
        list

        '''
        
        if axis not in ['X', 'Y']:
            raise ValueError('axis not equal "X", or "Y"')
        

        # to float
        self.data.loc[:, xcol] = self.data[xcol].astype('float32')  # xcol
        self.data.loc[:, ycol] = self.data[ycol].astype('float32')
        
        # length
        if axis == 'X':
            length = self.data[xcol].shift() - self.data[xcol]  # length      
        else:
            length = self.data[ycol].shift() - self.data[ycol]
        
        
        # Velocity
        v = length / self.data[timediff]
        v = v.astype('float16')
        v = v.fillna(0)  #
        v = v.replace([np.inf, -np.inf], 0)

        return v.astype('float16')

    
    def cal_acceleration(self, timediff='timediff', xcol='posX', ycol='posY'):
        '''
        calculating acceleration
        
        parameters
        ----------
        timediff: str
        xcol: str. posX
        ycol: str. posY 
        
        '''
                
        # Acceleration
        velocity =  self.cal_velocity(timediff, xcol, ycol)
        a = velocity / self.data[timediff]
    
        # fill na / inf / -inf
        a = a.astype('float16')
        a = a.fillna(0)
        a = a.replace([np.inf, -np.inf], 0)
        return a
        
        
    def cal_acceleration_along_axis(self, axis='X', timediff='timediff', xcol='posX', ycol='posY'):
        '''
        calculating acceleration along the 'x' or 'y' axis
        
        parameters
        ----------
            axis = str.
                'X':
                'Y':

            timediff: str.
            xcol: str. posX
            ycol: str. posY 
        
        Return
        ------
        
        '''
        
        
        
        # velocity
        
        v_along_axis = self.cal_velocity_along_axis(axis)
        a_along_axis = v_along_axis / self.data[timediff]

        # fill na / inf / -inf
        a = a_along_axis.astype('float16')
        a = a.fillna(0)
        a = a.replace([np.inf, -np.inf], 0)
        return a
    
    
    # Time fixed variables

    def get_max_coordinates(self, axis='X'):
        if axis == 'X':
            max_ = max(list(self.data['posX']))
        else:
            max_ = max(list(self.data['posY']))
        return max_

    def get_mean_coordinate(self, axis='X'):
        if axis == 'X':
            mean_ = np.mean(list(self.data['posX']))
        else:
            mean_ = np.mean(list(self.data['posY']))
        return mean_

    def cnt_n_line(self, colname='GameGrivenIndex'):
        '''
        게임을 플레이 할 때, 사용한 라인의 개수를 count

        Parameter
        ---------
        colname: name of stroke index

        Return
        --------
        int
        '''
        n_lines = max(self.data[colname])
        return n_lines


    def sign_change_velocity(self, axis='X'):
        velocity = self.cal_velocity_along_axis(axis=axis, timediff='timediff', xcol='posX', ycol='posY')
        sign_vel = np.sign(velocity)
        sign_change = ((sign_vel - np.roll(sign_vel, 1)) != 0).astype('int')
        return sign_change.sum()

    def sign_change_acceleration(self, axis='X'):
        acc = self.cal_acceleration_along_axis(axis='X', timediff='timediff', xcol='posX', ycol='posY')
        sign_acc = np.sign(acc)
        sign_change = ((sign_acc - np.roll(sign_acc, 1)) != 0).astype('int')
        return sign_change.sum()
