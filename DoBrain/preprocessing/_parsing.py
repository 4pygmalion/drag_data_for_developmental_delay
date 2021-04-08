#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Written by Ho Heon Kim.  hoheon0509@gmail.com 
# Co-worker: Sierra Lee. justforher12344@gmail.com

import sys
import re
import numpy as np
import pandas as pd
import json
import requests
import pymysql
import csv

from tqdm import tqdm
from datetime import date, timedelta

# UDF
sys.path.append('/home/hoheon/packages/')
from utils import flatten
from pandas import json_normalize


class Parser(object):
    '''To parse json file including drag and drop data of DoBrain new version

    Example
    -------

    >>> import sys
    >>> sys.path.append('/home/hoheon/Packages/')

    >>> from DoBrain.drag.parsing import DragData
    
    >>> json_path = './Data/AWS_sample/00fb528ea9666e95e367bdd5a6370498b950ef970c6a9c8c2ae0aff7c7056122.json'
    >>> db_parser = DragData(json_path)
    >>> db_parser.parsing()

    '''
    
    def __init__(self, json):
        assert type(json) is str, print('json path must be str.')
        
        self.path = json
        self.data = self._load_json()
        
        
    def _load_json(self):
        with open(self.path) as jsonfile:
            data = json.load(jsonfile)
        return data
    
    
    def _answerPlayLogs(self, subnode, node_name):
        '''Sub-node parsing (answerPlayLogs parsing)
        
        Parameters
        ----------
        subnode: dict
        
        node_name: str
         
        Return
        ------
        pd.DataFrame. data in structured form
        
        '''
        
        # strokePlayLogs parsing
        if node_name == 'strokePlayLogs':
            
            strokePlayLog = pd.DataFrame.from_dict(subnode)
            return strokePlayLog
        
        elif node_name == 'strokeValuePlayLogs':
            strokeValuePlayLogs = pd.DataFrame()
            for i in range(len(subnode)):  # d['strokePlayLogs'] : subnode
                a = pd.DataFrame.from_dict(subnode[i]['strokeValuePlayLogs'])
                if i == 0:
                    strokeValuePlayLogs = a
                    
                else :
                    strokeValuePlayLogs = pd.concat([strokeValuePlayLogs,a])

            return strokeValuePlayLogs
        
        
    def _arrangeId(self, data, node_name):
        '''Rename idenfier for merging in level0, strokePlayLogs, answerPlayLogs
        
        Parameters
        ----------
        data: pd.DataFrame
        
        node_name : str
        
        
        Return
        ------
        pd.DataFrame
        
        
        See Also
        --------
        Column Description
        
        
        id for merging
        --------------
        * id : identifier of specific logs
        * accountId, profileId : identifier of specific child
        * strokePlayLogId : identifier of specific strokeplaylog
        
        
        CreationTime
        ------------
        * creationUtcDateTimeGame : AnswerLog CreationTime
        * creationUtcDateTimeTouch : Touching CreationTime
        
        
        '''
         
        # strokePlaylogs id_matching
        if node_name == 'strokePlayLogs':   
            data = data.rename({'id' : 'strokePlayLogId'}, axis = 1)
            # derivedQuestionPlayLogId 가 Gyro 에도 FK로 쓰이기 떄문에 이름을 변경할 필요가 있음
            strokePlayLogs = data.rename({'derivedQuestionPlayLogId' : 'id'}, axis = 1)
            return strokePlayLogs
            
        elif node_name == 'strokeValuePlayLogs':
            data = data.drop('id', axis=1)
            # data = data.rename({'id': 'stroke_id'}, axis=1)    # to count number of stroke count
            strokeValuePlayLogs = data.rename({'creationUtcDateTime':'creationUtcDateTimeTouch'}, axis = 1)
            return strokeValuePlayLogs
        
        elif node_name == 'level0':
            del data['answerPlayLogs'], data['strokePlayLogs'], data['creationUtcDateTime']
            level0 = pd.DataFrame.from_dict([data])
            return level0
            
        elif node_name == 'answerPlayLogs' :
            data = data.drop('id', axis=1)
            data = data.rename({'derivedQuestionPlayLogId' : 'id'}, axis = 1)
            answerPlayLogs = data.rename({'creationUtcDateTime':'creationUtcDateTimeGame'}, axis=1)
            return answerPlayLogs
        

    
    
    def _multilevel_parsing(self):
        ''' Parsing subnode in simultaneously
        
        Parameters
        ----------
        None.
        
        
        Return
        ------
        Tuple (level0, answerPlayLogs, strokePlayLogs, strokeValuePlayLogs)
        
        '''
        
        # level 0
        level0 = self._load_json()
        cnt = 0
        for level in level0:
            cnt += 1
            
            # answerPlayLogs
            
            if len(level['answerPlayLogs']) != 0:
            
                answerplaylogs = pd.DataFrame.from_dict(level['answerPlayLogs'])
                answerplaylogs = self._arrangeId(answerplaylogs, 'answerPlayLogs')
            
            else : 
                answerplaylogs = pd.DataFrame()
 
                
 
            # strokePlayLogs
            if len(level['strokePlayLogs']) != 0:
                
                strokeplaylogs = level['strokePlayLogs']
                strokeplaylogs = self._answerPlayLogs(strokeplaylogs, 'strokePlayLogs')
                strokeplaylogs = self._arrangeId(strokeplaylogs, 'strokePlayLogs')
                
                # strokeValuePlayLogs
 
                if len(level['strokePlayLogs'][0]['strokeValuePlayLogs']) != 0:
 
                    strokeValueplaylogs = self._answerPlayLogs(level['strokePlayLogs'], 'strokeValuePlayLogs')
                    strokeValueplaylogs = self._arrangeId(strokeValueplaylogs, 'strokeValuePlayLogs')
                    strokeplaylogs = strokeplaylogs.drop(columns = ['strokeValuePlayLogs']).drop_duplicates()
 
                else :
                    strokeplaylogs = strokeplaylogs.drop(columns=['strokeValuePlayLogs']).drop_duplicates()
                    strokeValueplaylogs = pd.DataFrame()
 
            else :
                strokeplaylogs = pd.DataFrame()
                strokeValueplaylogs = pd.DataFrame()
 
            if cnt == 1:
                answerPlayLogs = answerplaylogs
                strokePlayLogs = strokeplaylogs
                strokeValuePlayLogs = strokeValueplaylogs
            else:
                answerPlayLogs = pd.concat([answerPlayLogs, answerplaylogs])
                strokePlayLogs = pd.concat([strokePlayLogs, strokeplaylogs])
                strokeValuePlayLogs = pd.concat([strokeValuePlayLogs, strokeValueplaylogs])
 
            # level_0_id arrange
            level = self._arrangeId(level, 'level0')
 
            if cnt == 1 :
                level0 = level
            else :
                level0 = pd.concat([level0, level])
 
 
        return level0, answerPlayLogs, strokePlayLogs, strokeValuePlayLogs
    

    
    
    
    def parsing(self, fullcols=True, by_day=True):
        '''Assgin the parsing result into self.parsing
        
        Parameters
        ----------
        fullcols: bool. Return full columns or not
        by_day: bool. 
        
        Example
        -------
        >>> drag_data.parsing(fullcolse=False)
        
        '''
             
        
        
        level0, answerPlayLogs, strokePlayLogs, strokeValuePlayLogs = self._multilevel_parsing()
        
        # strokePlay
        strokePlay = pd.merge(strokePlayLogs, strokeValuePlayLogs, on = 'strokePlayLogId', how = 'outer')
        strokePlay.sort_values(by='creationUtcDateTimeTouch', inplace = True)
        strokePlay = strokePlay.reset_index(drop=True)
 
        # level01
        level01 = pd.merge(level0, answerPlayLogs, on = 'id', how = 'outer')
        level01.sort_values(by = 'creationUtcDateTimeGame', inplace = True)
        level01 = level01.reset_index(drop=True)
        strokePlay = strokePlay.reset_index(drop=True)
        
        
        flatten_data = pd.merge(level01, strokePlay.iloc[:,1:], on = 'id', how = 'outer')
        
        essen_cols = []
        
        if fullcols:
            return flatten_data
        else:
            return flatten_data[essen_cols]
    
    
    def parsing_by_date(self):
        '''Assgin the parsing result into self.parsing for json file organized by dates 
        
        Note
        --------
        서브노드가 없는 경우도 있음.
        Drag data가 있는 경우만 Storke node들이 생성되기에 재개발해야함
        '''
        
        dfs = []
        n_users = len(self.data)
        return self.data[0]
        
        for idx in range(n_users):
            df = self.parsing(self.data[idx])
            
            dfs.append(df)
         
        return pd.concat(dfs)

    
    
    def plot(self):
        pass
     
    def to_gif(self, file_name):
        pass
    


class ParentEventParser(object): 
    
    '''
    To parse json file including parent event log data of DoBrain new version
    
    Example
    -------
    >>> import sys
    >>> sys.path.append('/home/hoheon/Packages/')
    >>> from DoBrain.drag.parsing import ParentEventParser
    
    >>> json_path = './Data/AWS_sample/00fb528ea9666e95e367bdd5a6370498b950ef970c6a9c8c2ae0aff7c7056122.json'
    >>> db_parser = ParentEventParser(json_path)
    >>> db_parser.parsing()
    
    
    # Written by Myung Han Yang.  suneverset@naver.com 
    # Co-worker: Ho Heon Kim.  hoheon0509@gmail.com
    # Co-worker: Sierra Lee. justforher12344@gmail.com
    '''
    
    def __init__(self, json_file):
        assert type(json_file) is str, print('json_file path must be str.')
        
        self.data = pd.read_json(json_file)
        
    def parsing(self):
        
        '''
        Parameters
        ----------
        None.
        
        
        Return
        ------
        DataFrame(including columns -> accountId, profileId, eventName, creationDateTime)
        
        '''
        
        self.data = self.data.rename(columns={'creationUtcDateTime':'creationDateTime'})
        self.data['creationDateTime'] = pd.to_datetime(self.data['creationDateTime'])
        
        parsing_result = self.data.sort_values(by=['accountId','profileId', 'creationDateTime'], ascending=[True, True, True])
                
        
        parsing_result = parsing_result.reset_index(drop=True)
        
        return parsing_result



class DragJsonParser(object):
    ''' Drag data parser for old dobrain.
    
    Drag Json available at
    1) https://drive.google.com/drive/u/0/folders/1mmgsbJbCITMOvN9UhAWVZBGj6guPYUq8
    2) https://drive.google.com/drive/folders/1GgvUB_EfDLcZmbygfVFFTpyAj7R4jvvs
    '''
    
    def _read_json(self, file):
        ''' read json

        return
        ------
        data: dict
        '''

        json_data = open(file).read()
        data = json.loads(json_data)
        return data
    
    def _parsing_by_os(self, data, os):
        '''
        os: 'iOS', 'Android'

        return
        ------
        dict: os level nodes
        '''
        return data[os]
    
    def _parsing_by_level(self, data, level):
        
        if level == 'device':
            try:
                device_col = ['_deviceModel','_deviceName','_dpi','_screenHeight','_screenWidth']
                device_info = json_normalize(data, meta=device_col, record_path='_userLevels')
                return device_info
            # some device columns are missing
            except:
                device_info = json_normalize(data, meta=list(data.keys())[:-1], record_path='_userLevels')
                return device_info
            
        elif level == 'question':
            contentIndexes, questions = [], []
            
            data = data['contentDragData'][0]
            
            data = json_normalize(data)
            n_data = len(data)
        
            for i in range(n_data):
                contentIndex, question = data[['_index', 'questionDragData']].iloc[i]
                question = json_normalize(question)
                question['contentIndex'] = contentIndex
                questions.append(question)

            questions = pd.concat(questions)
            questions = questions.rename(columns={'_index':'questionIndex'})
            questions = questions.reset_index(drop=True)
            return questions

        elif level == 'derivedQuestion':
            questionIndex, dervied_questions = [] ,[]
            for i in range(len(data)):
                dervied_question = data['derivedQuestionDragData'][i]
                dervied_question = json_normalize(dervied_question)
                dervied_questions.append(dervied_question)

                # index
                index_temp = data['questionIndex'].iloc[i]
                index_seq = [index_temp for _ in range(len(dervied_question))]
                questionIndex.append(index_seq)

            questionIndex = flatten(questionIndex)
            dervied_questions = pd.concat(dervied_questions)
            dervied_questions = dervied_questions.rename(columns={'_index':'derivedIndex'})

            dervied_questions['questionIndex'] = questionIndex
            return dervied_questions

        elif level == 'drag':
            drags = []
            for i in range(len(data)):
                derivedIndex, drag, questionIndex =  data.iloc[i]
                
                # gameCreationDateTime
                gameCreationDateTime = json_normalize(drag)['creationDateTime']
                                
                try:
                    drag = json_normalize(drag, record_path='dragData')  # CreationTime은 있는데 'dragData'노드가 없는 경우가 있음..
                    drag['questionIndex'] = questionIndex
                    drag['derivedIndex'] = derivedIndex
                    drag['gameCreationDateTime'] = gameCreationDateTime
                    drags.append(drag)
                except:
                    pass
            drags = pd.concat(drags)
            return drags.reset_index(drop=True)
        
    def _merge(self, data):
        device_info = self._parsing_by_level(data, level='device')
        a = self._parsing_by_level(device_info, level='question')
        b = self._parsing_by_level(a, level='derivedQuestion')
        c = self._parsing_by_level(b, level='drag')
        
        # question level에서 '_questionManagerCategory' 가 없는 경우도 있음
        if '_questionManagerCategory' in a.columns:
            d = a[['contentIndex', 'questionIndex', '_questionManagerCategory']].merge(c, on='questionIndex')
        else:
            d = a[['contentIndex', 'questionIndex']].merge(c, on='questionIndex')
        
        
        # Drop subnode
        device_info = device_info.drop('contentDragData', axis=1)
        
        return pd.concat([device_info, d], axis=1).fillna(method='ffill')
    
    def _parsing_user_node(self, json):
        '''
        json: usernode
        '''
        mapper = {'_deviceModel':'deviceModel',
                  '_deviceName':'deviceName',
                  '_dpi':'DPI',
                  '_screenHeight':'screenHeight',
                  '_screenWidth':'screenWidth',
                  '_level':'level',
                  '_index':'Index_',
                  '_index_x':'Index_X',
                  '_questionManagerCategory':'questionManagerCategory',
                  '_index_y':'Index_Y',
                  'appVersion':'appVersion',
                  'creationDateTime':'creationDatetime',
                  'category':'category',
                  'creationDateTime_m':'creationDatetime_M',
                  'GameDrivenIndex':'GameGrivenIndex',
                  'isOnCorrectAnswer':'IsOncorrectAnswer',
                  'posX':'PosX',
                  'posY':'PosY',
                  'touchPressure':'TouchPressure',
                  'Date':'Date',
                  'ID':'ID'}
        
        # parsing user json
        data = self._merge(json)
        data = data.rename(columns=mapper)
        
        return data

    def parsing_by_json(self, data):
        
        empty = []
        
        for OS in ['Android', 'iOS']:
            dfs = []
            os_specific_data = self._parsing_by_os(data, os=OS)
            ids = list(os_specific_data.keys())
            
            for id_ in tqdm(ids):
                df = self._parsing_user_node(os_specific_data[id_])
                df['accountID'] = id_
                dfs.append(df)
            df = pd.concat(dfs)
            
        # columns check
        target_cols = ['accountID', 'deviceModel', 'deviceName', 'DPI', 'screenHeight', 'screenWidth', 'level',
                       'contentIndex', 'questionIndex', 'derivedIndex', 'questionManagerCategory', 'gameCreationDateTime',
                       'creationDatetime', 'IsOncorrectAnswer', 'PosX', 'PosY', 'TouchPressure']
        return df[target_cols]
    
    def parsing_by_jsons(self, data):
        ''' root nodes comprised of list
        '''
        dfs = []
        for i in tqdm(range(len(data))):
            
            data_tmp = data[i]

            id_ = data_tmp.keys()  # id
            id_ = list(id_)[0]
            
            try:
                df = self._parsing_user_node(data[i][id_])
                df['accountID'] = id_
                dfs.append(df)
            except:
                pass
            
        df = pd.concat(dfs)
        
        # columns check
        target_cols = ['accountID', 'deviceModel', 'deviceName', 'DPI', 'screenHeight', 'screenWidth', 'level',
                       'contentIndex', 'questionIndex', 'derivedIndex', 'questionManagerCategory', 'gameCreationDateTime',
                       'creationDatetime', 'IsOncorrectAnswer', 'PosX', 'PosY', 'TouchPressure']
        return df[target_cols]
        
        
    def insert_into_tb(self, table, conn, table_name):
        
        cursor = conn.cursor()
        
        table = table.where((pd.notnull(table)), None)  # Replace nan to None
        
        for row in tqdm(range(len(table))):
            my_dict = dict(table.iloc[row,:])
            placeholder = ', '.join(['%s'] * len(my_dict))
            columns = ', '.join(my_dict.keys())
            sql = "INSERT INTO {} (%s) Values (%s)".format(table_name) % (columns, placeholder)
            dict_val = list(my_dict.values())
            cursor.execute(sql, dict_val)

            if row % 100000 == 0:
                conn.commit()
            
        conn.commit()
        print('commit')

        
        
        
        
        
        