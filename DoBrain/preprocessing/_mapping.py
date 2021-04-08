#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Written Sierra Lee. justforher12344@gmail.com
# Co-worker: by Ho Heon Kim.  hoheon0509@gmail.com 

import pandas as pd


class VersionMapper(object):
    '''
    To merge DoBrain game data with map_version data
    
    
    Parameters
    ----------
    map_version: str. path of map_version.xlsx
        
    
    Example
    -------
    from DoBrain.preprocessing import VersionMapper
    
    map_version = './data/두브레인 콘텐츠 MapVersion.xlsx'
    vm = VersionMapper(map_version)
    df = vm.code_mapping(score_new_dobrain)
    
    '''
    
    def __init__(self, map_version):      
        self.xl = pd.ExcelFile(map_version)        
        self.sheet_name = self.xl.sheet_names
        self.mapversion_excel = [self.xl.parse(i) for i in self.xl.sheet_names]       
        
        
        
    def _map_version_prep(self, sheet_order):      
        ''' To remove useless rows & columns in map version.
        
        Paratmers
        ---------
        sheet_order: int
        
        
        '''
        # Remove extra col, rows
        if sheet_order < len(self.sheet_name) - 2:
            df = self.mapversion_excel[sheet_order].iloc[2:,1:-3].fillna(0)
        else :
            df = self.mapversion_excel[sheet_order].iloc[2:,1:].fillna(0)
        
        # Rename columns
        df.columns = ['edu_session_index',
                      'edu_session_type',
                      'content_index',
                      'id',
                      'question_index=0', 
                      'question_index=1',
                      'question_index=2',
                      'question_index=3',
                      'question_index=4',
                      'question_index=5',
                      'question_index=6',
                      'question_index=7']

        return sheet_order, df
    
    
    def _previous_styling(self, data, dtype='score'):
        ''' To convert 2020 DoBrain data structure into 2019 style

        Drag-------
        index : total stroke

        '''
        
        # level : profileLevel -> level
        data = data.replace({100:'A', 200:'B', 300:'C'})
        
        # Story game
        data = data.loc[data.edu_session_type == 'DoBrainStory']
        
        # Select
        if dtype == 'score':
            cols = ['accountId', 'profileLevel', 'content_index',
                    'question_order', 'derivedIndex',
                    'creationUtcDateTimeGame', 'duration', 'isRight', 'point']
        elif dtype == 'drag':
            game_cols = ['accountId', 'profileLevel', 'content_index',
                        'question_order', 'derivedIndex',
                        'creationUtcDateTimeGame']
            cols = ['deviceModel', 'dpi',
#                     'indexNum',
                    'index',
                    'screenHeight',
                    'screenWidth', 'type', 'creationUtcDateTimeTouch',
                    'posX', 'posY', 'touchPressure']
            cols = game_cols + cols

        data = data[cols]


        # Rename
        mapper = {'accountId':'userID',
                 'profileLevel':'level',
                 'content_index':'contentIndex',
                 'question_order':'questionIndex',
                 'creationUtcDateTimeGame':'clearDateTime',
                 'screenHeight':'ScreenHeight',
                 'screenWidth':'ScreenWidth',
                 'creationUtcDateTimeTouch':'CreationDateTime',
                 'deviceModel':'DeviceModel',
                 'type':'IsOncorrectAnswer',
                 'indexNum': 'index'
                 }
        data = data.rename(columns=mapper)


        # For return
        if dtype == 'score':
            data = self._incorrect_answer_cnt(data)
        elif dtype == 'drag':
            # Columns has been renamed.
            for i, col in enumerate(cols):
                if col in list(mapper.keys()):
                    cols[i] = mapper[col]

            data = data[cols]
        return data
        
    
    def _incorrect_answer_cnt(self, data, verbose=True):
        ''' Calcucalte incorrect answer count
        
        '''
        data = data.sort_values(['userID', 'level', 'contentIndex', 'questionIndex', 'derivedIndex', 'clearDateTime'])
        data = data.drop_duplicates().reset_index(drop=True)

        # iteration (by rows)
        incorrectAnwerCnts = []

        for i in range(len(data)):

            # initialization
            if i == 0:
                if  data.iloc[i]['isRight'] == True:
                    incorrectAnwerCnt = 0
                else:
                    incorrectAnwerCnt = 1
                incorrectAnwerCnts.append(incorrectAnwerCnt)

            else:
                pre_pt = data.iloc[i-1]['point']
                current_pt = data.iloc[i]['point']

                if pre_pt == current_pt:
                    incorrectAnwerCnts.append(incorrectAnwerCnt)
                else:
                    if data.iloc[i]['isRight'] == True:
                        incorrectAnwerCnt = 0
                    else:
                        incorrectAnwerCnt = 1
                    incorrectAnwerCnts.append(incorrectAnwerCnt)

        data['incorrectAnswerCount'] = incorrectAnwerCnts
        
        
        # Duration
        durations = []
        
        for i in range(len(data)):
            if data.iloc[i]['incorrectAnswerCount'] == 0:
                duration = data.iloc[i]['duration']
                durations.append(duration)
            else:
                backward = data.iloc[i]['incorrectAnswerCount']
                duration = sum(data.iloc[i-backward:i+1]['duration'])
                durations.append(duration)
        data['tDuration'] = durations
                         
        

        # Summary
        data = data.loc[data.isRight == True]
        data = data.drop(['isRight', 'duration'], axis=1)
        data = data.rename(columns={'tDuration':'duration'})
        return data
    

        
        

    def _extraction(self, node):
        ''' contents in cell -> list
        '''

 

        return_list = []

 
        # F columsn ~ M columns in Map version 
        for col in range(4,12):
            
            question_order = col - 4 
            level0 = node.iloc[0, col] # cell
            
            if level0 != 0:
                level1 = level0.split('=') 
                
                category = level1[0]
                level2 = level1[1].split(';')  # split by game level
                
                for question_index in level2[:-1]:
                    level3 = question_index.split(':')
                    
                    if level3[0] == 'A':
                        level = 100
                    elif level3[0] == 'B':
                        level = 200
                    else :
                        level = 300
                        
                    # derived indexes                       
                    derived_indexes = 0 
                    for derived_id in level3[1].split('/'):
                        
                        return_list.append([level, derived_id, question_index, question_order, derived_indexes, category])
                        derived_indexes += 1
 

        return return_list

 

            
    
    def _map_version_to_key(self):
        ''''''
        parsed_list = []

        for i in range(len(self.sheet_name)):
        
            # map_version prep
            sheet_order, df = self._map_version_prep(i)

            for j in range(len(df)):
                map_version = "normal_{}".format(self.sheet_name[sheet_order]) 
                edu_session = df.iloc[j,0]
                edu_session_type = df.iloc[j,1]

                if edu_session_type != 'DoBrainGame':

                    content_index = df.iloc[j,2]
                    new_df = self._extraction(df.iloc[j:j+1,:])

                    for question_id in range(len(new_df)):

                        f = [map_version, edu_session, edu_session_type, content_index] + new_df[question_id]
                        parsed_list.append(f)
        
        # Make FK dataframe with long type form
        col_names = ['mapVersion',
                     'edu_session_index',
                     'edu_session_type',
                     'content_index',
                     'profileLevel',
                     'derivedQuestionId',
                     'question_index', # sub-game 번호: question_index
                     'question_order',
                     'derivedIndex',  # derived index for 1st, 2nd plays in sub-game
                     'category']
        parsed_df = pd.DataFrame(parsed_list, columns=col_names)            

        return parsed_df
            
    
    def code_mapping(self,
                     score_data,
                     previous_version_style=True,
                     retrun_ordered=False,
                     dtype='score'):
        '''
        Paramters
        ----------
        score_data: pd.DataFrame. new dobrain score data.
        
        old_style: like 2019 DoBrain Version
        
        previous_version_style: bool. 
            If True for ordered columns in pd.DataFrame

        retrun_ordered: bool
        
        
        Return
        ------
        pd.DataFrame
        
        '''
        
        parsed_df = self._map_version_to_key()
        merged_data = pd.merge(parsed_df, score_data, on = ['mapVersion','derivedQuestionId','profileLevel'])
        
        # Old_style
        if previous_version_style:
            merged_data = self._previous_styling(merged_data, dtype)
            return merged_data
        
        # Order
        if retrun_ordered:
            add_cols = ['edu_session_type', 'edu_session_index', 'content_index', 'question_index', 'question_order', 'derivedIndex']
            idx_cols = list(zip(range(5, 5+len(add_cols)), add_cols))
            
            col_names = list(score_data.columns)
            for pair in idx_cols:
                col_names.insert(*pair) 

            return merged_data[col_names]
        
        
        return merged_data



class DragMapper(object):

    def __init__(self, data):
        self.data = data

    def _derived_var(self):
        self.data['Date'] = self.data['clearDateTime'].apply(lambda x: str(x)[:10])


    def to_standard_style(self):
        '''transform new version of data into old style of app version

        parameters
        ----------
        data: pd.DataFrame

        :return
        pd.DataFrame
        '''


        mapper = {'ID':'userID',
                  'Index_': 'contentIndex',
                  'Index_X': 'questionIndex',
                  'Index_Y': 'derivedIndex',
                  'Dpi':'dpi',
                  'Level':'level',
                  'CreationDatetime_M': 'CreationDateTime',
                  'CreationDatetime':'clearDateTime',
                  'PosX':'posX',
                  'PosY':'posY'
                  }

        # New columns generation
        # self._derived_var()

        return self.data.rename(columns=mapper)

        mapper2 = {'CreationDatetime_M': 'CreationDateTime'}
        return self.data.rename(columns=mapper2)
