#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Written by Ho Heon Kim.  hoheon0509@gmail.com 
# Co-worker: Sierra Lee. justforher12344@gmail.com


# Updated
# -

import re
import pandas as pd
import json
import requests
import pymysql
from datetime import date, timedelta

class DragData(object):
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
            for i in range(len(subnode)): 
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
        Below column Description
        
        
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
            strokePlayLogs = data.rename({'derivedQuestionPlayLogId' : 'id'}, axis = 1)
            return strokePlayLogs
            
        elif node_name == 'strokeValuePlayLogs': 
            data = data.drop(columns = ['id'])
            strokeValuePlayLogs = data.rename({'creationUtcDateTime':'creationUtcDateTimeTouch'}, axis = 1)
            return strokeValuePlayLogs
        
        elif node_name == 'level0':
            level0 = pd.DataFrame.from_dict([data])
            level0 = level0.drop(columns = ['answerPlayLogs', 'strokePlayLogs'])
            return level0
            
        elif node_name == 'answerPlayLogs' :
            data = data.drop(columns = ['id'])
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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
# 2019 dobarin -----------------------------------


def survey_parser(OS):
    '''
    Dobrain firebase의 report survey data를 가공하여 dataframe으로 반환
    
    Parameters
    ----------
    OS : str. 
        'Android' : For Android users
        'iOS' : For iOS users
    
    
    Return
    ---------
    pd.DataFrame
    
    '''
    # Firebase : JSON
    url = 'https://dobrain-pro.firebaseio.com/report_survey_data'
    target_url = url + '/' + OS + '.json?shallow=true'
    resp = requests.get(url=target_url) # 리퀘스트 
    json_result = resp.json() # Json type으로 변경
    
    # JSON -> pandas.DataFrame
    ID_list = list(json_result.keys())
    ID_result = []
    State_result = []
    OS_result = []
    gender = []
    Date = []
    
    for id_ in ID_list:
        target_url = url + '/' + OS + '/' + id_ + '.json'  # OS 종류에 따른 json URL주소
        
        try:
            ID_result.append(id_)
            OS_result.append(OS)
            request_obj = requests.get(url=target_url)                
            json_obj = request_obj.json()
            if json_obj['wonDiagnosis'] == '없다':
                State_result.append('Normal')
            elif json_obj['wonDiagnosis'] == '있다':
                State_result.append('Abnormal')
            else :
                State_result.append('Worry')
                
            gender.append(json_obj['gender'])
            Date.append(json_obj['creationDateTime'])

        except:
            pass
        
    row = list(zip(Date, ID_result, OS_result,State_result, gender))
    survey_df = pd.DataFrame(row, columns=['Date','ID','OS','State','Gender'])
    survey_df['Date'] = survey_df.apply(lambda x : x['Date'].split()[0], axis=1)

    return survey_df


def date_format_convert(df):
    '''
    Date 형식을 yyy-MM-dd로 통일
    
    Parameters
    ----------
    df : pd.DataFrame.
        Date 이름의 열이 있는 Dataframe
        
    Return
    ---------
    DataFrame
    
    '''
    date_tmp = [re.sub('/', '-', _ ) for _ in df.Date]
    date_final = []
    
    for _ in date_tmp:
        if _[2]=='-':
            temp = str(_[6:])+'-'+str(_[:2])+'-'+str(_[3:5])
            date_final.append(temp)
        else:
            date_final.append(_)
            
    df['Date'] = date_final
    
    return df


def load_date(start_date, end_date, json_folder):
    '''
    drag_data폴더에 저장되어 있는 JSON파일 중, 서버에 업로드 할 날짜 선택
    
    시작일, 마지막일 모두 포함해서 선택
    
    
    
    json_folder = './drag_data/'
    
    Parameters
    ----------
    start_date: str. 
        
    end_date: str.
    json_folder: str.
    
    Return
    ----------
    list: 시작일과 마지막일 사이의 모든 날짜의 Json파일명
        * 각 원소는  json file name
        
    Exmaples
    --------
    load_data('2016-03-03', '2019-12-31', './data/')
    
    JSON파일을 전달받는 구글 드라이브 주소 : https://drive.google.com/drive/folders/1GgvUB_EfDLcZmbygfVFFTpyAj7R4jvvs
    다운로드 받아야함.
    
    '''
    
    sdate = pd.to_datetime(start_date)   # start date
    edate = pd.to_datetime(end_date)     # end date

    arr = os.listdir(json_folder)
    file_list =[]
    for _ in arr:
        if _.split('_')[-1].split('.')[0] in date_list:
            file_list.append(_)
    file_list = sorted(file_list)
    
    return file_list


def outerjoin(left_df, right_df):
    '''
    두 DataFrame을 outer join한 DataFrame을 반환
    
    Parameter
    ------------
    left_df : pd.DataFrmae.
        왼쪽에 위치할 DataFrame
    right_df : pd.DataFrame.
        오른쪽에 위치할 DataFrame
        
    Return
    -----------
    pandas.DataFrame
    
    '''
    left_df['dummy']= 'HH'
    right_df['dummy'] = 'HH'
    result = pd.merge(left_df, right_df, on='dummy', how='left')
    
    return result.loc[:, result.columns != 'dummy']


def JSON_to_Server_parser(file_list, server_schema, json_folder, conn, cursor):
    '''
    file_list의 날짜에 해당하는 데이터를 Server에 적재
    
    Parameter
    ----------
    file_list : list.
        날짜를 포함한 list
    server_schema : str.
        데이터를 적재할 서버의 테이블명 (Ex. Drag_Data_by_Date_2)
    json_folder: str.
        JSON 파일이 저장되어 있는 경로 ('./drag_data/')
        
    Return
    ----------
    None
    
    Example
    ----------
    JSON_to_Server_parser(file_list, 'Drag_Data_by_Date_2', './drag_data/')
    
    '''
    
    mapper = {'_deviceModel':'DeviceModel', 
          '_deviceName':'DeviceName',
          '_dpi':'Dpi',
          '_screenHeight':'ScreenHeight',
          '_screenWidth':'ScreenWidth',
          '_level':'Level',
          '_index':'Index_',
          '_index_x':'Index_X',
          '_questionManagerCategory':'QuestionManagerCategory',
          '_index_y':'Index_Y',
          'appVersion':'AppVersion',
          'creationDateTime':'CreationDatetime',
          'category':'Category',
          'creationDateTime_m':'CreationDatetime_M',
          'GameDrivenIndex':'GameGrivenIndex',
          'isOnCorrectAnswer':'IsOncorrectAnswer',
          'posX':'PosX',
          'posY':'PosY',
          'touchPressure':'TouchPressure',
          'Date':'Date',
          'ID':'ID'}
    
    
    print('generate cursor')
    SQL = 'INSERT INTO '+ server_schema + ' VALUES (%s)' % str('%s,'*20 + '%s')
    for _ in file_list:
        print(_)
        json_data=open(json_folder+_).read()
        data = json.loads(json_data)
        for id_ in tqdm_notebook(range(len(data))):
            final=pd.DataFrame()
            data_temp = data[id_]
            temp_ = pd.DataFrame.from_dict(data_temp)
            left = temp_.iloc[:-1]
            right = temp_.iloc[-1]
            left=left.drop_duplicates()
            left = left.T
            left.reset_index(inplace=True)
            left.rename(columns={'index':'ID'}, inplace=True) # ID, DeviceMdoel, DeviceName, dpi, screenHieght, Screen Width
            for level in range(len(right)):
                try:
                    right1 = pd.DataFrame.from_dict(right.values[level]).iloc[:,-1]
                    left1 = pd.DataFrame.from_dict(right.values[level]).iloc[:,:-1] #Level
                    left1=left1.drop_duplicates()

                except IndexError:
                    print("There is no id in the date")
                    continue
                    
                for h in right1:
                    left2 = pd.DataFrame.from_dict(h).iloc[:,:-1] #동화 번호
                    right2 = pd.DataFrame.from_dict(h).iloc[:,-1]
                    left2=left2.drop_duplicates()
                    
                    for h_ in range(len(left2)):
                        left3 = pd.DataFrame.from_dict(right2.iloc[h_]).iloc[:,:-1] #컨텐츠 번호
                        right3 = pd.DataFrame.from_dict(right2.iloc[h_]).iloc[:,-1]
                        left3=left3.drop_duplicates()
       
                        for i in range(len(left3)):
                            left4 = pd.DataFrame.from_dict(right3.iloc[i]).iloc[:,:-1] #Index_Y
                            right4 = pd.DataFrame.from_dict(right3.iloc[i]).iloc[:,-1]
                            left4=left4.drop_duplicates()

                            for j in range(len(left4)):
                                left5 = pd.DataFrame.from_dict(right4.iloc[j]).iloc[:,:-1] #appversion, creationDateTime
                                right5 = pd.DataFrame.from_dict(right4.iloc[j]).iloc[:,-1]
                                left5=left5.drop_duplicates()
                                
                                for k in range(len(left5)):
                                    try:
                                        left6 = pd.DataFrame.from_dict(right5.iloc[k]) #PosX, PosY, CreationDatatime_M, ...
                                        left6.rename(columns={'creationDateTime':'creationDateTime_m'}, inplace=True)
                                        output=outerjoin(pd.DataFrame(left5.iloc[k]).T,pd.DataFrame(left6))
                                        output1=outerjoin(pd.DataFrame(left4.iloc[j]).T,output)
                                        output2=outerjoin(pd.DataFrame(left3.iloc[i]).T,output1)
                                        output3=outerjoin(pd.DataFrame(left2.iloc[h_]).T,output2)
                                        output4=outerjoin(pd.DataFrame(left1),output3)
                                        output5=outerjoin(left,output4)
                                        output5['Date']=_.split('_')[-1].split('.')[0]
                                        final=final.append(output5, ignore_index=True)
                                        
                                    except ValueError:
                                        continue

                                        
        
    
            final.rename(columns=mapper, inplace=True)
            
            # Fillna(0)
            for col in mapper.values():
                if col not in final.columns:
                    final[col]=0
                else:
                    continue
            final=final[mapper.values()]

            for num in range(len(final)):
                try:
                    cursor.execute(SQL, final.iloc[num,:])
                    
                except:
                    break
            conn.commit()
    cursor.close()
    conn.close()
