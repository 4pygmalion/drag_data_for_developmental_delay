
import pandas as pd

class GameSelector(object):
    
    def __init__(self, score_data, map_version):
        self.score = score_data
        self.map_version = pd.read_excel(map_version)
        self.game_cnt = self._game_cnt()

    def _game_cnt(self):
        ''' To remove useless rows & columns in map version.'''
        
        map_version = self.map_version.iloc[2:, 2:-3].reset_index(drop=True)
        map_version.columns = ['edu_session_type',
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
        map_version = map_version.loc[map_version.edu_session_type == 'DoBrainStory']
        map_version_cnt = map_version.set_index('content_index').stack().reset_index()
        
        # question 게임 카운트
        question_indexes = ['question_index={}'.format(i) for i in range(0, 8)]
        map_version_cnt = map_version_cnt.loc[map_version_cnt.level_1.isin(question_indexes)]
        game_cnt = pd.DataFrame(map_version_cnt.groupby('content_index').count().reset_index())
        game_cnt.columns = ['contentIndex', 'questionIndex', 'cnt']
        return game_cnt[['contentIndex', 'questionIndex', 'cnt']]
        
    
    
    def cnt_full_game_usr(self, level='A', delay_ids=None, target=30):
        '''
        각 동화를 모두 끝낸 아동들의 수를 카운트 함
        
        
        '''
        level_games =  self.score.loc[self.score.level == level]
        
        # questionIndex
        freq_cnt = pd.DataFrame(level_games.groupby(['userID', 'contentIndex', 'questionIndex'])['point'].count()).reset_index()
        freq_cnt = freq_cnt.groupby(['userID', 'contentIndex']).count().reset_index()
        freq_cnt = freq_cnt.merge(self.game_cnt, on=['contentIndex', 'questionIndex'], how='left')
        
        if delay_ids is not None:
            freq_cnt.loc[:, 'delay'] = False
            freq_cnt.loc[freq_cnt.userID.isin(delay_ids), 'delay'] = True
            
            freq_cnt =  freq_cnt[['userID', 'contentIndex', 'questionIndex', 'cnt', 'delay']]
            freq_cnt = freq_cnt.loc[~freq_cnt.cnt.isna()]
            
            results = []
            for order in range(1, target):
                freq = freq_cnt.loc[freq_cnt.contentIndex.isin(list(range(1, order+1)))]
                
                # order 개수와 맞는 사용자들
                mask = freq.groupby(['userID', 'delay']).count()['cnt'] == order 
                data = freq.groupby(['userID', 'delay']).count().loc[mask]
                data = pd.DataFrame(data).reset_index()
                col =  pd.DataFrame(data.groupby('delay').count()['cnt'])
                results.append(col)
                
            data = pd.concat(results, axis=1)
            data.columns = range(1, target)
            return data

            
        # Return
        
            
        freq_cnt =  freq_cnt[['userID', 'contentIndex', 'questionIndex', 'cnt']]
        freq_cnt = freq_cnt.loc[~freq_cnt.cnt.isna()]
        

def feature_comparsion(data, game_index, feature='point', level='A'):
    import sys
    sys.path.append('/home/hoheon/packages/')

    from HHstat.effect_size import cohen_d

    
    '''
    Parameters
    ----------
    data: pd.DataFrame
    game_index:
    
    
    Return
    ------
    tuple:  (t-test result, es)
 
    '''
    from scipy.stats import ttest_ind
    
    if type(game_index) is not tuple:
        raise ValueError('game_index not tuple')
    
    # level
    data = data.loc[data.level == level]
    
    # 각 사용자의 게임 histroy
    game_history = data.loc[game_index]
    
    abnormal_feature = game_history.loc[game_history.delay == False][feature]
    normal_feature = game_history.loc[game_history.delay == True][feature]
    

    # Result
    normal_mean = round(normal_feature.mean(), 3)
    normal_std = round(normal_feature.std(), 3)
    normal_stat = '{} ({})'.format(normal_mean, normal_std)
    
    abnormal_mean = round(abnormal_feature.mean(), 3)
    abnormal_std = round(abnormal_feature.std(), 3)
    abnormal_stat = '{} ({})'.format(abnormal_mean, abnormal_std)
    
    stat, p = ttest_ind(abnormal_feature, normal_feature)
    d = cohen_d(abnormal_feature, normal_feature)
    n = len(normal_feature), len(abnormal_feature)
    
    return normal_stat, abnormal_stat, stat, p, d, n
    
    
    
    
        
        