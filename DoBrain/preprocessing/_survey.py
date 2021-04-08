import re
import numpy as np

def code_gender(gender):
    ''' To str gender type into numeric

    Parameters
    ----------
    gender : str

    :return
    int:
        male : 0
        female: 1
        Unknown: 2
    '''

    if gender == '남':
        return 0
    elif gender in ['boy', 'Boy']:
        return 0
    elif gender in ['여', 'gril', 'Gril', 'Girl']:
        return 1
    elif gender is np.nan:
        return 2


def to_date_time(date):

    # int type 인경우
    if type(date) is int:
        pass

    if len(date) == 0:
        return np.nan
    else:
        return date


def birthday_to_age(birthday, init_game_date):
    ''' birthday to age (months)

    Paramteres
    ----------
    birthday: str. YYYY/MM, or YYYY-MM-DD, or YYYY-MM-DD HH-MM-SS
    init_game_date: YYYY-MM-DD

    :return:
    months: int
    '''

    try:
        return int(birthday)
    except:
        pass

    if len(birthday) == 0:
        return np.nan


    if bool(re.search('[0-9]{4}\/[0-9]{2}', birthday)):
        return birthday + '/01'
    else:
        return birthday


def missing_gender(gender, sex):
    ''' Missing gender fill-up
    
    '''
    pair = np.array([gender, sex])
    n_nan = np.isnan(pair).sum()
    
    if n_nan == 0:
        return gender
    elif n_nan == 1:
        if np.isnan(np.array(gender)):
            return sex
        else:
            return gender
    elif n_nan == 2:
        return np.nan


    
def ext_diagnosis(string, dx):
    ''' diagnosis information parsing
    '''
    
    dx = str(dx)
    if dx == 'ASD':
        keywords = ['ASD', 'Autism', '자폐', 'Spec']
    elif dx == '언어발달지연':
        keywords = ['언어', 'language']
    elif dx == 'ADHD':
        keywords = ['ADHD', '주의력']
    elif dx == 'ID':
        keywords = ['ID', 'intel', 'MD', '지적']
    elif dx == 'DD':
        keywords = ['Delay', '지연']
    elif dx == 'other':
        keywords = ['BL', 'Down', '유전', '조음', '기타', '청각', '뇌병변', '대근육']
    return bool(re.search('|'.join(keywords), string, re.IGNORECASE))