import numpy as np


def new_point(incorrectAnswerCount, duration):
    ''' To calculate new point in new app version
    
    Parameters
    ----------
    incorrectAnswerCount: float
    duration: float
    
    Return
    ------
    new_point: float
    '''
    new_point = 100 * np.exp(-0.3 * incorrectAnswerCount) + 100 * np.exp(-0.1 *duration)
    
    return new_point