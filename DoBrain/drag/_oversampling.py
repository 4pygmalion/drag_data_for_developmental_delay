import numpy as np
import random
from imblearn.over_sampling import SMOTE, SMOTENC
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DragImageResampler(object):
    
    def __init__(self, x, x_aux, multiple):
        '''
        x: Minor class data
        x_aux: minor class data
        '''
        
        self.image = x
        self.n_sample = x.shape[0]
        self.x_aux = x_aux
        self.multiple = multiple
    
    
    def fit_resample(self, max_range):
        images = []
        
        shift_range = [random.uniform(-max_range, max_range) for i in range(100)]
        max_shift, min_shift = max(shift_range), min(shift_range)
        
        for i in range(self.multiple):
            print('-'*20, '{} th processed'.format(i+1), '-'*20, end='\r')
            shift_data_gen = ImageDataGenerator(width_shift_range=shift_range, 
                                                height_shift_range=shift_range,
                                                rotation_range=max_range)
            shift_data_gen.fit(self.image)
            X_shift = next(shift_data_gen.flow(self.image))
            images.append(X_shift)
            
        print('-'*20, 'In merging process', end='\r')
        
        return np.vstack(images)

    
    def fit_resample_with_replacement(self):
        return self.x_aux.repeat(self.multiple, axis=0)