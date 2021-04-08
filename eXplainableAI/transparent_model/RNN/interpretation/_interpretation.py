import re
import numpy as np
import matplotlib.pyplot as plt

from functools import reduce
from tensorflow.keras.models import Model

class Interpreter(object):
    
    '''
    Parameters
    ----------
    model: keras.model
    x_time_var: np.array.
        instance's time variant features
    x_aux: np.array.
        instance's time invariant features
    y: np.arary
    
    Raise
    -----
    ValueError
    '''
    
    def __init__(self, model, x_time_var, x_aux, y):
        self.model = model
        self.x_time_var = x_time_var
        self.x_time_inv = x_aux
        self.y_data = y

        if (len(y) != len(x_time_var)) or (len(y) != len(x_aux)):
            raise ValueError('y, x_time_var, x_aux must have same length but got: y, x_time, x_aux', len(y_data), len(x_time_var), len(x_aux))

    def get_model_weight(self, return_val='all'):
        '''get model weight in each layer
        
        Parameters
        ----------
        return_val: str
            * 'alpha'
            * 'beta'
            * 'weight': weight of fully connected layer, and bias
            * 'bias'
            * 'multi-weight'
            (default = None) : for retuning perdicted Y, alpha, beta
        
        
        Returns
        -------
        Tuple: predicted_y, alpha, beta
        '''
        from tensorflow.keras.models import Model

        model_with_attention = Model(self.model.inputs, [self.model.output, 
                                                        self.model.get_layer(name='alpha_softmax').output, 
                                                        self.model.get_layer(name='beta_dense').output])
        predicted_y, alpha, beta = model_with_attention.predict_on_batch([self.x_time_var, self.x_time_inv]) # Must be 3-D shape
        
        # Return
        if return_val == 'all':
            return predicted_y, alpha, beta
        
        elif return_val == 'alpha':
            return alpha
        
        elif return_val == 'beta':
            return beta
        
        elif return_val == 'weight':
            W = self.model.get_layer('output').get_weights()[0]
            bias = self.model.get_layer('output').get_weights()[1]
            return W, bias
        
        elif return_val == 'bias':
            return self.model.get_layer('output').get_weights()[1]
        
        elif return_val == 'multi_weight':
            import re
            from functools import reduce

            # Cascade weight (fully connected layers)
            names = [layer.name for layer in self.model.layers if bool(re.search('fc', layer.name))]

            weights = [self.model.get_layer(name).get_weights()[0] for name in names]
            weights = weights + [self.model.get_layer('output').get_weights()[0]]
            cascade_weight = reduce(lambda x, y: np.matmul(x, y), weights)

            return cascade_weight
        
        
    def plot(self, weeks=None):
        ''' plot the coefficient contribution
        
        '''
        
        assert (weeks == None) or (weeks>=1)
        
        # 1. Get model weight
        predicted_y, alpha, beta = self._get_model_weight(return_val=None)
        W = self._get_model_weight(return_val='weight')        
        
        # For Two Fully connected layers
        
        # Fro One fully connected layer

        
        # F
        user_kg = self.y_data
        fig = plt.figure(facecolor='w', figsize=(7,7))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(0, 17)
        actual_logs = wt_mat.loc[(wt_mat['1week'] == self.x_time_inv.reshape(-1,16,5)[idx,0,0]) & 
                                 (wt_mat['16week'] == self.y_data.reshape(-1,16)[idx,0])]
        
        # Figure
        if weeks == None:
            ax = fig.add_subplot(1,1,1)
            ax.plot([i for i in range(1, 16)], np.array(actual_logs)[0][:16].astype('float')) # 실제 체중감소
            ax.scatter(15, y_predict.reshape(-1,16)[idx][15], color='red')
            ax.annotate('predicted: {}kg loss'.format(round(user_kg)), 
                        xy=(15, x_test_inv.reshape(-1,16,5)[idx,0,0]-y_predict[idx][0]), 
                        xytext=(15, x_test_inv.reshape(-1,16,5)[idx,0,0]-y_test.reshape(-1,16,1)[idx,0]+3), 
                        arrowprops=dict(arrowstyle='->'))

        else:
            actual_logs = np.array(actual_logs)[0][:16].astype('float')
            actual_y = actual_logs[:weeks+1]
            y[:weeks] = actual_y
            ax.plot([i for i in range(1, weeks+2)], actual_y) # 실제 체중감소
            ax.scatter(16, y_predict.reshape(-1,16,1)[idx][0], color='red')
            ax.annotate('predicted: {}kg loss'.format(round(y_predict.reshape(-1,16,1)[idx][0][0])), 
                        xy=(16, y_predict.reshape(-1,16,1)[idx][0]), 
                        xytext=(14,y_test.reshape(-1,16,1)[idx][0]+3), 
                        arrowprops=dict(arrowstyle='->'))
            
    def save_gif(self, case, value='weight'):
        '''
        Parameters
        ----------
        case: case number in input x_data
        
        '''
        import os
        import re
        import imageio
        import matplotlib.pyplot as plt
        
        # fig save
        if value == 'weight':
            folder = './result/case{}/'.format(case)
        elif value == 'features':
            folder = './result/case{}/features/'.format(case)
            
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        
        if value == 'weight':
            for weeks in range(1, 16):
                self.plot(case, weeks)
                plt.savefig(folder + '{:02d}week.png'.format(weeks))
                plt.close()
                print('case: {}, fig:{} th saved'.format(case, weeks), end='\r')
        elif value == 'features':
            for weeks in range(1, 16):
                self.plot_x(case, weeks)
                plt.savefig(folder + '{:02d}week.png'.format(weeks))
                plt.close()
                print('case: {}, fig:{} th saved'.format(case, weeks), end='\r')
            
        # File names
        files = os.listdir(folder)
        files = sorted([file for file in files if bool(re.search('\.png', file))])
        figure_path = os.getcwd() + folder[1:]
        filenames = [figure_path+file for file in files]

        # GIF maker
        images =[]
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(folder+'{}.gif'.format(case), images, fps=3)
    
    
    def plot_x(self, case, weeks=None, save=False):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        df = pd.DataFrame(self.x_data.reshape(-1, 16, 16, 6)[case, weeks], columns=FEATURES)
        df = pd.DataFrame(df.stack()).reset_index().rename({'level_0':'weeks', 'level_1':'variables', 0:'value'}, axis=1)
        
        # plot
        fig, axes = plt.subplots(3, 2, figsize=(15,6))
        for _ in range(len(FEATURES)):
            var_df = df.loc[df.variables == FEATURES[_]]
            if _ <= 2:
                sns.lineplot(x='weeks', y='value', data=var_df, ax=axes[_, 0])
                axes[_, 0].set(xlabel='Time', ylabel=FEATURES[_])
            else:
                sns.lineplot(x='weeks', y='value', data=var_df, ax=axes[_-3, 1])
                axes[_-3, 1].set(xlabel='Time', ylabel=FEATURES[_])
                
          
    def coef_plot(self, accessCode, weeks=None):
        pass



class InterpreterWithGenerator(Interpreter):
    '''
    Interpreter class for model with generator (batch bucketting)

    Parameters
    ----------
    model : keras.model

    '''

    def __init__(self, model, test_index, generator):
        self.model = model
        self.test_index = test_index
        self.generator = generator

    def get_model_weights(self):
        '''get model weight in each layer

        Parameters
        ----------
        None

        Returns
        -------
        Generator to return predicted Y, alpha, beta, W (Casecade weights)
        '''


        model_with_attention = Model(self.model.inputs, [self.model.output,
                                                         self.model.get_layer(name='alpha_softmax').output,
                                                         self.model.get_layer(name='beta_dense').output])
        # iterative return
        for i in range(len(self.test_index)):
            [x, x_aux], label = next(self.generator)

            # step must be 1 due to various timestamp
            predicted_y, alpha, beta  = model_with_attention.predict([x, x_aux])

            # Cascade weight
            names = [layer.name for layer in self.model.layers if bool(re.search('fc', layer.name))]

            weights = [self.model.get_layer(name).get_weights()[0] for name in names]
            weights = weights + [self.model.get_layer('output').get_weights()[0]]
            cascade_weight = reduce(lambda x, y: np.matmul(x, y), weights)

            yield predicted_y, alpha, beta, cascade_weight
