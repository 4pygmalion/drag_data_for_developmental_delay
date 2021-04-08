import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, file_name=None):

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    if file_name is None:
        pass
    else:
        plt.savefig(file_name)
        

def model_week_evaluation(model, x, y, eval_all=True):
    ''' Regresison model evaluation 
    
    parameters
    ----------
    x: list. x_test data
    y: array
    
    
    returns
    -------
    
    '''
    
    
    y_predict = model.predict(x)
    
    
    # 16주 전
    X_test_var = x[0]
    X_test_fix = x[1]
    len_vars = X_test_var[0].shape[-1]  # The number of time variant varible 
    len_inv = X_test_fix[0].shape[-1]
    
    # Model performance
    if eval_all == True:
        performances = []
        
        for week in range(16):
            tmp_x_test_var = X_test_var.reshape(-1, 16, 16, len_vars)[:, week, : : ].reshape(-1, 16, len_vars)
            tmp_x_test_fix = X_test_fix.reshape(-1, 16, len_inv)[:, week, :].reshape(-1, len_inv)
            tmp_y_test = y.reshape(-1, 16, 1)[:, week, :].reshape(-1, 1)
            y_predict = model.predict(x=[tmp_x_test_var, tmp_x_test_fix])
            
            MAPE = mean_absolute_percentage_error(tmp_y_test, y_predict)
            performances.append(MAPE)
        
        return performances
        
    week=0

    tmp_x_test_var = X_test_var.reshape(-1, 16, 16, len_vars)[:, week, : : ].reshape(-1, 16, len_vars)
    tmp_x_test_fix = X_test_fix.reshape(-1, 16, len_inv)[:, week, :].reshape(-1, len_inv)
    tmp_y_test = y.reshape(-1, 16, 1)[:, week, :].reshape(-1, 1)
    y_predict = model.predict(x=[tmp_x_test_var, tmp_x_test_fix])
    MAPE_1 = mean_absolute_percentage_error(tmp_y_test, y_predict)

    week=7
    tmp_x_test_var = X_test_var.reshape(-1, 16, 16, len_vars)[:, week, : : ].reshape(-1, 16, len_vars)
    tmp_x_test_fix = X_test_fix.reshape(-1, 16, len_inv)[:, week, :].reshape(-1, len_inv)
    tmp_y_test = y.reshape(-1, 16, 1)[:, week, :].reshape(-1, 1)
    y_predict = model.predict(x=[tmp_x_test_var, tmp_x_test_fix])
    MAPE_8 = mean_absolute_percentage_error(tmp_y_test, y_predict)

    week=13
    tmp_x_test_var = X_test_var.reshape(-1, 16, 16, len_vars)[:, week, : : ].reshape(-1, 16, len_vars)
    tmp_x_test_fix = X_test_fix.reshape(-1, 16, len_inv)[:, week, :].reshape(-1, len_inv)
    tmp_y_test = y.reshape(-1, 16, 1)[:, week, :].reshape(-1, 1)
    y_predict = model.predict(x=[tmp_x_test_var, tmp_x_test_fix])
    MAPE_14 = mean_absolute_percentage_error(tmp_y_test, y_predict)

    week=15
    tmp_x_test_var = X_test_var.reshape(-1, 16, 16, len_vars)[:, week, : : ].reshape(-1, 16, len_vars)
    tmp_x_test_fix = X_test_fix.reshape(-1, 16, len_inv)[:, week, :].reshape(-1, len_inv)
    tmp_y_test = y.reshape(-1, 16, 1)[:, week, :].reshape(-1, 1)
    y_predict = model.predict(x=[tmp_x_test_var, tmp_x_test_fix])
    MAPE_16 = mean_absolute_percentage_error(tmp_y_test, y_predict)
    
    return MAPE_1, MAPE_8, MAPE_14, MAPE_16
    
    

def mean_absolute_percentage_error(y_true, y_pred): 
    '''
    See Also
    --------
    https://stackoverflow.com/questions/47266383/save-and-load-weights-in-keras
    '''
    from sklearn.utils import check_array
    import numpy as np


    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          fontsize=15):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=fontsize,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     fontsize=fontsize,
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
def auc(y_true, y_pred):
    '''
    See also
    --------
    https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/80807
    '''
    
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)


    
def roc_curve(y_true, y_predict, n_classes=2):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true.ravel(), y_predict.ravel())
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_predict.ravel(), pos_label=2)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()