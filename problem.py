import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import KFold



problem_title = "Real estate's prices from 2014 to 2018 in France"

Predictions = rw.prediction_types.make_regression()

workflow = rw.workflows.FeatureExtractorRegressor()


#--------------------------------------------
# Scoring
#--------------------------------------------



# Relative absolute error:


# class RAE(BaseScoreType):
    
#     def __init__(self, name='rae'):
#         self.name = name

#     def __call__(self, y_true, y_pred):
#         num_rae = 0
#         denum_rae = 0
#         for val_true, val_pred in zip(y_true, y_pred):
#             num_rae += np.abs(val_pred - val_true)
#             denum_rae += np.abs(np.mean(val_true) - val_true)
            
#         return num_rae / denum_rae



# score_types = [
    
#     # Relative absolute error:
# 	RAE(name='rae')

# ]

# # # # # # # # # # # # # # 



class RAE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')
    
    def __init__(self, name='rae', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        num_rae = 0
        denum_rae = 0
        for val_true, val_pred in zip(y_true, y_pred):
            num_rae += np.abs(val_pred - val_true)
            denum_rae += np.abs(np.mean(y_true) - val_true)
            
        return num_rae / denum_rae



score_types = [
    
    # Relative absolute error:
    RAE(name='rae', precision=2)

]







#--------------------------------------------
# Cross validation
#--------------------------------------------

def get_cv(X, y):
    cv = KFold(n_splits=5, random_state=45)
    #print("get_cv = ", cv.split(X, y))
    return cv.split(X, y)



    
#--------------------------------------------
# Data reader
#--------------------------------------------

_target_column_name = 'valeur_fonciere'


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False,
                        compression='gzip')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[::30], y_array[::30]
    else:
        return X_df, y_array
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'real_estate_train.csv.gz'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'real_estate_test.csv.gz'
    return _read_data(path, f_name)



