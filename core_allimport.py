# -*- coding: utf-8 -*-
####################################################################################################
import gc
import os
import logging
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn import covariance, linear_model, model_selection, preprocessing
from sklearn.cluster import dbscan, k_means
from sklearn.decomposition import PCA, pca, TruncatedSVD, LatentDirichletAllocation, NMF

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, make_scorer,
                             mean_absolute_error, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



#################
import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb
from tqdm import tqdm_notebook
import pickle
import seaborn as sns
import matplotlib.pyplot as plt




print("imported", lgb)





def save(path, name_list, glob) :
  import pickle, os
  os.makedirs(path, exist_ok=True)
  for t in name_list :
    print(t)
    pickle.dump( glob[t] , open( f'{t}.pkl', mode='wb'))


def load(name) :
  import pickle  
  return pickle.load(  open( f'{name}.pkl', mode='rb') ) 
    




        



