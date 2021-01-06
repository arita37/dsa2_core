# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""

https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.lightgbm.train.html

https://github.com/optuna/optuna/blob/master/examples/lightgbm_tuner_simple.py





"""
import os
import pandas as pd, numpy as np, scipy as sci

import sklearn
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.tree import *
from lightgbm import LGBMModel, LGBMRegressor, LGBMClassifier

try :
  import optuna.integration.lightgbm as LGBMModel_optuna
except Exception as e :
  raise Exception(f"cannot import Optuna {e}" )


### https://github.com/optuna/optuna/blob/master/examples/pruning/lightgbm_integration.py


 
####################################################################################################
VERBOSE = True

# MODEL_URI = get_model_uri(__file__)

def log(*s):
    print(*s, flush=True)


####################################################################################################
global model, session


def init(*kw, **kwargs):
    global model, session
    model   = Model(*kw, **kwargs)
    session = None


class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars
        if model_pars is None:
            self.model = None
        else:
            model_object_name = 'LGBMModel_optuna'
            if 'LGBMClassifier' in model_pars['model_class']  :
               self.model_pars['model_pars']['objective'] =  'binary'

            elif  'LGBMRegressor' in model_pars['model_class'] :
               self.model_pars['model_pars']['objective'] =  'huber'

            model_class     = globals()[model_object_name]
            self.model_meta = model_class  ### Hyper param seerch Model
            self.model      = None         ### Best model saved after train
            #self.model = model_class()
            if VERBOSE: log(model_class, self.model)

def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xval, yval = get_dataset(data_pars, task_type="train")
    if VERBOSE: log(Xtrain.shape, model.model)


    dtrain = model.model_meta.Dataset(Xtrain,label = ytrain)
    dval   = model.model_meta.Dataset(Xval,  label = yval)
    # dtrain = LGBMModel_optuna.Dataset(Xtrain, label=ytrain)
    # dval = LGBMModel_optuna.Dataset(Xtest, label=ytest)

    pars_lightgbm = model.model_pars['model_pars']   #### from model_pars
    pars_optuna   = compute_pars.get("optuna_params", {})    ### Specific to Optuna
    optuna_engine = compute_pars.get('optuna_engine', 'simple')
    print("pars_lightgbm", pars_lightgbm)
    print("pars_optuna", pars_optuna)

    if optuna_engine == 'LightGBMTuner':
        model_fit = model.model_meta.LightGBMTuner(pars_lightgbm, dtrain, valid_sets=[dtrain, dval], **pars_optuna).run()

    elif optuna_engine == 'LightGBMTunerCV':
        model_fit = model.model_meta.LightGBMTunerCV(pars_lightgbm, dtrain, valid_sets=[dtrain, dval], **pars_optuna).run()

    else :
        model_fit = model.model_meta.train( pars_lightgbm, dtrain, valid_sets=[dtrain, dval], **pars_optuna)

    """
       print("Best score:", tuner.best_score)
       best_params = tuner.best_params
         
    """

    ### Best model store as
    print(model_fit.params)
    model.model                    = model_fit
    model.model_pars['model_pars'] = model_fit.params
    return model_fit



def eval(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    global model, session
    data_pars['train'] = True
    Xval, yval = get_dataset(data_pars, task_type="eval")
    # ypred      = model.model.predict(Xval)
    ypred = predict(Xval, data_pars, compute_pars, out_pars)

    # log(data_pars)
    mpars = compute_pars.get("metrics_pars", {'metric_name': 'mae'})

    scorer = {
        "rmse": sklearn.metrics.mean_squared_error,
        "mae": sklearn.metrics.mean_absolute_error
    }[mpars['metric_name']]

    mpars2 = mpars.get("metrics_pars", {})  ##Specific to score
    score_val = scorer(yval, ypred, **mpars2)

    ddict = [{"metric_val": score_val, 'metric_name': mpars['metric_name']}]

    return ddict


def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    global model, session
    ## optuna_model = model.model_pars.get('optuna_model', None)   #### NO model is saved in model.model
    post_process_fun = model.model_pars.get('post_process_fun', None)

    if post_process_fun is None:
        def post_process_fun(y):
            return y

    if Xpred is None:
        data_pars['train'] = False
        Xpred = get_dataset(data_pars, task_type="predict")

    # ypred = optuna_model.predict(Xpred, num_iteration=optuna_model.best_iteration)
    ypred = model.model.predict(Xpred,) # num_iteration=model.model.best_iteration)

    #ypred = post_process_fun(ypred)
    
    ypred_proba = None  ### No proba    
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred) 
    return ypred, ypred_proba


def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    global model, session
    import cloudpickle as pickle
    os.makedirs(path, exist_ok=True)

    filename = "model.pkl"
    pickle.dump(model, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )

    filename = "info.pkl"
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )

    filename = "model_pars.pkl"
    pickle.dump(model.model_pars, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model      = model0.model
    model.model_meta = model0.model_meta

    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd


def preprocess(prepro_pars):
    if prepro_pars['type'] == 'test':
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)

        # log(X,y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        return Xtrain, ytrain, Xtest, ytest

    if prepro_pars['type'] == 'train':
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]
        dfy = df[prepro_pars['coly']]
        Xtrain, Xtest, ytrain, ytest = train_test_split(dfX.values, dfy.values,
                                                        stratify=dfy.values,test_size=0.1)
        return Xtrain, ytrain, Xtest, ytest

    else:
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]

        Xtest, ytest = dfX, None
        return None, None, Xtest, ytest


####################################################################################################
############ Do not change #########################################################################
def get_dataset(data_pars=None, task_type="train", **kw):
    """
      "ram"  : 
      "file" :
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    if data_type == "ram":
        if task_type == "predict":
            d = data_pars[task_type]
            return d["X"]

        if task_type == "eval":
            d = data_pars[task_type]
            return d["X"], d["y"]

        if task_type == "train":
            d = data_pars[task_type]
            return d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')


def get_params_sklearn(deep=False):
    return model.model.get_params(deep=deep)


def get_params(param_pars={}, **kw):
    import json
    # from jsoncomment import JsonComment ; json = JsonComment()
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "json":
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")
