# -*- coding: utf-8 -*-
"""
activate py36 && python source/run_inference.py  run_predict  --n_sample 1000  --config_name lightgbm  --path_model /data/output/a01_test/   --path_output /data/output/a01_test_pred/     --path_data_train /data/input/train/

"""
import warnings
warnings.filterwarnings('ignore')
import sys, gc, os, pandas as pd, importlib


#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")
# import util_feature


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement Logging
    print(sjump, sspace, *s, sspace, flush=True)



from util_feature import load, load_function_uri, load_dataset
####################################################################################################
def model_dict_load(model_dict, config_path, config_name, verbose=True):
    if model_dict is None :
       log("#### Model Params Dynamic loading  ###############################################")
       model_dict_fun = load_function_uri(uri_name=config_path + "::" + config_name)
       model_dict     = model_dict_fun()   ### params
    if verbose : log( model_dict )
    return model_dict


def map_model(model_name):
    if 'optuna' in model_name:
       ##  'models.model_bayesian_pyro'   'model_widedeep'
       mod    = f'models.model_optuna'
       modelx = importlib.import_module(mod)

    else :
        ### Al SKLEARN API
        #['ElasticNet', 'ElasticNetCV', 'LGBMRegressor', 'LGBMModel', 'TweedieRegressor', 'Ridge']:
       mod    = 'models.model_sklearn'
       modelx = importlib.import_module(mod)

    return modelx


def predict(model_name, path_model, dfX, cols_family):
    """
    """
    modelx = map_model(model_name)
    modelx.reset()
    log(modelx, path_model)
    #log(os.getcwd())
    sys.path.append( root)    #### Needed due to import source error


    log("#### Load model  ############################################")
    print(path_model + "/model/model.pkl")
    # modelx.model = load(path_model + "/model//model.pkl")
    modelx.model = load(path_model + "/model.pkl")

    # stats = load(path_model + "/model/info.pkl")
    # colsX       = load(path_model + "/model/colsX.pkl")   ## column name
    colsX       = load(path_model + "/colsX.pkl")   ## column name

    # coly  = load( path_model + "/model/coly.pkl"   )
    assert colsX is not None, "cannot load colsx, " + path_model
    assert modelx.model is not None, "cannot load modelx, " + path_model
    log("#### modelx\n", modelx.model.model)

    log("### Prediction  ############################################")
    dfX1  = dfX.reindex(columns=colsX)   #reindex included

    ypred = modelx.predict(dfX1)

    return ypred


####################################################################################################
############CLI Command ############################################################################
def run_predict(config_name, config_path, n_sample=-1,
                path_data=None, path_output=None, pars={}, model_dict=None):

    model_dict = model_dict_load(model_dict, config_path, config_name, verbose=True)
    m          = model_dict['global_pars']

    model_class      = model_dict['model_pars']['model_class']
    path_data        = m['path_pred_data']   if path_data   is None else path_data
    path_pipeline    = m['path_pred_pipeline']    #   path_output + "/pipeline/" )
    path_model       = m['path_pred_model']

    path_output      = m['path_pred_output'] if path_output is None else path_output
    log(path_data, path_model, path_output)

    pars = {'cols_group': model_dict['data_pars']['cols_input_type'],
            'pipe_list' : model_dict['model_pars']['pre_process_pars']['pipe_list']}


    ##########################################################################################
    colid            = load(f'{path_pipeline}/colid.pkl')
    df               = load_dataset(path_data, path_data_y=None, colid=colid, n_sample=n_sample)

    from run_preprocess import preprocess_inference   as preprocess
    dfX, cols_family = preprocess(df, path_pipeline, preprocess_pars=pars)
    ypred, yproba    = predict(model_class, path_model, dfX, cols_family)


    log("############ Saving prediction  ###################################################" )
    log(ypred.shape, path_output)
    os.makedirs(path_output, exist_ok=True)
    df[cols_family["coly"] + "_pred"]       = ypred
    if yproba is not None :
       df[cols_family["coly"] + "_pred_proba"] = yproba
    df.to_csv(f"{path_output}/prediction.csv")
    log(df.head(8))


    log("###########  Export Specific ######################################################")
    df[cols_family["coly"]] = ypred
    df[[cols_family["coly"]]].to_csv(f"{path_output}/pred_only.csv")



def run_data_check(path_data, path_data_ref, path_model, path_output, sample_ratio=0.5):
    """
     Calcualata Dataset Shift before prediction.
    """
    from run_preprocess import preprocess_inference   as preprocess
    path_output   = root + path_output
    path_data     = root + path_data
    path_data_ref = root + path_data_ref
    path_pipeline = root + path_model + "/pipeline/"

    os.makedirs(path_output, exist_ok=True)
    colid          = load(f'{path_pipeline}/colid.pkl')

    df1                = load_dataset(path_data_ref,colid=colid)
    dfX1, cols_family1 = preprocess(df1, path_pipeline)

    df2                = load_dataset(path_data,colid=colid)
    dfX2, cols_family2 = preprocess(df2, path_pipeline)

    colsX       = cols_family1["colnum_bin"] + cols_family1["colcat_bin"]
    dfX1        = dfX1[colsX]
    dfX2        = dfX2[colsX]

    from util_feature import pd_stat_dataset_shift
    nsample     = int(min(len(dfX1), len(dfX2)) * sample_ratio)
    metrics_psi = pd_stat_dataset_shift(dfX2, dfX1,
                                        colsX, nsample=nsample, buckets=7, axis=0)
    metrics_psi.to_csv(f"{path_output}/prediction_features_metrics.csv")
    log(metrics_psi)



###########################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()