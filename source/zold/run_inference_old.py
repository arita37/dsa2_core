# -*- coding: utf-8 -*- 
"""

 ! activate py36 && python source/run_inference.py  run_predict  --n_sample 1000  --config_name lightgbm  --path_model /data/output/a01_test/   --path_output /data/output/a01_test_pred/     --path_data_train /data/input/train/
 

"""
import warnings
warnings.filterwarnings('ignore')
import sys
import gc
import os
import pandas as pd
import importlib

#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")
import util_feature


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement Logging
    print(sjump, sspace, s, sspace, flush=True)



from util_feature import load, load_function_uri
from util_feature import  load_dataset
####################################################################################################
####################################################################################################

def preprocess(df, path_pipeline="data/pipeline/pipe_01/", preprocess_pars={}):
    """
      FUNCTIONNAL approach is used for pre-processing, so the code can be EASILY extensible to PYSPPARK.
      PYSPARK  supports better UDF, lambda function
    """
    from util_feature import (pd_colnum_tocat, pd_col_to_onehot, pd_colcat_toint,
                              pd_feature_generate_cross)

    log("########### Load column by column type ##################################")
    colid          = load(f'{path_pipeline}/colid.pkl')
    coly           = load(f'{path_pipeline}/coly.pkl')
    colcat         = load(f'{path_pipeline}/colcat.pkl')
    colcat_onehot  = load(f'{path_pipeline}/colcat_onehot.pkl')
    colcat_bin_map = load(f'{path_pipeline}/colcat_bin_map.pkl')

    colnum         = load(f'{path_pipeline}/colnum.pkl')
    colnum_binmap  = load(f'{path_pipeline}/colnum_binmap.pkl')
    colnum_onehot  = load(f'{path_pipeline}/colnum_onehot.pkl')

    ### OneHot column selected for cross features
    colcross_single_onehot_select = load(f'{path_pipeline}/colcross_single_onehot_select.pkl')

    pipe_default    = [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]
    pipe_list       = preprocess_pars.get('pipe_list', pipe_default)


    if "dfcat_bin" in pipe_list :
        log("###### Colcat as integer encoded  ####################################")
        dfcat_bin, _ = pd_colcat_toint(df[colcat],  colname=colcat,
                                                     colcat_map=colcat_bin_map, suffix="_int")
        colcat_bin = list(dfcat_bin.columns)

    if "dfcat_hot" in pipe_list :
       log("###### Colcat to onehot ###############################################")
       dfcat_hot, _ = pd_col_to_onehot(df[colcat],  colname=colcat,
                                                  colonehot=colcat_onehot, return_val="dataframe,param")
       log(dfcat_hot[colcat_onehot].head(5))


    if "dfnum_bin" in pipe_list :
        log("###### Colnum Preprocess   ###########################################")
        dfnum_bin, _ = pd_colnum_tocat(df, colname=colnum, colexclude=None,
                                         colbinmap=colnum_binmap,
                                         bins=-1, suffix="_bin", method="",
                                         return_val="dataframe,param")
        log(colnum_binmap)
        colnum_bin = [x + "_bin" for x in list(colnum_binmap.keys())]
        log(dfnum_bin[colnum_bin].head(5))


    if "dfnum_hot" in pipe_list :
        ###### Map numerics bin to One Hot
        dfnum_hot, _ = pd_col_to_onehot(dfnum_bin[colnum_bin], colname=colnum_bin,
                                         colonehot=colnum_onehot, return_val="dataframe,param")
        log(dfnum_hot[colnum_onehot].head(5))

    print('------------dfcat_hot---------------------', dfcat_hot)
    print('------------dfnum_hot---------------------', dfnum_hot)
    print('------------colcross_single_onehot_select---------------------', colcross_single_onehot_select)
    if "dfcross_hot" in pipe_list :
        log("####### colcross cross features   ###################################################")
        dfcross_hot = pd.DataFrame()
        if colcross_single_onehot_select is not None :
            df_onehot = dfcat_hot.join(dfnum_hot, on=colid, how='left')

            # colcat_onehot2 = [x for x in colcat_onehot if 'companyId' not in x]
            # log(colcat_onehot2)
            # colcross_single = colnum_onehot + colcat_onehot2
            df_onehot = df_onehot[colcross_single_onehot_select]
            dfcross_hot, colcross_pair = pd_feature_generate_cross(df_onehot, colcross_single_onehot_select,
                                                                    pct_threshold=0.02,
                                                                    m_combination=2)
            log(dfcross_hot.head(2).T)
            colcross_onehot = list(dfcross_hot.columns)
            del df_onehot ;    gc.collect()


    log("##### Merge data type together  :   #######################3############################ ")
    dfX = df[ colnum + colcat ]
    for t in [ 'dfnum_bin', 'dfnum_hot', 'dfcat_bin', 'dfcat_hot', 'dfcross_hot',   ] :
        if t in locals() :
           dfX = pd.concat((dfX, locals()[t] ), axis=1)
           # log(t, list(dfX.columns))

    colX = list(dfX.columns)
    #colX.remove(coly)
    del df ;    gc.collect()


    log("###### Export columns group   ##########################################################")
    cols_family = {}
    for t in ['colid','coly', #added 'coly'
              "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
              "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
              'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
              'colsX', 'coly'
              ]:
        t_val = locals().get(t, None)
        if t_val is not None :
           cols_family[t] = t_val


    return dfX, cols_family


####################################################################################################
####################################################################################################
def map_model(model_name):
    try :
       ##  'models.model_bayesian_pyro'   'model_widedeep'
       mod    = f'models.{model_name}'
       modelx = importlib.import_module(mod) 
       
    except :
        ### Al SKLEARN API
        #['ElasticNet', 'ElasticNetCV', 'LGBMRegressor', 'LGBMModel', 'TweedieRegressor', 'Ridge']:
       mod    = 'models.model_sklearn'
       modelx = importlib.import_module(mod) 
    
    return modelx


def predict(model_name, path_model, dfX, cols_family):
    """
    if config_name in ['ElasticNet', 'ElasticNetCV', 'LGBMRegressor', 'LGBMModel', 'TweedieRegressor', 'Ridge']:
        from models import model_sklearn as modelx

    elif config_name == 'model_bayesian_pyro':
        from models import model_bayesian_pyro as modelx

    elif config_name == 'model_widedeep':
        from models import model_widedeep as modelx
    """
    modelx = map_model(model_name)    
    modelx.reset()
    log(modelx, path_model)    
    #log(os.getcwd())
    sys.path.append( root)    #### Needed due to import source error    
    

    modelx.model = load(path_model + "/model/model.pkl")
    # stats = load(path_model + "/model/info.pkl")
    colsX       = load(path_model + "/model/colsX.pkl")   ## column name
    # coly  = load( path_model + "/model/coly.pkl"   )
    assert colsX is not None
    assert modelx.model is not None

    log(modelx.model.model)

    ### Prediction
    dfX1=dfX.reindex(columns=colsX)   #reindex included
    ypred = modelx.predict(dfX1)

    return ypred




####################################################################################################
############CLI Command ############################################################################
def run_predict(model_name, path_model, path_data, path_output, n_sample=-1):
    path_output   = root + path_output
    path_data     = root + path_data + "/features.zip"#.zip
    path_model    = root + path_model
    path_pipeline = path_model + "/pipeline/"
    path_test_X = path_data + "/features.zip"   #.zip #added path to testing features
    log(path_data, path_model, path_output)

    colid            = load(f'{path_pipeline}/colid.pkl')

    df               = load_dataset(path_data, path_data_y=None, colid=colid, n_sample=n_sample)
  
    dfX, cols_family = preprocess(df, path_pipeline)
    
    ypred, yproba    = predict(model_name, path_model, dfX, cols_family)


    log("Saving prediction", ypred.shape, path_output)
    os.makedirs(path_output, exist_ok=True)
    df[cols_family["coly"] + "_pred"]       = ypred
    if yproba is not None :
       df[cols_family["coly"] + "_pred_proba"] = yproba
    df.to_csv(f"{path_output}/prediction.csv")
    log(df.head(8))

    #####  Export Specific
    df[cols_family["coly"]] = ypred
    df[[cols_family["coly"]]].to_csv(f"{path_output}/pred_only.csv")



def run_check(path_data, path_data_ref, path_model, path_output, sample_ratio=0.5):
    """
     Calcualata Dataset Shift before prediction.
    """
    path_output   = root + path_output
    path_data     = root + path_data
    path_data_ref = root + path_data_ref
    path_pipeline = root + path_model + "/pipeline/"

    os.makedirs(path_output, exist_ok=True)

    colid          = load(f'{path_pipeline}/colid.pkl')

    df1 = load_dataset(path_data_ref,colid=colid)
    dfX1, cols_family1 = preprocess(df1, path_pipeline)

    df2 = load_dataset(path_data,colid=colid)
    dfX2, cols_family2 = preprocess(df2, path_pipeline)

    colsX       = cols_family1["colnum_bin"] + cols_family1["colcat_bin"]
    dfX1        = dfX1[colsX]
    dfX2        = dfX2[colsX]

    nsample     = int(min(len(dfX1), len(dfX2)) * sample_ratio)
    metrics_psi = util_feature.pd_stat_dataset_shift(dfX2, dfX1,
                                                     colsX, nsample=nsample, buckets=7, axis=0)
    metrics_psi.to_csv(f"{path_output}/prediction_features_metrics.csv")
    log(metrics_psi)



###########################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()
