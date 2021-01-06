# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
https://github.com/Automunge/AutoMunge#library-of-transformations
Library of Transformations
Library of Transformations Subheadings:
Intro
Numerical Set Normalizations
Numerical Set Transformations
Numercial Set Bins and Grainings
Sequential Numerical Set Transformations
Categorical Set Encodings
Date-Time Data Normalizations
Date-Time Data Bins
Differential Privacy Noise Injections
Misc. Functions
String Parsing
More Efficient String Parsing
Multi-tier String Parsing
List of Root Categories
List of Suffix Appenders
Other Reserved Strings
Root Category Family Tree Definitions



"""
import warnings
warnings.filterwarnings('ignore')
import sys, gc, os, pandas as pd, json, copy
import numpy as np

####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


#### Debuging state (Ture/False)
DEBUG_=True

####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)

def logs(*s):
    if DEBUG_:
        print(*s, flush=True)


def log_pd(df, *s, n=0, m=1):
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump,  df.head(n), flush=True)


from util_feature import  save, load_function_uri, load
import util_feature
####################################################################################################
####################################################################################################
def save_features(df, name, path):
    """
    :param df:
    :param name:
    :param path:
    :return:
    """
    if path is not None :
       os.makedirs( f"{path}/{name}" , exist_ok=True)
       if isinstance(df, pd.Series):
           df0=df.to_frame()
       else:
           df0=df
       df0.to_parquet( f"{path}/{name}/features.parquet")




####################################################################################################
def coltext_stopwords(text, stopwords=None, sep=" "):
    tokens = text.split(sep)
    tokens = [t.strip() for t in tokens if t.strip() not in stopwords]
    return " ".join(tokens)


def pd_coltext_clean( df, col, stopwords= None , pars=None):
    import string, re
    ntoken= pars.get('n_token', 1)
    df      = df.fillna("")
    dftext = df
    log(dftext)
    log(col)
    list1 = col
    # list1 = []
    # list1.append(col)
    # fromword = [ r"\b({w})\b".format(w=w)  for w in fromword    ]
    # print(fromword)
    for col_n in list1:
        dftext[col_n] = dftext[col_n].fillna("")
        dftext[col_n] = dftext[col_n].str.lower()
        dftext[col_n] = dftext[col_n].apply(lambda x: x.translate(string.punctuation))
        dftext[col_n] = dftext[col_n].apply(lambda x: x.translate(string.digits))
        dftext[col_n] = dftext[col_n].apply(lambda x: re.sub("[!@,#$+%*:()'-]", " ", x))
        dftext[col_n] = dftext[col_n].apply(lambda x: coltext_stopwords(x, stopwords=stopwords))
    return dftext



def pd_coltext_wordfreq(df, col, stopwords, ntoken=100):
    """
    :param df:
    :param coltext:  text where word frequency should be extracted
    :param nb_to_show:
    :return:
    """
    sep=" "
    logs('----col-----\n', col)
    coltext_freq = df[col].apply(str).apply(lambda x: pd.value_counts(x.split(sep))).sum(axis=0).reset_index()
    coltext_freq.columns = ["word", "freq"]
    coltext_freq = coltext_freq.sort_values("freq", ascending=0)
    log(coltext_freq)

    word_tokeep  = coltext_freq["word"].values[:ntoken]
    word_tokeep  = [  t for t in word_tokeep if t not in stopwords   ]

    return coltext_freq, word_tokeep


def nlp_get_stopwords():
    import json
    import string
    stopwords = json.load(open("source/utils/stopwords_en.json") )["word"]
    stopwords = [ t for t in string.punctuation ] + stopwords
    stopwords = [ "", " ", ",", ".", "-", "*", '€', "+", "/" ] + stopwords
    stopwords =list(set( stopwords ))
    stopwords.sort()
    print( stopwords )
    stopwords = set(stopwords)
    return stopwords


def pd_coltext(df, col, pars={}):
    """
    df : Datframe
    col : list of columns
    pars : dict of pars

    """
    from utils import util_text, util_model

    #### Load pars ###################################################################
    path_pipeline        = pars.get('path_pipeline', None)
    word_tokeep_dict_all = load(  path_pipeline + "/word_tokeep_dict_all.pkl" )  if path_pipeline is not None else {}
    # dftext_tdidf_all = load(f'{path_pipeline}/dftext_tdidf.pkl') if  path_pipeline else None
    # dftext_svd_list_all      = load(f'{path_pipeline}/dftext_svd.pkl')   if  path_pipeline else None
    dimpca       = pars.get('dimpca', 2)
    word_minfreq = pars.get('word_minfreq', 3)

    #### Process  ####################################################################
    stopwords           = nlp_get_stopwords()
    dftext              = pd_coltext_clean(df, col, stopwords= stopwords , pars=pars)
    dftext_svd_list_all = None
    dftext_tdidf_all    = None

    ### Processing each of text columns to create a bag of word/to load the bag of word -> tf-idf -> svd
    for col_ in col:

            if path_pipeline is not None:
                ### If it is in Inference step, use the saved bag of word for the column `col_`
                word_tokeep = word_tokeep_dict_all[col_]

            else:
                ### If it is not, create a bag of word
                coltext_freq, word_tokeep = pd_coltext_wordfreq(df, col_, stopwords, ntoken=100)  ## nb of words to keep
                word_tokeep_dict_all[col_] = word_tokeep  ## save the bag of wrod for `col_` in a dict

            dftext_tdidf_dict, word_tokeep_dict = util_text.pd_coltext_tdidf(dftext, coltext=col_, word_minfreq= word_minfreq,
                                                                             word_tokeep = word_tokeep,
                                                                             return_val  = "dataframe,param")

            dftext_tdidf_all = pd.DataFrame(dftext_tdidf_dict) if dftext_tdidf_all is None else pd.concat((dftext_tdidf_all,pd.DataFrame(dftext_tdidf_dict)),axis=1)
            log(word_tokeep_dict)

            ###  Dimesnion reduction for Sparse Matrix
            dftext_svd_list, svd_list = util_model.pd_dim_reduction(dftext_tdidf_dict,
                                                           colname        = None,
                                                           model_pretrain = None,
                                                           colprefix      = col_ + "_svd",
                                                           method         = "svd",  dimpca=dimpca,  return_val="dataframe,param")

            dftext_svd_list_all = dftext_svd_list if dftext_svd_list_all is None else pd.concat((dftext_svd_list_all,dftext_svd_list),axis=1)
    #################################################################################

    ###### Save and Export ##########################################################
    if 'path_features_store' in pars:
            save_features(dftext_svd_list_all, 'dftext_svd' + "-" + str(col), pars['path_features_store'])
            # save(dftext_svd_list_all,  pars['path_pipeline_export'] + "/dftext_svd.pkl")
            # save(dftext_tdidf_all,     pars['path_pipeline_export'] + "/dftext_tdidf.pkl" )
            save(word_tokeep_dict_all,     pars['path_pipeline_export'] + "/word_tokeep_dict_all.pkl" )

    col_pars = {}
    col_pars['cols_new'] = {
     # 'coltext_tdidf'    : dftext_tdidf_all.columns.tolist(),       ### list
     'coltext_svd'      : dftext_svd_list_all.columns.tolist()      ### list
    }

    dftext_svd_list_all.index = dftext.index
    # return pd.concat((dftext_svd_list_all,dftext_svd_list_all),axis=1), col_pars
    return dftext_svd_list_all, col_pars



##### Filtering / cleaning rows :   #########################################################
def pd_filter_rows(df, col, pars):
    import re
    coly = col
    filter_pars =  pars
    def isfloat(x):
        #x = re.sub("[!@,#$+%*:()'-]", "", str(x))
        try :
            a= float(x)
            return 1
        except:
            return 0

    ymin, ymax = pars.get('ymin', -9999999999.0), filter_pars.get('ymax', 999999999.0)

    df['_isfloat'] = df[ coly ].apply(lambda x : isfloat(x),axis=1 )
    df = df[ df['_isfloat'] > 0 ]
    df = df[df[coly] > ymin]
    df = df[df[coly] < ymax]
    del df['_isfloat']

    return df, col


def pd_filter_resample(df=None, col=None, pars=None):
    """
        Over-sample, Under-sample
    """
    prefix = 'col_imbalance'
    ######################################################################################
    from imblearn.over_sampling import SMOTE

    model_resample = { 'SMOTE' : SMOTE}[  pars.get("model_name", 'SMOTE') ]

    pars_resample  = pars.get('pars_resample',
                             {'sampling_strategy' : 'auto', 'random_state':0, 'k_neighbors':5, 'n_jobs': 2})

    if 'path_pipeline' in pars :   #### Inference time
        return df, {'col_new': col }
        #gp   = load(pars['path_pipeline'] + f"/{prefix}_model.pkl" )
        #pars = load(pars['path_pipeline'] + f"/{prefix}_pars.pkl" )

    else :     ### Training time
        colX          = col # [col_ for col_ in col if col_ not in coly]
        train_X       = df[colX].fillna(method='ffill')
        coly     = pars['coly']
        train_y  = pars['dfy']
        gp       = model_resample( **pars_resample)
        X_resample, y_resample = gp.fit_resample(train_X, train_y)

        df2       = pd.DataFrame(X_resample, columns = col, index=train_X.index)
        df2[coly] = y_resample


    col_new = col
    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(df2, 'df_resample', pars['path_features_store'])
       save(gp,             pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )
       save(col,            pars['path_pipeline_export'] + f"/{prefix}.pkl" )
       save(pars_resample,   pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )


    col_pars = {'prefix' : prefix , 'path' :   pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
    col_pars['cols_new'] = {
       prefix :  col_new  ### list
    }
    return df2, col_pars





##### Label processing   ##################################################################
def pd_label_clean(df, col, pars):
    path_features_store = pars['path_features_store']
    # path_pipeline_export = pars['path_pipeline_export']
    coly = col=[0]
    y_norm_fun = None
    # Target coly processing, Normalization process  , customize by model
    log("y_norm_fun preprocess_pars")
    y_norm_fun = pars.get('y_norm_fun', None)
    if y_norm_fun is not None:
        df[coly] = df[coly].apply(lambda x: y_norm_fun(x))
        # save(y_norm_fun, f'{path_pipeline_export}/y_norm.pkl' )
        save_features(df[coly], 'dfy', path_features_store)
    return df,coly


def pd_coly(df, col, pars):
    ##### Filtering / cleaning rows :   #########################################################
    coly=col
    def isfloat(x):
        try :
            a= float(x)
            return 1
        except:
            return 0
    df['_isfloat'] = df[ coly ].apply(lambda x : isfloat(x) )
    df             = df[ df['_isfloat'] > 0 ]
    df[coly] = df[coly].astype('float32')
    del df['_isfloat']
    logs("----------df[coly]------------",df[coly])
    ymin, ymax = pars.get('ymin', -9999999999.0), pars.get('ymax', 999999999.0)
    df = df[df[coly] > ymin]
    df = df[df[coly] < ymax]

    ##### Label processing   ####################################################################
    y_norm_fun = None
    # Target coly processing, Normalization process  , customize by model
    log("y_norm_fun preprocess_pars")
    y_norm_fun = pars.get('y_norm_fun', None)
    if y_norm_fun is not None:
        df[coly] = df[coly].apply(lambda x: y_norm_fun(x))
        # save(y_norm_fun, f'{path_pipeline_export}/y_norm.pkl' )

    if pars.get('path_features_store', None) is not None:
        path_features_store = pars['path_features_store']
        save_features(df[coly], 'dfy', path_features_store)

    return df,col


def pd_colnum(df, col, pars):
    colnum = col
    for x in colnum:
        df[x] = df[x].astype("float32")
    log(df.dtypes)


def pd_colnum_normalize(df, col, pars):
    log("### colnum normalize  ###############################################################")
    from util_feature import pd_colnum_normalize
    colnum = col

    pars = { 'pipe_list': [ {'name': 'fillna', 'naval' : 0.0 }, {'name': 'minmax'} ]}
    dfnum_norm, colnum_norm = pd_colnum_normalize(df, colname=colnum,  pars=pars, suffix = "_norm",
                                                  return_val="dataframe,param")
    log(colnum_norm)
    if pars.get('path_features_store', None) is not None:
        path_features_store = pars['path_features_store']
        save_features(dfnum_norm, 'dfnum_norm', path_features_store)
    return dfnum_norm, colnum_norm





def pd_colnum_quantile_norm(df, col, pars={}):
  """
     colnum normalization by quantile
  """
  prefix  = "colnum_quantile_norm"
  df      = df[col]
  num_col = col

  ##### Grab previous computed params  ################################################
  pars2 = {}
  if  'path_pipeline' in pars :   #### Load existing column list
       colnum_quantile_norm = load( pars['path_pipeline']  +f'/{prefix}.pkl')
       model                = load( pars['path_pipeline']  +f'/{prefix}_model.pkl')
       pars2                = load( pars['path_pipeline']  +f'/{prefix}_pars.pkl')

  lower_bound_sparse = pars2.get('lower_bound_sparse', None)
  upper_bound_sparse = pars2.get('upper_bound_sparse', None)
  lower_bound        = pars2.get('lower_bound_sparse', None)
  upper_bound        = pars2.get('upper_bound_sparse', None)
  sparse_col         = pars2.get('colsparse', ['capital-gain', 'capital-loss'] )

  ####### Find IQR and implement to numericals and sparse columns seperately ##########
  Q1  = df.quantile(0.25)
  Q3  = df.quantile(0.75)
  IQR = Q3 - Q1

  for col in num_col:
    if col in sparse_col:
      df_nosparse = pd.DataFrame(df[df[col] != df[col].mode()[0]][col])

      if lower_bound_sparse is not None:
        pass

      elif df_nosparse[col].quantile(0.25) < df[col].mode()[0]: #Unexpected case
        lower_bound_sparse = df_nosparse[col].quantile(0.25)

      else:
        lower_bound_sparse = df[col].mode()[0]

      if upper_bound_sparse is not None:
        pass

      elif df_nosparse[col].quantile(0.75) < df[col].mode()[0]: #Unexpected case
        upper_bound_sparse = df[col].mode()[0]

      else:
        upper_bound_sparse = df_nosparse[col].quantile(0.75)

      n_outliers = len(df[(df[col] < lower_bound_sparse) | (df[col] > upper_bound_sparse)][col])

      if n_outliers > 0:
        df.loc[df[col] < lower_bound_sparse, col] = lower_bound_sparse * 0.75 #--> MAIN DF CHANGED
        df.loc[df[col] > upper_bound_sparse, col] = upper_bound_sparse * 1.25 # --> MAIN DF CHANGED

    else:
      if lower_bound is None or upper_bound is None :
         lower_bound = df[col].quantile(0.25) - 1.5 * IQR[col]
         upper_bound = df[col].quantile(0.75) + 1.5 * IQR[col]

      df[col] = np.where(df[col] > upper_bound, 1.25 * upper_bound, df[col])
      df[col] = np.where(df[col] < lower_bound, 0.75 * lower_bound, df[col])

  df.columns = [ t + "_qt_norm" for t in df.columns ]
  pars_new   = {'lower_bound' : lower_bound, 'upper_bound': upper_bound,
                'lower_bound_sparse' : lower_bound_sparse, 'upper_bound_sparse' : upper_bound_sparse  }
  dfnew    = df
  model    = None
  colnew   = list(df.columns)

  ##### Export ##############################################################################
  if 'path_features_store' in pars and 'path_pipeline_export' in pars:
      save_features(df,  prefix, pars['path_features_store'])
      save(colnew,     pars['path_pipeline_export']  + f"/{prefix}.pkl" )
      save(pars_new,   pars['path_pipeline_export']  + f"/{prefix}_pars.pkl" )
      save(model,      pars['path_pipeline_export']  + f"/{prefix}_model.pkl" )


  col_pars = {'prefix' : prefix, 'path': pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
  col_pars['cols_new'] = {
    prefix :  colnew  ### list
  }
  return dfnew,  col_pars





def pd_colnum_bin(df, col, pars):
    from util_feature import  pd_colnum_tocat

    path_pipeline = pars.get('path_pipeline', False)
    colnum_binmap  = load(f'{path_pipeline}/colnum_binmap.pkl') if  path_pipeline else None
    log(colnum_binmap)

    colnum = col

    log("### colnum Map numerics to Category bin  ###########################################")
    dfnum_bin, colnum_binmap = pd_colnum_tocat(df, colname=colnum, colexclude=None, colbinmap=colnum_binmap,
                                               bins=10, suffix="_bin", method="uniform",
                                               return_val="dataframe,param")
    log(colnum_binmap)
    ### Renaming colunm_bin with suffix
    colnum_bin = [x + "_bin" for x in list(colnum_binmap.keys())]
    log(colnum_bin)

    if 'path_features_store' in pars:
        scol = "_".join(col[:5])
        save_features(dfnum_bin, 'colnum_bin' + "-" + scol, pars['path_features_store'])
        save(colnum_binmap,  pars['path_pipeline_export'] + "/colnum_binmap.pkl" )
        save(colnum_bin,     pars['path_pipeline_export'] + "/colnum_bin.pkl" )


    col_pars = {}
    col_pars['colnumbin_map'] = colnum_binmap
    col_pars['cols_new'] = {
     'colnum'     :  col ,    ###list
     'colnum_bin' :  colnum_bin       ### list
    }
    return dfnum_bin, col_pars


def pd_colnum_binto_onehot(df, col=None, pars=None):
    assert isinstance(col, list) and isinstance(df, pd.DataFrame)

    dfnum_bin = df[col]
    colnum_bin = col

    path_pipeline = pars.get('path_pipeline', False)
    colnum_onehot = load(f'{path_pipeline}/colnum_onehot.pkl') if  path_pipeline else None


    log("###### colnum bin to One Hot  #################################################")
    from util_feature import  pd_col_to_onehot
    dfnum_hot, colnum_onehot = pd_col_to_onehot(dfnum_bin[colnum_bin], colname=colnum_bin,
                                                colonehot=colnum_onehot, return_val="dataframe,param")
    log(colnum_onehot)

    if 'path_features_store' in pars :
        save_features(dfnum_hot, 'colnum_onehot', pars['path_features_store'])
        save(colnum_onehot,  pars['path_pipeline_export'] + "/colnum_onehot.pkl" )

    col_pars = {}
    col_pars['colnum_onehot'] = colnum_onehot
    col_pars['cols_new'] = {
     # 'colnum'        :  col ,    ###list
     'colnum_onehot' :  colnum_onehot       ### list
    }
    return dfnum_hot, col_pars



def pd_colcat_to_onehot(df, col=None, pars=None):
    """

    """
    log("#### colcat to onehot")
    if len(col)==1:
        colnew       = [col[0] + "_onehot"]
        df[ colnew ] =  df[col]
        col_pars     = {}
        col_pars['colcat_onehot'] = colnew
        col_pars['cols_new'] = {
         # 'colnum'        :  col ,    ###list
         'colcat_onehot'   :  colnew      ### list
        }
        return df[colnew], col_pars

    colcat_onehot = None
    if  'path_pipeline' in pars :
       colcat_onehot = load( pars['path_pipeline'] + '/colcat_onehot.pkl')

    ######################################################################################
    colcat = col
    dfcat_hot, colcat_onehot = util_feature.pd_col_to_onehot(df[colcat], colname=colcat,
                                                colonehot=colcat_onehot, return_val="dataframe,param")
    log(dfcat_hot[colcat_onehot].head(5))

    ######################################################################################
    if 'path_features_store' in pars :
        save_features(dfcat_hot, 'colcat_onehot', pars['path_features_store'])
        save(colcat_onehot,  pars['path_pipeline_export'] + "/colcat_onehot.pkl" )
        save(colcat,         pars['path_pipeline_export'] + "/colcat.pkl" )

    col_pars = {}
    col_pars['colcat_onehot'] = colcat_onehot
    col_pars['cols_new'] = {
     # 'colnum'        :  col ,    ###list
     'colcat_onehot' :  colcat_onehot       ### list
    }

    print("ok ------------")
    return dfcat_hot, col_pars



def pd_colcat_bin(df, col=None, pars=None):
    # dfbum_bin = df[col]
    path_pipeline = pars.get('path_pipeline', False)
    colcat_bin_map = load(f'{path_pipeline}/colcat_bin_map.pkl') if  path_pipeline else None

    colcat = col
    log("#### Colcat to integer encoding ")
    dfcat_bin, colcat_bin_map = util_feature.pd_colcat_toint(df[colcat], colname=colcat,
                                                            colcat_map=  colcat_bin_map ,
                                                            suffix="_int")
    colcat_bin = list(dfcat_bin.columns)
    ##### Colcat processing   ################################################################
    colcat_map = util_feature.pd_colcat_mapping(df, colcat)
    log(df[colcat].dtypes, colcat_map)


    if 'path_features_store' in pars :
       save_features(dfcat_bin, 'dfcat_bin', pars['path_features_store'])
       save(colcat_bin_map,  pars['path_pipeline_export'] + "/colcat_bin_map.pkl" )
       save(colcat_bin,      pars['path_pipeline_export'] + "/colcat_bin.pkl" )


    col_pars = {}
    col_pars['colcat_bin_map'] = colcat_bin_map
    col_pars['cols_new'] = {
     'colcat'     :  col ,    ###list
     'colcat_bin' :  colcat_bin       ### list
    }

    return dfcat_bin, col_pars



def pd_colcross(df, col, pars):
    """


    """
    prefix = 'colcross_onehot'
    log("#####  Cross Features From OneHot Features   ######################################")
    from util_feature import pd_feature_generate_cross

    dfcat_hot = pars['dfcat_hot']
    dfnum_hot = pars['dfnum_hot']
    colid     = pars['colid']

    try :
       df_onehot = dfcat_hot.join(dfnum_hot, on=colid, how='left')
    except :
       df_onehot = copy.deepcopy(dfcat_hot)

    colcross_single = pars['colcross_single']
    pars_model      = { 'pct_threshold' :0.02,  'm_combination': 2 }
    if  'path_pipeline' in pars :   #### Load existing column list
       colcross_single = load( pars['path_pipeline']  + f'/{prefix}_select.pkl')
       # pars_model      = load( pars['path_pipeline']  + f'/{prefix}_pars.pkl')

    colcross_single_onehot_select = []  ## Select existing columns
    for t in list(df_onehot.columns):
       for c1 in colcross_single:
           if c1 in t:
               colcross_single_onehot_select.append(t)

    df_onehot = df_onehot[colcross_single_onehot_select ]
    dfcross_hot, colcross_pair = pd_feature_generate_cross(df_onehot, colcross_single_onehot_select,
                                                           **pars_model)
    log(dfcross_hot.head(2).T)
    colcross_pair_onehot = list(dfcross_hot.columns)

    model = None
    ##############################################################################
    if 'path_features_store' in pars:
        save_features(dfcross_hot, 'colcross_onehot', pars['path_features_store'])
        save(colcross_single_onehot_select, pars['path_pipeline_export'] + f'/{prefix}_select.pkl')
        save(colcross_pair,                 pars['path_pipeline_export'] + f'/{prefix}_stats.pkl')
        save(colcross_pair_onehot,          pars['path_pipeline_export'] + f'/{prefix}_pair.pkl')
        save(model,                         pars['path_pipeline_export'] + f'/{prefix}_pars.pkl')

    col_pars = {'model': model, 'stats' : colcross_pair }
    col_pars['cols_new'] = {
     # 'colcross_single'     :  col ,    ###list
     'colcross_pair' :  colcross_pair_onehot       ### list
    }
    return dfcross_hot, col_pars



def pd_coldate(df, col, pars):
    log("##### Coldate processing   ##########################################")
    from utils import util_date
    coldate = col
    dfdate  = None
    for coldate_i in coldate :
        dfdate_i = util_date.pd_datestring_split( df[[coldate_i]] , coldate_i, fmt="auto", return_val= "split" )
        dfdate   = pd.concat((dfdate, dfdate_i),axis=1)  if dfdate is not None else dfdate_i
        # if 'path_features_store' in pars :
        #    path_features_store = pars['path_features_store']
        #    #save_features(dfdate_i, 'dfdate_' + coldate_i, path_features_store)

    if 'path_features_store' in pars :
        save_features(dfdate, 'dfdate', pars['path_features_store'])

    col_pars = {}
    col_pars['cols_new'] = {
        # 'colcross_single'     :  col ,    ###list
        'dfdate': list(dfdate.columns)  ### list
    }
    return dfdate, col_pars


def pd_autoencoder(df, col, pars):
    """"
    (4) Autoencoder
An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore noise.

(i) Feed Forward

The simplest form of an autoencoder is a feedforward, non-recurrent neural network similar to single layer perceptrons that participate in multilayer perceptrons

from sklearn.preprocessing import minmax_scale
import tensorflow as tf
import numpy as np

def encoder_dataset(df, drop=None, dimesions=20):

  if drop:
    train_scaled = minmax_scale(df.drop(drop,axis=1).values, axis = 0)
  else:
    train_scaled = minmax_scale(df.values, axis = 0)

  # define the number of encoding dimensions
  encoding_dim = dimesions
  # define the number of features
  ncol = train_scaled.shape[1]
  input_dim = tf.keras.Input(shape = (ncol, ))

  # Encoder Layers
  encoded1 = tf.keras.layers.Dense(3000, activation = 'relu')(input_dim)
  encoded2 = tf.keras.layers.Dense(2750, activation = 'relu')(encoded1)
  encoded3 = tf.keras.layers.Dense(2500, activation = 'relu')(encoded2)
  encoded4 = tf.keras.layers.Dense(750, activation = 'relu')(encoded3)
  encoded5 = tf.keras.layers.Dense(500, activation = 'relu')(encoded4)
  encoded6 = tf.keras.layers.Dense(250, activation = 'relu')(encoded5)
  encoded7 = tf.keras.layers.Dense(encoding_dim, activation = 'relu')(encoded6)

  encoder = tf.keras.Model(inputs = input_dim, outputs = encoded7)
  encoded_input = tf.keras.Input(shape = (encoding_dim, ))

  encoded_train = pd.DataFrame(encoder.predict(train_scaled),index=df.index)
  encoded_train = encoded_train.add_prefix('encoded_')
  if drop:
    encoded_train = pd.concat((df[drop],encoded_train),axis=1)

  return encoded_train

df_out = mapper.encoder_dataset(df.copy(), ["Close_1"], 15); df_out.head()

    """
    pass


def pd_colcat_encoder_generic(df, col, pars):
    """
        Create a Class or decorator
        https://pypi.org/project/category-encoders/
        encoder = ce.BackwardDifferenceEncoder(cols=[...])
        encoder = ce.BaseNEncoder(cols=[...])
        encoder = ce.BinaryEncoder(cols=[...])
        encoder = ce.CatBoostEncoder(cols=[...])
        encoder = ce.CountEncoder(cols=[...])
        encoder = ce.GLMMEncoder(cols=[...])
        encoder = ce.HashingEncoder(cols=[...])
        encoder = ce.HelmertEncoder(cols=[...])
        encoder = ce.JamesSteinEncoder(cols=[...])
        encoder = ce.LeaveOneOutEncoder(cols=[...])
        encoder = ce.MEstimateEncoder(cols=[...])
        encoder = ce.OneHotEncoder(cols=[...])
        encoder = ce.OrdinalEncoder(cols=[...])
        encoder = ce.SumEncoder(cols=[...])
        encoder = ce.PolynomialEncoder(cols=[...])
        encoder = ce.TargetEncoder(cols=[...])
        encoder = ce.WOEEncoder(cols=[...])
    """
    prefix     = "colcat_encoder_generic"
    pars_model = None
    if 'path_pipeline' in  pars  :   ### Load during Inference
       colcat_encoder = load( pars['path_pipeline'] + f"/{prefix}.pkl" )
       pars_model     = load( pars['path_pipeline'] + f"/{prefix}_pars.pkl" )
       #model         = load( pars['path_pipeline'] + f"/{prefix}_model.pkl" )

    ####### Custom Code ###############################################################
    from category_encoders import HashingEncoder, WOEEncoder
    pars_model         = pars.get('model_pars', {})  if pars_model is None else pars_model
    pars_model['cols'] = col
    model_name         = pars.get('model_name', 'HashingEncoder')

    model_class        = { 'HashingEncoder' : HashingEncoder  }[model_name]
    model              = model_class(**pars_model)
    dfcat_encoder      = model.fit_transform(df[col])

    dfcat_encoder.columns = [t + "_cod" for t in dfcat_encoder.columns ]
    colcat_encoder        = list(dfcat_encoder.columns)


    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(dfcat_encoder, 'dfcat_encoder', pars['path_features_store'])
       save(model,           pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )
       save(pars_model,      pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )
       save(colcat_encoder,  pars['path_pipeline_export'] + f"/{prefix}.pkl" )

    col_pars = { 'prefix' : prefix,  'path' :   pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
    col_pars['cols_new'] = {
     'colcat_encoder_generic' :  colcat_encoder  ### list
    }
    return dfcat_encoder, col_pars



def pd_colcat_minhash(df, col, pars):
    """
       MinHash Algo for category
       https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087

    """
    prefix = 'colcat_minhash'
    colcat              = col

    pars_minhash = {'n_component' : [4, 2], 'model_pretrain_dict' : None,}
    if 'path_pipeline_export' in pars :
        try :
            pars_minhash = load( pars['path_pipeline_export'] + '/colcat_minhash_pars.pkl')
        except : pass

    log("#### Colcat to Hash encoding #############################################")
    from utils import util_text
    dfcat_bin, col_hash_model= util_text.pd_coltext_minhash(df[colcat], colcat,
                                                            return_val="dataframe,param", **pars_minhash )
    colcat_minhash = list(dfcat_bin.columns)
    log(col_hash_model)

    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(dfcat_bin, prefix, pars['path_features_store'])
       save(colcat_minhash, pars['path_pipeline_export'] + f"/{prefix}.pkl" )
       save(pars_minhash,   pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )
       save(col_hash_model, pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )

    col_pars = {}
    col_pars['col_hash_model'] = col_hash_model
    col_pars['cols_new'] = {
     'colcat_minhash' :  colcat_minhash  ### list
    }
    return dfcat_bin, col_pars



def os_convert_topython_code(txt):
    # from sympy import sympify
    # converter = {
    #     'sub': lambda x, y: x - y,
    #     'div': lambda x, y: x / y,
    #     'mul': lambda x, y: x * y,
    #     'add': lambda x, y: x + y,
    #     'neg': lambda x: -x,
    #     'pow': lambda x, y: x ** y
    # }
    # formula = sympify( txt, locals=converter)
    # print(formula)
    pass


def save_json(js, pfile, mode='a'):
    import  json
    with open(pfile, mode=mode) as fp :
        json.dump(js, fp)


def pd_col_genetic_transform(df=None, col=None, pars=None):
    """
        Find Symbolic formulae for faeture engineering

    """
    prefix = 'col_genetic'
    ######################################################################################
    from gplearn.genetic import SymbolicTransformer
    from gplearn.functions import make_function
    import random

    colX          = col # [col_ for col_ in col if col_ not in coly]
    train_X       = df[colX].fillna(method='ffill')
    feature_name_ = colX

    def squaree(x):  return x * x
    square_ = make_function(function=squaree, name='square_', arity=1)

    function_set = pars.get('function_set',
                            ['add', 'sub', 'mul', 'div',  'sqrt', 'log', 'abs', 'neg', 'inv','tan', square_])
    pars_genetic = pars.get('pars_genetic',
                             {'generations': 5, 'population_size': 10,  ### Higher than nb_features
                              'metric': 'spearman',
                              'tournament_size': 20, 'stopping_criteria': 1.0, 'const_range': (-1., 1.),
                              'p_crossover': 0.9, 'p_subtree_mutation': 0.01, 'p_hoist_mutation': 0.01,
                              'p_point_mutation': 0.01, 'p_point_replace': 0.05,
                              'parsimony_coefficient' : 0.005,   ####   0.00005 Control Complexity
                              'max_samples' : 0.9, 'verbose' : 1,

                              #'n_components'      ### Control number of outtput features  : n_components
                              'random_state' :0, 'n_jobs' : 4,
                              })

    if 'path_pipeline' in pars :   #### Inference time
        gp   = load(pars['path_pipeline'] + f"/{prefix}_model.pkl" )
        pars = load(pars['path_pipeline'] + f"/{prefix}_pars.pkl" )

    else :     ### Training time
        coly     = pars['coly']
        train_y  = pars['dfy']
        gp = SymbolicTransformer(hall_of_fame  = train_X.shape[1] + 1,  ### Buggy
                                 n_components  = pars_genetic.get('n_components', train_X.shape[1] ),
                                 feature_names = feature_name_,
                                 function_set  = function_set,
                                 **pars_genetic)
        gp.fit(train_X, train_y)

    ##### Transform Data  #########################################
    df_genetic = gp.transform(train_X)
    tag = random.randint(0,10)   #### UNIQUE TAG
    col_genetic  = [ f"gen_{tag}_{i}" for i in range(df_genetic.shape[1])]
    df_genetic   = pd.DataFrame(df_genetic, columns= col_genetic, index = train_X.index )
    df_genetic.index = train_X.index
    pars_gen_all = {'pars_genetic'  : pars_genetic , 'function_set' : function_set }

    ##### Formulae Exrraction #####################################
    formula   = str(gp).replace("[","").replace("]","")
    flist     = formula.split(",\n")
    form_dict = {  x: flist[i]  for i,x in enumerate(col_genetic) }
    pars_gen_all['formulae_dict'] = form_dict
    log("########## Formulae ", form_dict)
    # col_pars['map_dict'] = dict(zip(train_X.columns.to_list(), feature_name_))

    col_new = col_genetic

    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(df_genetic, 'df_genetic', pars['path_features_store'])
       save(gp,             pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )
       save(col_genetic,    pars['path_pipeline_export'] + f"/{prefix}.pkl" )
       save(pars_gen_all,   pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )
       # save(form_dict,      pars['path_pipeline_export'] + f"/{prefix}_formula.pkl")
       save_json(form_dict, pars['path_pipeline_export'] + f"/{prefix}_formula.json")   ### Human readable


    col_pars = {'prefix' : prefix , 'path' :   pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
    col_pars['cols_new'] = {
       prefix :  col_new  ### list
    }
    return df_genetic, col_pars







def pd_coltext_universal_google(df, col, pars={}):
    """
     # Universal sentence encoding from Tensorflow
       Text ---> Vectors
    from source.preprocessors import  pd_coltext_universal_google
    https://tfhub.dev/google/universal-sentence-encoder-multilingual/3

    #latest Tensorflow that supports sentencepiece is 1.13.1
    !pip uninstall --quiet --yes tensorflow
    !pip install --quiet tensorflow-gpu==1.13.1
    !pip install --quiet tensorflow-hub
    pip install --quiet tf-sentencepiece, simpleneighbors
    !pip install --quiet simpleneighbors

    # df : dataframe
    # col : list of text colnum names
    pars
    """
    prefix = "coltext_universal_google"
    if 'path_pipeline' in  pars  :   ### Load during Inference
       coltext_embed = load( pars['path_pipeline'] + "/{prefix}.pkl" )
       pars_model    = load( pars['path_pipeline'] + "/{prefix}_pars.pkl" )

    ####### Custom Code ###############################################################
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text
    #from tqdm import tqdm #progress bar
    uri_list = [
    ]
    url_default = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    url         = pars.get("model_uri", url_default )
    model       = hub.load( url )
    pars_model  = {}
    dfall       = None
    for coli in col[:1] :
        X = []
        for r in (df[coli]):
            if pd.isnull(r)==True :
                r=""
            emb = model(r)
            review_emb = tf.reshape(emb, [-1]).numpy()
            X.append(review_emb)

        dfi   = pd.DataFrame(X, columns= [ coli + "_" + str(i) for i in range( len(X[0]))   ] ,
                             index = df.index)
        dfall = pd.concat((dfall, dfi))  if dfall is not None else dfi

    coltext_embed = list(dfall.columns)


    ##### Export ####################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(dfall, 'dftext_embed', pars['path_features_store'])
       save(coltext_embed,  pars['path_pipeline_export'] + "/{prefix}.pkl" )
       save(pars_model,     pars['path_pipeline_export'] + "/{prefix}_pars.pkl" )
       save(model,          pars['path_pipeline_export'] + "/{prefix}_model.pkl" )
       model_uri = pars['path_pipeline_export'] + "/{prefix}_model.pkl"


    col_pars = {'model_uri' :  model_uri, 'pars': pars_model}
    col_pars['cols_new']      = {
       'coltext_universal_google' :  coltext_embed ### list
    }
    return dfall, col_pars



def pd_col_covariate_shift_adjustment():
   """
    https://towardsdatascience.com/understanding-dataset-shift-f2a5a262a766

     Covariate shift has been extensively studied in the literature, and a number of proposals to work under it have been published. Some of the most important ones include:
        Weighting the log-likelihood function (Shimodaira, 2000)
        Importance weighted cross-validation (Sugiyama et al, 2007 JMLR)
        Integrated optimization problem. Discriminative learning. (Bickel et al, 2009 JMRL)
        Kernel mean matching (Gretton et al., 2009)
        Adversarial search (Globerson et al, 2009)
        Frank-Wolfe algorithm (Wen et al., 2015)

import numpy as np
from scipy import sparse

# .. for plotting ..
import pylab as plt
# .. to generate a synthetic dataset ..
from sklearn import datasets

n_samples, n_features = 1000, 10000
A, b = datasets.make_regression(n_samples, n_features)

def FW(alpha, max_iter=200, tol=1e-8):
    # .. initial estimate, could be any feasible point ..
    x_t = sparse.dok_matrix((n_features, 1))
    trace = []  # to keep track of the gap

    # .. some quantities can be precomputed ..
    Atb = A.T.dot(b)
    for it in range(max_iter):
        # .. compute gradient. Slightly more involved than usual because ..
        # .. of the use of sparse matrices ..
        Ax = x_t.T.dot(A.T).ravel()
        grad = (A.T.dot(Ax) - Atb)

        # .. the LMO results in a vector that is zero everywhere except for ..
        # .. a single index. Of this vector we only store its index and magnitude ..
        idx_oracle = np.argmax(np.abs(grad))
        mag_oracle = alpha * np.sign(-grad[idx_oracle])
        g_t = x_t.T.dot(grad).ravel() - grad[idx_oracle] * mag_oracle
        trace.append(g_t)
        if g_t <= tol:
            break
        q_t = A[:, idx_oracle] * mag_oracle - Ax
        step_size = min(q_t.dot(b - Ax) / q_t.dot(q_t), 1.)
        x_t = (1. - step_size) * x_t
        x_t[idx_oracle] = x_t[idx_oracle] + step_size * mag_oracle
    return x_t, np.array(trace)

# .. plot evolution of FW gap ..
sol, trace = FW(.5 * n_features)
plt.plot(trace)
plt.yscale('log')
plt.xlabel('Number of iterations')
plt.ylabel('FW gap')
plt.title('FW on a Lasso problem')
plt.grid()
plt.show()

sparsity = np.mean(sol.toarray().ravel() != 0)
print('Sparsity of solution: %s%%' % (sparsity * 100))

   """
   pass

if __name__ == "__main__":
    import fire
    fire.Fire()


