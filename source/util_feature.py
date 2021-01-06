# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Methods for feature extraction and preprocessing
util_feature: input/output is pandas
"""
import copy
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#############################################################################################
print("os.getcwd", os.getcwd())

def log(*s, n=0, m=1, **kw):
    sspace = "#" * n
    sjump = "\n" * m

    ### Implement Logging
    print(sjump, sspace, s, sspace, flush=True, **kw)

class dict2(object):
    def __init__(self, d):
        self.__dict__ = d


#############################################################################################
def save_list(path, name_list, glob):
    import pickle, os
    os.makedirs(path, exist_ok=True)
    for t in name_list:
        log(t)
        pickle.dump(glob[t], open(f'{path}/{t}.pkl', mode='wb'))        #problem with not saving in right folder solved

def save(obj, path):
    import cloudpickle as pickle, os
    os.makedirs(  os.path.dirname( path), exist_ok=True)
    log(f'{path}')
    pickle.dump(obj, open(f'{path}', mode='wb'))


def load(file_name):
    import cloudpickle  as pickle
    return pickle.load(open(f'{file_name}', mode='rb'))



def load_dataset(path_data_x, path_data_y='',  colid="jobId", n_sample=-1):
    log('loading', colid, path_data_x)
    import glob 
    import ntpath
    flist = glob.glob( ntpath.dirname(path_data_x)+"/*" )#ntpath.dirname(path_data_x)+"/*"
    flist = [ f for f in flist if os.path.splitext(f)[1][1:].strip().lower() in [ 'zip', 'parquet'] and ntpath.basename(f)[:8] in ['features'] ]
    assert len(flist) > 0 , " No file: " +path_data_x

    log("###### Load dfX target values #####################################")
    print(flist)
    df    = None
    for fi in flist :
        if ".parquet" in fi :  dfi = pd.read_parquet(fi) # + "/features.zip")
        if ".zip" in fi  :     dfi = pd.read_csv(fi) # + "/features.zip")
        df = pd.concat((df, dfi))  if df is not None else dfi
    assert len(df) > 0 , " Dataframe is empty: " + path_data_x
    log("dfX_raw", df.T.head(4))


    # df = pd.read_csv(path_data_x) # + "/features.zip")
    if colid not in list(df.columns ):
      df[colid] = np.arange(0, len(df))
    df        = df.set_index(colid)

        
    if n_sample > 0: 
        df = df.iloc[:n_sample, :]

    log("###### Load dfy target values ###################################")
    try:
        flist = glob.glob( ntpath.dirname(path_data_y)+"/*" )
        flist = [ f for f in flist if os.path.splitext(f)[1][1:].strip().lower() in [ 'zip', 'parquet'] and ntpath.basename(f)[:6] in ['target']]
        dfy   = pd.DataFrame()
        dfi   = None
        for fi in flist :
            if ".parquet" in fi :  dfi = pd.read_parquet(fi) # + "/features.zip")
            if ".zip" in fi  :     dfi = pd.read_csv(fi) # + "/features.zip")
            dfy = pd.concat((dfy, dfi)) 

        log("dfy", dfy.head(4).T)        
        if colid not in list(dfy.columns) :
            dfy[colid] = np.arange(0, len(dfy))
        
        df = df.join(dfy.set_index(colid), on=colid, how='left', )
    except Exception as e :
        log("dfy not loaded", path_data_y, e  )
        
    return df





def pd_read_file(path_glob="*.pkl", ignore_index=True,  cols=None,
                  verbose=False, nrows=-1, concat_sort=True, n_pool=1, 
                  drop_duplicates=None, shop_id=None, nmax= 1000000000,  **kw):
  """
     "*.pkl, *.parquet"
  
  """ 
  # os.environ["MODIN_ENGINE"] = "dask"   
  # import modin.pandas as pd  
  import glob, gc,  pandas as pd, os
  readers = {
          ".pkl"     : pd.read_pickle,
          ".parquet" : pd.read_parquet,
          ".csv"     : pd.read_csv,           
          ".txt"     : pd.read_csv,
   }
  from multiprocessing.pool import ThreadPool
  pool = ThreadPool(processes=n_pool)
  
  path_glob_list = [ t.strip() for t in  path_glob.split(",") ]
  file_list = []
  for pg in path_glob_list : 
    file_list = file_list + glob.glob(pg)     
  file_list.sort()
  n_file = len(file_list)
  if n_file < 1: raise Exception("No file exist", path_glob)  
  
  # print("ok", verbose)
  dfall = pd.DataFrame()

  if verbose : log(n_file,  n_file // n_pool )
  for j in range(n_file // n_pool +1 ) :
      log("Pool", j)  
      job_list =[]   
      for i in range(n_pool):  
         if n_pool*j + i >= n_file  : break 
         filei         = file_list[n_pool*j + i]
         ext           = os.path.splitext(filei)[1]
         pd_reader_obj = readers[ext]                            
         job_list.append( pool.apply_async(pd_reader_obj, (filei, )))  
         if verbose : log(j, filei)
    
      for i in range(n_pool):  
        if i >= len(job_list): break  
        dfi   = job_list[i].get()

        if shop_id is not None and "shop_id" in  dfi.columns : dfi = dfi[ dfi['shop_id'] == shop_id ]         
        if cols is not None :    dfi = dfi[cols] 
        if nrows > 0        :    dfi = dfi.iloc[:nrows,:]
        if drop_duplicates is not None  : dfi = dfi.drop_duplicates(drop_duplicates) 
        gc.collect()            
            
        dfall = pd.concat( (dfall, dfi), ignore_index=ignore_index, sort= concat_sort)        
        #log("Len", n_pool*j + i, len(dfall))
        del dfi; gc.collect()
        
        if len(dfall) > nmax : return dfall
        
  if verbose : log(n_file, j * n_file//n_pool )
  gc.collect()
  return dfall  



def load_function_uri(uri_name="myfolder/myfile.py::myFunction"):
    """
    #load dynamically function from URI pattern
    #"dataset"        : "mlmodels.preprocess.generic:pandasDataset"
    ###### External File processor :
    #"dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
    """

    import importlib, sys
    from pathlib import Path
    pkg = uri_name.split("::")

    assert len(pkg) > 1, "  Missing :   in  uri_name module_name:function_or_class "
    package, name = pkg[0], pkg[1]

    try:
        #### Import from package mlmodels sub-folder
        return  getattr(importlib.import_module(package), name)

    except Exception as e1:
        try:
            ### Add Folder to Path and Load absoluate path module
            path_parent = str(Path(package).parent.parent.absolute())
            sys.path.append(path_parent)
            log(path_parent)

            #### import Absolute Path model_tf.1_lstm
            model_name   = Path(package).stem  # remove .py
            package_name = str(Path(package).parts[-2]) + "." + str(model_name)
            #log(package_name, config_name)
            return  getattr(importlib.import_module(package_name), name)

        except Exception as e2:
            raise NameError(f"Module {pkg} notfound, {e1}, {e2}")


#############################################################################################
def metrics_eval(metric_list=["mean_squared_error"], ytrue=None, ypred=None, ypred_proba=None, return_dict=False):
    """
      Generic metrics calculation, using sklearn naming pattern
    """
    import pandas as pd, importlib
    mdict = {"metric_name": [],
             "metric_val": [],
             "n_sample": [len(ytrue)] * len(metric_list)}

    if isinstance(metric_list, str):
        metric_list = [metric_list]

    for metric_name in metric_list:
        mod = "sklearn.metrics"


        if metric_name in ["roc_auc_score"]:        #y_pred_proba is not defined
            #### Ok for Multi-Class
            metric_scorer = getattr(importlib.import_module(mod), metric_name)
            mval_=[]
            for i_ in range(ypred_proba.shape[1]):
                mval_.append(metric_scorer(pd.get_dummies(ytrue).to_numpy()[:,i_], ypred_proba[:,i_]))
            mval          = np.mean(np.array(mval_))

        elif metric_name in ["root_mean_squared_error"]:
            metric_scorer = getattr(importlib.import_module(mod), "mean_squared_error")
            mval          = np.sqrt(metric_scorer(ytrue, ypred))

        else:
            metric_scorer = getattr(importlib.import_module(mod), metric_name)
            mval = metric_scorer(ytrue, ypred)

        mdict["metric_name"].append(metric_name)
        mdict["metric_val"].append(mval)

    if return_dict: return mdict

    mdict = pd.DataFrame(mdict)
    return mdict


def pd_stat_dataset_shift(dftrain, dftest, colused, nsample=10000, buckets=5, axis=0):
    ### Population Stability Index
    ll = {'colname': [], 'psi': []}
    for coli in colused:
        print(coli)
        psi = pd_stat_datashift_psi(expected=dftrain[coli].sample(nsample),
                                    actual=dftest[coli].sample(nsample),
                                    buckettype='bins', buckets=buckets, axis=axis)

        ll['colname'].append(coli)
        ll['psi'].append(psi)
    metrics_psi = pd.DataFrame(ll)
    metrics_psi['size'] = nsample
    return metrics_psi


def pd_stat_datashift_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return (value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return (psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i, :], actual[i, :], buckets)

    return (psi_values)


####################################################################################################
####################################################################################################
def estimator_std_normal(err, alpha=0.05, ):
    # estimate_std( err, alpha=0.05, )
    from scipy import stats
    n = len(err)  # sample sizes
    s2 = np.var(err, ddof=1)  # sample variance
    df = n - 1  # degrees of freedom
    upper = np.sqrt((n - 1) * s2 / stats.chi2.ppf(alpha / 2, df))
    lower = np.sqrt((n - 1) * s2 / stats.chi2.ppf(1 - alpha / 2, df))

    return np.sqrt(s2), (lower, upper)


def estimator_boostrap_bayes(err, alpha=0.05, ):
    from scipy.stats import bayes_mvs
    mean, var, std = bayes_mvs(err, alpha=alpha)
    return mean, var, std


def estimator_bootstrap(err, custom_stat=None, alpha=0.05, n_iter=10000):
    """
      def custom_stat(values, axis=1):
      # stat_val = np.mean(np.asmatrix(values),axis=axis)
      # stat_val = np.std(np.asmatrix(values),axis=axis)p.mean
      stat_val = np.sqrt(np.mean(np.asmatrix(values*values),axis=axis))
      return stat_val
    """
    import bootstrapped.bootstrap as bs
    res = bs.bootstrap(err, stat_func=custom_stat, alpha=alpha, num_iterations=n_iter)
    return res


####################################################################################################
def test_heteroscedacity(y, y_pred, pred_value_only=1):
    ss = """
       Test  Heteroscedacity :  Residual**2  = Linear(X, Pred, Pred**2)
       F pvalues < 0.01 : Null is Rejected  ---> Not Homoscedastic
       het_breuschpagan
    
    """
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    error    = y_pred - y

    ypred_df = pd.DataFrame({"pcst": [1.0] * len(y), "pred": y_pred, "pred2": y_pred * y_pred})
    labels   = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]
    test1    = het_breuschpagan(error * error, ypred_df.values)
    test2    = het_white(error * error, ypred_df.values)
    ddict    = {"het-breuschpagan": dict(zip(labels, test1)),
             "het-white": dict(zip(labels, test2)),
             }

    return ddict


def test_normality(error, distribution="norm", test_size_limit=5000):
    """
       Test  Is Normal distribution
       F pvalues < 0.01 : Rejected
    
    """
    from scipy.stats import shapiro, anderson, kstest

    error2 = error

    error2 = error2[np.random.choice(len(error2), 5000)]  # limit test
    test1  = shapiro(error2)
    ddict1 = dict(zip(["shapiro", "W-p-value"], test1))

    test2  = anderson(error2, dist=distribution)
    ddict2 = dict(zip(["anderson", "p-value", "P critical"], test2))

    test3  = kstest(error2, distribution)
    ddict3 = dict(zip(["kstest", "p-value"], test3))

    ddict  = dict(zip(["shapiro", "anderson", "kstest"], [ddict1, ddict2, ddict3]))

    return ddict


def test_mutualinfo(error, Xtest, colname=None, bins=5):
    """
       Test  Error vs Input Variable Independance byt Mutual ifno
       sklearn.feature_selection.mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
    
    """
    from sklearn.feature_selection import mutual_info_classif
    error = pd.DataFrame({"error": error})
    error_dis, _ = pd_colnum_tocat(error, bins=bins, method="quantile")
    # print(error_dis)

    res = mutual_info_classif(Xtest.values, error_dis.values.ravel())

    return dict(zip(colname, res))


####################################################################################################
def feature_importance_perm(clf, Xtrain, ytrain, cols, n_repeats=8, scoring='neg_root_mean_squared_error',
                            show_graph=1):
    from sklearn.inspection import permutation_importance
    result = permutation_importance(clf, Xtrain[cols], ytrain, n_repeats=n_repeats,
                                    random_state=42, scoring=scoring)
    perm_sorted_idx = result.importances_mean.argsort()

    cols_sorted = [cols[i] for i in perm_sorted_idx]
    cols_scores = list(result.importances_mean[perm_sorted_idx])
    # print(cols_sorted, cols_scores)

    dfmetrics = pd.DataFrame({"colname": cols_sorted,
                              "score": cols_scores}).sort_values("score", ascending=False)

    dfmetrics["rank"] = np.arange(len(cols))

    cmax = min(10, len(cols))

    if show_graph:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

        ax1.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                    labels=cols_sorted)

        ax1.set_yticklabels(cols_sorted)
        ax1.set_ylim((0, len(cols)))

        fig.tight_layout()
        plt.show()

    return dfmetrics, result


def feature_selection_multicolinear(df, threshold=1.0):
    from scipy.stats import spearmanr
    from scipy.cluster import hierarchy
    from collections import defaultdict

    cols                      = list(df.columns)
    corr                      = spearmanr(df).correlation  # Ordinalon
    corr_linkage              = hierarchy.ward(corr)
    cluster_ids               = hierarchy.fcluster(corr_linkage, threshold, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    ### return valid cols
    return [cols[i] for i in selected_features]


def feature_correlation_cat(df, colused):
    from scipy.stats import spearmanr
    from scipy.cluster import hierarchy

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(df[colused]).correlation  # Ordinalon
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(corr_linkage, labels=colused, ax=ax1,
                                  leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.show()


####################################################################################################
####################################################################################################
def pd_feature_generate_cross(df, cols, cols_cross_input=None, pct_threshold=0.2, m_combination=2):
    """
       Generate Xi.Xj features and filter based on stats threshold
    """
    import itertools

    dfX = df[cols]
    n = len(df) + 0.0
    m = len(cols)
    col_cross = []
    # cols = list(dfX.columns)

    if cols_cross_input is None:
        for i, j in itertools.combinations([k for k in range(0, m)], m_combination):
            # print(i,j)
            y = dfX.iloc[:, i] * dfX.iloc[:, j]
            ratio = y.sum() / n
            if ratio > pct_threshold:
                col_cross.append((cols[i], cols[j], ratio))
    else:
        col_cross = cols_cross_input

    # col_cross = [ col   for col,x in msize.items() if x > pct_threshold ]
    # print(col_cross)
    dfX_cross = dfX.iloc[:, :1]

    for coli, colj, _ in col_cross:
        # coli, colj = colij.split("-")[0], colij.split("-")[1]
        dfX_cross[coli + "-" + colj] = dfX[coli] * dfX[colj]

      
    del dfX_cross[cols[0]]      #when colcross is empty, this make problem
    return dfX_cross, col_cross


def pd_col_to_onehot(dfref, colname=None, colonehot=None, return_val="dataframe,column"):
    """
    :param df:
    :param colname:
    :param colonehot: previous one hot columns
    :param returncol:
    :return:
    """
    df = copy.deepcopy(dfref)
    coladded = []
    colname = list(df.columns) if colname is None else colname

    # Encode each column into OneHot
    for x in colname:
        try:
            nunique = len(df[x].unique())
            print(x, nunique, df.shape, flush=True)

            if nunique > 2:
                df = pd.concat([df, pd.get_dummies(df[x], prefix=x)], axis=1).drop([x], axis=1)
            else:
                df[x] = df[x].factorize()[0]  # put into 0,1 format
            coladded.append(x)
        except Exception as e:
            print(x, e)

    # Add missing category columns
    if colonehot is not None:
        for x in colonehot:
            if not x in df.columns:
                df[x] = 0
                print(x, "added")
                coladded.append(x)

    #### Include Pre-defined columns
    colnew = colonehot if colonehot is not None else [c for c in df.columns if c not in colname]
    if return_val == "dataframe,param":
        return df[colnew], colnew

    else:
        return df[colnew]


def pd_colcat_mergecol(df, col_list, x0, colid="easy_id"):
    """
       Merge category onehot column
    :param df:
    :param l:
    :param x0:
    :return:
    """
    dfz = pd.DataFrame({colid: df[colid].values})
    for t in col_list:
        ix     = t.rfind("_")
        val    = int(t[ix + 1:])
        print(ix, t[ix + 1:])
        dfz[t] = df[t].apply(lambda x: val if x > 0 else 0)

    # print(dfz)
    dfz = dfz.set_index(colid)
    dfz[x0] = dfz.iloc[:, :].sum(1)
    for t in dfz.columns:
        if t != x0:
            del dfz[t]
    return dfz


def pd_colcat_tonum(df, colcat="all", drop_single_label=False, drop_fact_dict=True):
    """
    Encoding a data-set with mixed data (numerical and categorical) to a numerical-only data-set,
    using the following logic:
    * categorical with only a single value will be marked as zero (or dropped, if requested)
    * categorical with two values will be replaced with the result of Pandas `factorize`
    * categorical with more than two values will be replaced with the result of Pandas `get_dummies`
    * numerical columns will not be modified
    **Returns:** DataFrame or (DataFrame, dict). If `drop_fact_dict` is True, returns the encoded DataFrame.
    else, returns a tuple of the encoded DataFrame and dictionary, where each key is a two-value column, and the
    value is the original labels, as supplied by Pandas `factorize`. Will be empty if no two-value columns are
    present in the data-set
    Parameters
    ----------
    df : NumPy ndarray / Pandas DataFrame
        The data-set to encode
    colcat : sequence / string
        A sequence of the nominal (categorical) columns in the dataset. If string, must be 'all' to state that
        all columns are nominal. If None, nothing happens. Default: 'all'
    drop_single_label : Boolean, default = False
        If True, nominal columns with a only a single value will be dropped.
    drop_fact_dict : Boolean, default = True
        If True, the return value will be the encoded DataFrame alone. If False, it will be a tuple of
        the DataFrame and the dictionary of the binary factorization (originating from pd.factorize)
    """
    # df = convert(df, "dataframe")
    if colcat is None:
        return df
    elif colcat == "all":
        colcat = df.columns
    df_out = pd.DataFrame()
    binary_columns_dict = dict()

    for col in df.columns:
        if col not in colcat:
            df_out.loc[:, col] = df[col]

        else:
            unique_values = pd.unique(df[col])
            if len(unique_values) == 1 and not drop_single_label:
                df_out.loc[:, col] = 0
            elif len(unique_values) == 2:
                df_out.loc[:, col], binary_columns_dict[col] = pd.factorize(df[col])
            else:
                dummies = pd.get_dummies(df[col], prefix=col)
                df_out = pd.concat([df_out, dummies], axis=1)
    if drop_fact_dict:
        return df_out
    else:
        return df_out, binary_columns_dict


def pd_colcat_mapping(df, colname):
    """
       map category to integers
    :param df:
    :param colname:
    :return:
    """
    mapping_rev = {
        col: {n: cat for n, cat in enumerate(df[col].astype("category").cat.categories)}
        for col in df[colname]
    }

    mapping = {
        col: {cat: n for n, cat in enumerate(df[col].astype("category").cat.categories)}
        for col in df[colname]
    }

    return {"cat_map": mapping, "cat_map_inverse": mapping_rev}


def pd_colcat_toint(dfref, colname, colcat_map=None, suffix=None):
    df = dfref[colname]
    suffix = "" if suffix is None else suffix
    colname_new = []

    if colcat_map is not None:
        for col in colname:
            print(col, col + suffix)
            ddict            = colcat_map[col]["encode"]
            # print(ddict)
            df[col + suffix] = df[col].apply(lambda x: ddict.get(x))
            colname_new.append(col + suffix)

        return df[colname_new], colcat_map

    colcat_map = {}
    for col in colname:
        colcat_map[col]           = {}
        df[col + suffix], label   = df[col].factorize()
        colcat_map[col]["decode"] = {i: t for i, t in enumerate(list(label))}
        colcat_map[col]["encode"] = {t: i for i, t in enumerate(list(label))}
        colname_new.append(col + suffix)

    return df[colname_new], colcat_map



def pd_colnum_tocat(  df, colname=None, colexclude=None, colbinmap=None, bins=5, suffix="_bin",
        method="uniform", na_value=-1, return_val="dataframe,param",
        params={"KMeans_n_clusters": 8, "KMeans_init": 'k-means++', "KMeans_n_init": 10,
                "KMeans_max_iter": 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances": 'auto',
                "KMeans_verbose": 0, "KMeans_random_state": None,
                "KMeans_copy_x": True, "KMeans_n_jobs": None, "KMeans_algorithm": 'auto'}
):
    """
    colbinmap = for each column, definition of bins
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
       :param df:
       :param method:
       :return:
    """

    colexclude = [] if colexclude is None else colexclude
    colname    = colname if colname is not None else list(df.columns)
    colnew     = []
    col_stat   = OrderedDict()
    colmap     = OrderedDict()

    # Bin Algo
    # p = dict2(params)  # Bin  model params
    def bin_create(dfc, bins):
        mi, ma     = dfc.min(), dfc.max()
        space      = (ma - mi) / bins
        lbins      = [mi + i * space for i in range(bins + 1)]
        lbins[0]  -= 0.0001
        return lbins

    def bin_create_quantile(dfc, bins):
        qt_list_ref = np.arange(0, 1.00001, 1.0 / bins)
        # print(qt_list_ref )
        qt_list = dfc.quantile(qt_list_ref)
        # print(qt_list )
        lbins = list(qt_list.values)
        lbins[0] -= 0.01
        return lbins

    # def bin_create_cluster(dfc):
    #     kmeans = KMeans(n_clusters= p.KMeans_n_clusters, init=p.KMeans_init, n_init=p.KMeans_n_init,
    #         max_iter=p.KMeans_max_iter, tol=p.KMeans_tol, precompute_distances=p.KMeans_precompute_distances,
    #         verbose=p.KMeans_verbose, random_state=p.KMeans_random_state,
    #         copy_x=p.KMeans_copy_x, n_jobs=p.KMeans_n_jobs, algorithm=p.KMeans_algorithm).fit(dfc)
    #     return kmeans.predict(dfc)

    # Loop  on all columns
    for c in colname:
        if c in colexclude:
            continue
        print(c)
        df[c] = df[c].astype(np.float32)

        # Using Prebin Map data
        if colbinmap is not None:
            lbins = colbinmap.get(c)
        else:
            if method == "quantile":
                lbins = bin_create_quantile(df[c], bins)
            # elif method == "cluster":
            #     non_nan_index = np.where(~np.isnan(df[c]))[0]
            #     lbins = bin_create_cluster(df.loc[non_nan_index][c].values.reshape((-1, 1))).reshape((-1,))
            else:
                lbins = bin_create(df[c], bins)

        cbin = c + suffix
        # if method == 'cluster':
        #     df.loc[non_nan_index][cbin] = lbins
        # else:
        labels   = np.arange(0, len(lbins) - 1)
        df[cbin] = pd.cut(df[c], bins=lbins, labels=labels)

        # NA processing
        df[cbin]  = df[cbin].astype("float")
        df[cbin]  = df[cbin].apply(lambda x: x if x >= 0.0 else na_value)  # 3 NA Values
        df[cbin]  = df[cbin].astype("int")
        col_stat  = df.groupby(cbin).agg({c: {"size", "min", "mean", "max"}})
        colmap[c] = lbins
        colnew.append(cbin)

        print(col_stat)

    if return_val == "dataframe":
        return df[colnew]

    elif return_val == "param":
        return colmap
    else:
        return df[colnew], colmap


def pd_colnum_normalize(df0, colname, pars, suffix="_norm", return_val='dataframe,param'):
    """
    :param df:
    :param colnum_log:
    :param colproba:
    :return:
    """
    df = df0[colname]
    for x in colname:
        for t in pars['pipe_list'] :
            try:
                if t['name'] == 'log'         : df[x] = np.log(df[x].values.astype(np.float64))
                if t['name'] == 'fillna'      : df[x] = df[x].fillna( t['na_val'] )
                if t['name'] == 'minmax_norm' : df[x] = (df[x] - df[x].min() )/ ( df[x].max() - df[x].min() )
            except Exception as e:
                pass

    df.columns  = [ t + suffix for t in df.columns ]
    colnum_norm = list(df.columns)
    return df, colnum_norm


def pd_col_merge_onehot(df, colname):
    """
      Merge columns into single (hotn
    :param df:
    :param colname:
    :return :
    """
    dd = {}
    for x in colname:
        merge_array = []
        for t in df.columns:
            if x in t and t[len(x): len(x) + 1] == "_":
                merge_array.append(t)
        dd[x] = merge_array
    return dd


def pd_col_to_num(df, colname=None, default=np.nan):
    def to_float(x):
        try:
            return float(x)
        except BaseException:
            return default

    colname = list(df.columns) if colname is None else colname
    for c in colname:
        df[c] = df[c].apply(lambda x: to_float(x))
    return df


def pd_col_filter(df, filter_val=None, iscol=1):
    """
   # Remove Columns where Index Value is not in the filter_value
   # filter1= X_client['client_id'].values
   :param df:
   :param filter_val:
   :param iscol:
   :return:
   """
    axis = 1 if iscol == 1 else 0
    col_delete = []
    for colname in df.index.values:  # !!!! row Delete
        if colname in filter_val:
            col_delete.append(colname)

    df2 = df.drop(col_delete, axis=axis, inplace=False)
    return df2


def pd_col_fillna(
        dfref,
        colname=None,
        method="frequent",
        value=None,
        colgroupby=None,
        return_val="dataframe,param",
):
    """
    Function to fill NaNs with a specific value in certain columns
    Arguments:
        df:            dataframe
        colname:      list of columns to remove text
        value:         value to replace NaNs with
    Returns:
        df:            new dataframe with filled values
    """
    colname = list(dfref.columns) if colname is None else colname
    df = dfref[colname]
    params = {"method": method, "na_value": {}}
    for col in colname:
        nb_nans = df[col].isna().sum()

        if method == "frequent":
            x = df[col].value_counts().idxmax()

        if method == "mode":
            x = df[col].mode()

        if method == "median":
            x = df[col].median()

        if method == "median_conditional":
            x = df.groupby(colgroupby)[col].transform("median")  # Conditional median

        value = x if value is None else value
        print(col, nb_nans, "replaceBY", value)
        params["na_value"][col] = value
        df[col] = df[col].fillna(value)

    if return_val == "dataframe,param":
        return df, params
    else:
        return df


def pd_pipeline_apply(df, pipeline):
    """
      pipe_preprocess_colnum = [
      (pd_col_to_num, {"val": "?", })
    , (pd_colnum_tocat, {"colname": None, "colbinmap": colnum_binmap, 'bins': 5,
                         "method": "uniform", "suffix": "_bin",
                         "return_val": "dataframe"})
    , (pd_col_to_onehot, {"colname": None, "colonehot": colnum_onehot,
                          "return_val": "dataframe"})
      ]
    :param df:
    :param pipeline:
    :return:
    """
    dfi = copy.deepcopy(df)
    for i, function in enumerate(pipeline):
        print(
            "############## Pipeline ", i, "Start", dfi.shape, str(function[0].__name__), flush=True
        )
        dfi = function[0](dfi, **function[1])
        print("############## Pipeline  ", i, "Finished", dfi.shape, flush=True)
    return dfi


def pd_stat_correl_pair(df, coltarget=None, colname=None):
    """
      Genearte correletion between the column and target column
      df represents the dataframe comprising the column and colname comprising the target column
    :param df:
    :param colname: list of columns
    :param coltarget : target column
    :return:
    """
    from scipy.stats import pearsonr

    colname = colname if colname is not None else list(df.columns)
    target_corr = []
    for col in colname:
        target_corr.append(pearsonr(df[col].values, df[coltarget].values)[0])

    df_correl = pd.DataFrame({"colx": [""] * len(colname), "coly": colname, "correl": target_corr})
    df_correl[coltarget] = colname
    return df_correl


def pd_stat_pandas_profile(df, savefile="report.html", title="Pandas Profile"):
    """ Describe the tables
        #Pandas-Profiling 2.0.0
        df.profile_report()
    """

    print("start profiling")
    profile = df.profile_report(title=title)
    profile.to_file(output_file=savefile)
    colexclude = profile.get_rejected_variables(threshold=0.98)
    return colexclude


def pd_stat_distribution_colnum(df):
    """ Describe the tables
   """
    coldes = ["col", "coltype", "dtype", "count", "min", "max", "nb_na", "pct_na", "median", "mean", "std", "25%", "75%", "outlier",]

    def getstat(col):
        """
         max, min, nb, nb_na, pct_na, median, qt_25, qt_75,
         nb, nb_unique, nb_na, freq_1st, freq_2th, freq_3th
         s.describe()
         count    3.0  mean     2.0 std      1.0
         min      1.0   25%      1.5  50%      2.0
         75%      2.5  max      3.0
      """
        ss = list(df[col].describe().values)
        ss = [str(df[col].dtype)] + ss
        nb_na = df[col].isnull().sum()
        ntot = len(df)
        ss = ss + [nb_na, nb_na / (ntot + 0.0)]

        return pd.Series(
            ss,
            ["dtype", "count", "mean", "std", "min", "25%", "50%", "75%", "max", "nb_na", "pct_na"],
        )

    dfdes = pd.DataFrame([], columns=coldes)
    cols = df.columns
    for col in cols:
        dtype1 = str(df[col].dtype)
        if dtype1[0:3] in ["int", "flo"]:
            row1 = getstat(col)
            dfdes = pd.concat((dfdes, row1))

        if dtype1 == "object":
            pass


def pd_stat_histogram(df, bins=50, coltarget="diff"):
    """
    :param df:
    :param bins:
    :param coltarget:
    :return:
    """
    hh = np.histogram(
        df[coltarget].values, bins=bins, range=None, normed=None, weights=None, density=None
    )
    hh2 = pd.DataFrame({"bins": hh[1][:-1], "freq": hh[0]})
    hh2["density"] = hh2["freqall"] / hh2["freqall"].sum()
    return hh2


def col_extractname(col_onehot):
    """
    Column extraction from onehot name
    :param col_onehot
    :return:
    """
    colnew = []
    for x in col_onehot:
        if len(x) > 2:
            if x[-2] == "_":
                if x[:-2] not in colnew:
                    colnew.append(x[:-2])

            elif x[-2] == "-":
                if x[:-3] not in colnew:
                    colnew.append(x[:-3])

            else:
                if x not in colnew:
                    colnew.append(x)
    return colnew


def col_remove(cols, colsremove, mode="exact"):
    """
    """
    if mode == "exact":
        for x in colsremove:
            try:
                cols.remove(x)
            except BaseException:
                pass
        return cols

    if mode == "fuzzy":
        cols3 = []
        for t in cols:
            flag = 0
            for x in colsremove:
                if x in t:
                    flag = 1
                    break
            if flag == 0:
                cols3.append(t)
        return cols3


####################################################################################################
def pd_colnum_tocat_stat(df, feature, target_col, bins, cuts=0):
    """
    Bins continuous features into equal sample size buckets and returns the target mean in each bucket. Separates out
    nulls into another bucket.
    :param df: dataframe containg features and target column
    :param feature: feature column name
    :param target_col: target column
    :param bins: Number bins required
    :param cuts: if buckets of certain specific cuts are required. Used on test data to use cuts from train.
    :return: If cuts are passed only df_grouped data is returned, else cuts and df_grouped data is returned
    """
    has_null = pd.isnull(df[feature]).sum() > 0
    if has_null == 1:
        data_null = df[pd.isnull(df[feature])]
        df = df[~pd.isnull(df[feature])]
        df.reset_index(inplace=True, drop=True)

    is_train = 0
    if cuts == 0:
        is_train = 1
        prev_cut = min(df[feature]) - 1
        cuts = [prev_cut]
        reduced_cuts = 0
        for i in range(1, bins + 1):
            next_cut = np.percentile(df[feature], i * 100 / bins)
            if next_cut > prev_cut + .000001:  # float numbers shold be compared with some threshold!
                cuts.append(next_cut)
            else:
                reduced_cuts = reduced_cuts + 1
            prev_cut = next_cut

        # if reduced_cuts>0:
        #     print('Reduced the number of bins due to less variation in feature')
        cut_series = pd.cut(df[feature], cuts)
    else:
        cut_series = pd.cut(df[feature], cuts)

    df_grouped = df.groupby([cut_series], as_index=True).agg(
        {target_col: [np.size, np.mean], feature: [np.mean]})
    df_grouped.columns                = ['_'.join(cols).strip() for cols in df_grouped.columns.values]
    df_grouped[df_grouped.index.name] = df_grouped.index
    df_grouped.reset_index(inplace    = True, drop=True)
    df_grouped                        = df_grouped[[feature] + list(df_grouped.columns[0:3])]
    df_grouped                        = df_grouped.rename(index=str, columns={target_col + '_size': 'Samples_in_bin'})
    df_grouped                        = df_grouped.reset_index(drop=True)
    corrected_bin_name                = '[' + str(min(df[feature])) + ', ' + str(df_grouped.loc[0, feature]).split(',')[1]
    df_grouped[feature]               = df_grouped[feature].astype('category')
    df_grouped[feature]               = df_grouped[feature].cat.add_categories(corrected_bin_name)
    df_grouped.loc[0, feature]        = corrected_bin_name

    if has_null == 1:
        grouped_null                              = df_grouped.loc[0:0, :].copy()
        grouped_null[feature]                     = grouped_null[feature].astype('category')
        grouped_null[feature]                     = grouped_null[feature].cat.add_categories('Nulls')
        grouped_null.loc[0, feature]              = 'Nulls'
        grouped_null.loc[0, 'Samples_in_bin']     = len(data_null)
        grouped_null.loc[0, target_col + '_mean'] = data_null[target_col].mean()
        grouped_null.loc[0, feature + '_mean']    = np.nan
        df_grouped[feature]                       = df_grouped[feature].astype('str')
        df_grouped                                = pd.concat([grouped_null, df_grouped], axis=0)
        df_grouped.reset_index(inplace=True, drop=True)

    df_grouped[feature] = df_grouped[feature].astype('str').astype('category')
    if is_train == 1:
        return (cuts, df_grouped)
    else:
        return (df_grouped)


def pd_stat_shift_trend_changes(df, feature, target_col, threshold=0.03):
    """
    Calculates number of times the trend of feature wrt target changed direction.
    :param df: df_grouped dataset
    :param feature: feature column name
    :param target_col: target column
    :param threshold: minimum % difference required to count as trend change
    :return: number of trend chagnes for the feature
    """
    df                            = df.loc[df[feature] != 'Nulls', :].reset_index(drop=True)
    target_diffs                  = df[target_col + '_mean'].diff()
    target_diffs                  = target_diffs[~np.isnan(target_diffs)].reset_index(drop=True)
    max_diff                      = df[target_col + '_mean'].max() - df[target_col + '_mean'].min()
    target_diffs_mod              = target_diffs.fillna(0).abs()
    low_change                    = target_diffs_mod < threshold * max_diff
    target_diffs_norm             = target_diffs.divide(target_diffs_mod)
    target_diffs_norm[low_change] = 0
    target_diffs_norm             = target_diffs_norm[target_diffs_norm != 0]
    target_diffs_lvl2             = target_diffs_norm.diff()
    changes                       = target_diffs_lvl2.fillna(0).abs() / 2
    tot_trend_changes             = int(changes.sum()) if ~np.isnan(changes.sum()) else 0
    return (tot_trend_changes)


def pd_stat_shift_trend_correlation(df, df_test, colname, target_col):
    """
    Calculates correlation between train and test trend of colname wrt target.
    :param df: train df data
    :param df_test: test df data
    :param colname: colname column name
    :param target_col: target column name
    :return: trend correlation between train and test
    """
    df      = df[df[colname] != 'Nulls'].reset_index(drop=True)
    df_test = df_test[df_test[colname] != 'Nulls'].reset_index(drop=True)

    if df_test.loc[0, colname] != df.loc[0, colname]:
        df_test[colname]        = df_test[colname].cat.add_categories(df.loc[0, colname])
        df_test.loc[0, colname] = df.loc[0, colname]
    df_test_train = df.merge(df_test[[colname, target_col + '_mean']], on=colname,
                             how='left',
                             suffixes=('', '_test'))
    nan_rows = pd.isnull(df_test_train[target_col + '_mean']) | pd.isnull(
        df_test_train[target_col + '_mean_test'])
    df_test_train = df_test_train.loc[~nan_rows, :]
    if len(df_test_train) > 1:
        trend_correlation = np.corrcoef(df_test_train[target_col + '_mean'],
                                        df_test_train[target_col + '_mean_test'])[0, 1]
    else:
        trend_correlation = 0
        print("Only one bin created for " + colname + ". Correlation can't be calculated")

    return (trend_correlation)


def pd_stat_shift_changes(df, target_col, features_list=0, bins=10, df_test=0):
    """
    Calculates trend changes and correlation between train/test for list of features
    :param df: dfframe containing features and target columns
    :param target_col: target column name
    :param features_list: by default creates plots for all features. If list passed, creates plots of only those features.
    :param bins: number of bins to be created from continuous colname
    :param df_test: test df which has to be compared with input df for correlation
    :return: dfframe with trend changes and trend correlation (if test df passed)
    """

    if type(features_list) == int:
        features_list = list(df.columns)
        features_list.remove(target_col)

    stats_all = []
    has_test = type(df_test) == pd.core.frame.DataFrame
    ignored = []
    for colname in features_list:
        if df[colname].dtype == 'O' or colname == target_col:
            ignored.append(colname)
        else:
            cuts, df_grouped = pd_colnum_tocat_stat(df=df, colname=colname, target_col=target_col, bins=bins)
            trend_changes    = pd_stat_shift_trend_correlation(df=df_grouped, colname=colname, target_col=target_col)
            if has_test:
                df_test            = pd_colnum_tocat_stat(df=df_test.reset_index(drop=True), colname=colname,
                                                          target_col  = target_col, bins=bins, cuts=cuts)
                trend_corr         = pd_stat_shift_trend_correlation(df_grouped, df_test, colname, target_col)
                trend_changes_test = pd_stat_shift_changes(df=df_test, colname=colname,
                                                           target_col=target_col)
                stats = [colname, trend_changes, trend_changes_test, trend_corr]
            else:
                stats = [colname, trend_changes]
            stats_all.append(stats)
    stats_all_df = pd.DataFrame(stats_all)
    stats_all_df.columns = ['colname', 'Trend_changes'] if has_test == False else ['colname', 'Trend_changes',
                                                                                   'Trend_changes_test',
                                                                                   'Trend_correlation']
    if len(ignored) > 0:
        print('Categorical features ' + str(ignored) + ' ignored. Categorical features not supported yet.')

    print('Returning stats for all numeric features')
    return (stats_all_df)


def np_conv_to_one_col(np_array, sep_char="_"):
    """
    converts string/numeric columns to one string column
    :param np_array: the numpy array with more than one column
    :param sep_char: the separator character
    """
    def row2string(row_):
        return sep_char.join([str(i) for i in row_])

    np_array_=np.apply_along_axis(row2string,1,np_array)
    return np_array_[:,None]
