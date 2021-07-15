import pandas as pd
import numpy as np


def build_amplitude_dataset(intervals, categorical_names=None):
    """
    Output dataset: one row for each interval 
    Continuous features columns: how much was moved in the feature
    Categorical features columns: Value of the category (intervals do not move along categorical features)
    """
    out = []
    for di in intervals:
        interval_change = pd.DataFrame((di.border_features.T.iloc[:,0] - di.border_features.T.iloc[:,1]).abs()).T
        if categorical_names is not None:
            interval_change[categorical_names] = di.border_features[categorical_names].iloc[0,:] #whatever
            interval_change[categorical_names] = (interval_change[categorical_names] > 0).astype('int') # put 1-0 for categorical, easier for analysis

        out.append(interval_change.values[0])
    out = pd.DataFrame(out, columns = di.border_features.columns)
    return out

def build_discrepancy_nodes_dataset(p2g, categorical_names=None):
    """
    Output dataset: one row for each node that was in a discrepancy area.
    """
    disc_nodes = [n[1]['features'] for n in p2g.G.nodes(data=True) if n[1]['discrepancies'] == 1]
    df_nodes = pd.DataFrame(disc_nodes)
    if categorical_names is not None:
        df_nodes[categorical_names] = (df_nodes[categorical_names] > 0).astype('int')
    return df_nodes

def categories_summary(df_amp, Xtrain, categorical_names):
    """Count of categories, a comparison between intervals and Xtrain. Descriptive."""
    Xtrain_b =  (Xtrain[categorical_names] > 0).astype('int')
    Xtrain_count = Xtrain_b.mean(axis=0)
    print("Representation in the training set")
    print(Xtrain_count.sort_values(ascending=False))
    print("Representation in Amplitude dataset")
    df_amp_count = df_amp[categorical_names].mean(axis=0)
    print(df_amp_count.sort_values(ascending=False))
    print("Representation in Amplitude dataset normalized")
    print((df_amp_count - Xtrain_count).sort_values(ascending=False, key=lambda x: x.abs()))
        
def get_global_discrepancy_importances(df_amp, Xtrain, categorical_names=None, plot=False):
    """
    Returns two feature rankings of how represented the feature is in the discrepancy intervals:
    one for continuous and one for categorical attributes.
    - for continuous features, the amplitude of change is measured (e.g. Age-> 5yrs of age in average)
    - for categorical features, the representation of the feature is returned (e.g. 80% of discrepancy intervals were found for Males)
    Normalized by the presence in the Xtrain? Maybe
    N.B. rankings cannot be compared
    """
    if categorical_names is not None:
        continous_names = set(Xtrain.columns) - set(categorical_names)
    else:
        continuous_names = Xtrain.columns
    Xtrain_c =  (Xtrain[categorical_names] > 0).astype('int')
    ranking_categorical = (df_amp[categorical_names].mean(axis=0) - Xtrain_c.mean(axis=0)).sort_values(ascending=False)
    ranking_continuous = df_amp[continous_names].mean(axis=0).sort_values(ascending=False)
    
    #if plot
    
    return {'continous_features': ranking_continuous, 'categorical_features': ranking_categorical}
    
    
def interval_uncertainty(df_amp, continuous_names, metric='sum', aggregation='mean'):
    """
    Uncertainty measure associated to one or several intervals
    For now: area of the hypercube "suggested" by the rule, OR total sum of ranges
    Calculated over the continuous features only.
    """
    if metric == 'sum':
        score = df_amp.sum(axis=1)
    elif metric == 'prod':
        #Not sure: some features have 0
        score = df_amp.prod(axis=1)
    return score.mean(axis=0)
    
    
    
    
    
    
    
    
    