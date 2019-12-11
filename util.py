import pandas as pd
import numpy as np

def fill_by_group(df, group_cols, value_col, strategy):
    """Fill NaN values in a pandas dataframe by group
    
    Args:
        df (pandas DataFrame): dataset
        group_cols (str or list): single column or list of columns to group by
        value_col (str): column of which value is being filled
        strategy (str): a fill strategy. May be mean, median, min, max, forward fill (ffill)
            or backwards fill (bfill)
            
    Returns:
        new_series (pandas Series): new series with values filled by group
    """
    
    strategies = {
        'mean': np.mean,
        'median': np.median,
        'min': np.min, 
        'max': np.max,
        'ffill': pd.Series.ffill,
        'bfill': pd.Series.bfill
    }
    
    try:
        assert strategy in strategies.keys(), 'must be a valid strategy'
    except:
        raise
    
    fill_fxn = strategies[strategy]
    new_series = df.groupby(group_cols)[value_col].transform(lambda x: x.fillna(fill_fxn(x)))
    
    return new_series


def create_dummy_df(df, cat_cols, dummy_na, drop_first=True, drop_orig=True):
    """Create a new dataframe with category columns one-hot-encoded.

    Args:
        df (pandas DataFrame): pandas dataframe with categorical variables you want to dummy
        cat_cols (list): list of columns to encode
        dummy_na (bool): whether you want to dummy NA vals of categorical columns or not
        drop_first (bool): whether you want to drop the first one-hot column 
        drop_orig (bool): whether you want to drop the original nonencoded cat_cols
    
    Returns:
        df (pandas DataFrame): new dataframe with one-hot-encoded columns minus cat_cols
    """

    dummy_df = pd.get_dummies(df[cat_cols], dummy_na=dummy_na, prefix_sep='_', drop_first=drop_first)
    
    if drop_orig:
        new_df = df.drop(cat_cols, axis=1).merge(dummy_df, left_index=True, right_index=True)
    else:
        new_df = df.merge(dummy_df, left_index=True, right_index=True)

    return new_df