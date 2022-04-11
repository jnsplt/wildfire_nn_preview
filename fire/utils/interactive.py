import pandas as pd
from IPython.display import display


def display_all(df, all_rows=True, all_cols=True, complete_columns=True):
    context_params = []
    if all_rows:
        context_params += ['display.max_rows', None]
    if all_cols:
        context_params += ['display.max_columns', None,]
    if complete_columns:
        context_params += ['display.max_colwidth', -1]
    
    with pd.option_context(*context_params):
        display(df)