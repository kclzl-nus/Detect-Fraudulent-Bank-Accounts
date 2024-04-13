#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

def bin_session_length(file_path):
    df = pd.read_csv(file_path)

    # specify the bin edges
    bin_edges = [-10, 0, 20, 40, 60, 80, 100, 300]

    # define bin labels
    bin_labels = ['<0', '0-20', '20-40', '40-60', '60-80', '80-100', '>100']

    # create the new column with binned values
    df['session_length_bins'] = pd.cut(df['session_length_in_minutes'], bins=bin_edges, labels=bin_labels)

    # return modified df
    return df


# In[ ]:




