#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

def bin_name_email_similarity(df):

    # specify the bin edges
    bin_edges = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # define bin labels
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']

    # create the new column with binned values
    df['name_email_similarity_bins'] = pd.cut(df['name_email_similarity'], bins=bin_edges, labels=bin_labels)

    # return modified df
    return df


# In[ ]:




