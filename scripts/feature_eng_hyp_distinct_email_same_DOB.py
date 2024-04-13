#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

def bin_distinct_emails_same_DOB(file_path):
    df = pd.read_csv(file_path)

    # specify the bin edges
    bin_edges = [0, 10, 20, 30, 50]

    # define bin labels
    bin_labels = ['0-10', '10-20', '20-30', '>30']

    # create the new column with binned values
    df['emails_bin'] = pd.cut(df['date_of_birth_distinct_emails_4w'], bins=bin_edges, labels=bin_labels)

    # return modified df
    return df


# In[ ]:




