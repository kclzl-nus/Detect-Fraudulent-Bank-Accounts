# import other libraries
# import libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def generateDeviceActivityFeatures(data):

    # # count fraud and non-fraud
    # # total_fraud_count = data['fraud_bool'].value_counts().to_frame().loc[1, "count"]
    # # total_non_fraud_count = data['fraud_bool'].value_counts().to_frame().loc[0, "count"]
    # helper function to extract fraud proportion for feature engineering
    # def getFraudInfo(group):
    #     # get total count of fraud within this group
    #     fraud_count = group['fraud_bool'].value_counts().to_frame().loc[1, "count"]
    #     total_count = group.shape[0]
    #     fraud_proportion = round(fraud_count / total_count, 4)

    #     group['num_fraud'] = fraud_count
    #     group['proportion_of_fraud_in_group'] = fraud_proportion
    #     return group[['num_fraud', 'proportion_of_fraud_in_group']].drop_duplicates()

    #############
    # Feature 1 #
    #############

    # split data
    group01 = data[(data['device_os'] == 'windows') & (data['device_distinct_emails_8w'].isin([0, 2]))]
    group02 = data[(data['device_os'] == 'macintosh') & (data['device_distinct_emails_8w'].isin([0, 2]))]
    group03 = data[(data['device_os'] == 'x11') & (data['device_distinct_emails_8w'].isin([0, 2]))]
    group04 = data[(data['device_os'] == 'other') & (data['device_distinct_emails_8w'].isin([0, 2]))]
    # group05 will be all the groups except those in group01, group02, group03, group04
    group05 = data[~data.index.isin(group01.index) & ~data.index.isin(group02.index) & ~data.index.isin(group03.index) & ~data.index.isin(group04.index)]

    # combine into 1 dataframe, add label to that dataframe, label each group 'A', 'B', 'C', 'D', 'E', all in new column 'FE_01'
    group01['FE_01'] = 'A'; group02['FE_01'] = 'B'; group03['FE_01'] = 'C'; group04['FE_01'] = 'D'; group05['FE_01'] = 'E'

    # add back data
    data = pd.concat([group01, group02, group03, group04, group05])

    # change 'FE_01' to category
    data['FE_01'] = data['FE_01'].astype('category')

    FE_01_prob_mapping = {"A": 0.6047,
                        "B": 0.4529,
                        "C": 0.3846,
                        "D": 0.2978,
                        "E": 0.1564}

    # map the probability of fraud to the device_acitivtiy_df, as a new column 'FE_01_device_os_emails_prob'
    data['FE_01_device_os_emails_prob'] = data['FE_01'].map(FE_01_prob_mapping)

    #print("Feature 1 created.")


    #############
    # Feature 2 #
    #############

    group01 = data[(data['keep_alive_session'] == 0) & (data['device_distinct_emails_8w'] == 0)]
    group02 = data[(data['keep_alive_session'] == 0) & (data['device_distinct_emails_8w'] == 1)]
    group03 = data[(data['keep_alive_session'] == 0) & (data['device_distinct_emails_8w'] == 2)]
    group04 = data[(data['keep_alive_session'] == 1) & (data['device_distinct_emails_8w'] == 0)]
    group05 = data[(data['keep_alive_session'] == 1) & (data['device_distinct_emails_8w'] == 1)]
    group06 = data[(data['keep_alive_session'] == 1) & (data['device_distinct_emails_8w'] == 2)]

    # label each group from "A" to "F"
    group01['FE_02'] = "A"; group02['FE_02'] = "B"; group03['FE_02'] = "C"; group04['FE_02'] = "D"; group05['FE_02'] = "E"; group06['FE_02'] = "F"

    # concatenate all the group
    data = pd.concat([group01, group02, group03, group04, group05, group06])

    # change 'FE_02' to category
    data['FE_02'] = data['FE_02'].astype('category')

    # generate mapping
    FE_02_prob_mappping = {"A": 0.3520,
                        "B": 0.2230,
                        "C": 0.4732,
                        "D": 0.1591,
                        "E": 0.1016,
                        "F": 0.3131
                        }

    # map the probability of fraud to the device_acitivtiy_df, as a new column 'FE_01_device_os_emails_prob'
    data['FE_02_keep_alive_device_emails_prob'] = data['FE_02'].map(FE_02_prob_mappping)

    #print("Feature 1 created.")

    #############
    # Feature 3 #
    #############

    group01 = data[(data['source'] == "INTERNET") & (data['foreign_request'] == 0)]
    group02 = data[(data['source'] == "INTERNET") & (data['foreign_request'] == 1)]
    group04 = data[(data['source'] == "TELEAPP") & (data['foreign_request'] == 0)]
    group03 = data[(data['source'] == "TELEAPP") & (data['foreign_request'] == 1)]

    # label each group from "A" to "D", label_name = "FE_03"
    group01['FE_03'] = "A"; group02['FE_03'] = "B"; group03['FE_03'] = "C"; group04['FE_03'] = "D"


    # concatenate all the group
    data = pd.concat([group01, group02, group03, group04])

    # change 'FE_03' to category
    data['FE_03'] = data['FE_03'].astype('category')

    # generate mapping
    FE_03_prob_mappping = {"A": 0.1627,
                        "B": 0.2782,
                        "C": 0.2448,
                        "D": 0.18
                        }

    # map the probability of fraud to the device_acitivtiy_df, as a new column
    data['FE_03_source_foreign_request_prob'] = data['FE_03'].map(FE_03_prob_mappping)

    #print('Feature 3 created.')

    #############
    # Feature 4 #
    #############

    FE_04_prob_mappping = {"windows": 0.3083,
                        "macintosh": 0.2041,
                        "linux": 0.0001,
                        "other": 0.0001,
                        "x11": 0.0001
                        }

    # map the probability of fraud to the device_acitivtiy_df, as a new column
    data['FE_04_device_os_prob'] = data['device_os'].map(FE_04_prob_mappping)

    #print('Feature 4 created.')


    ###########
    # Tidy Up #
    ###########

    # drop labelled columns 'FE_01' to 'FE_04'
    data.drop(columns=['FE_01','FE_02','FE_03'], inplace=True)

    # change all other feature engineered columns to float instead of category
    data['FE_01_device_os_emails_prob'] = data['FE_01_device_os_emails_prob'].astype('float')
    data['FE_02_keep_alive_device_emails_prob'] = data['FE_02_keep_alive_device_emails_prob'].astype('float')
    data['FE_03_source_foreign_request_prob'] = data['FE_03_source_foreign_request_prob'].astype('float')
    data['FE_04_device_os_prob'] = data['FE_04_device_os_prob'].astype('float')

    #print("Features for Device Activity Hypothesis generated.")

    return data

# # uncomment to test code
# X_train = pd.read_csv("../data/processed/X_train.csv")
# y_train = pd.read_csv("../data/processed/y_train.csv")
# data = pd.concat([y_train, X_train], axis=1)
# print(pd.concat([y_train, X_train], axis=1)['fraud_bool'].value_counts().to_frame().loc[1:, "count"])
# data = generateDeviceActivityFeatures(data)

# # check shape and head
# print(data.shape)
# print(data.head())