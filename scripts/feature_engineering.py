import sys
import pandas as pd

sys.path.append('../scripts')
import feature_eng_hyp_device_activity
import feature_eng_account_kmeans
import feature_eng_hyp_distinct_email_same_DOB
import feature_eng_hyp_name_email_similarity
import feature_eng_hyp_session_length

def feature_engineer(X_train, y_train):
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df1 = feature_eng_hyp_device_activity.generateDeviceActivityFeatures(train_df)
    train_df2 = feature_eng_account_kmeans.get_account_related_and_kmeans_feature(train_df1)
    train_df3 = feature_eng_hyp_distinct_email_same_DOB.bin_distinct_emails_same_DOB(train_df2)
    train_df4 = feature_eng_hyp_name_email_similarity.bin_name_email_similarity(train_df3)
    train_df5 = feature_eng_hyp_session_length.bin_session_length(train_df4)

    X_train_FE = train_df5.drop(columns=['fraud_bool'])
    y_train_FE = train_df5['fraud_bool']

    return X_train_FE, y_train_FE