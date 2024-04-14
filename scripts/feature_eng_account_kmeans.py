import pandas as pd
import numpy as np
import pickle


def get_account_related_and_kmeans_feature(df):

    df['have_prev_address'] = np.where(df['prev_address_months_count']<0, 0, 1)

    df['have_initial_bal'] = np.where(df['intended_balcon_amount']<=0, 0, 1)

    df['bank_integration'] = df['bank_months_count'] + df['has_other_cards']*df['bank_months_count']

    df['income_credit_limit_ratio'] = df['income'] / df['proposed_credit_limit']

    # standardize features before labelling

    standardisation_dict = {'mean': {'income': 0.5814915870850601,
                            'name_email_similarity': 0.47785263986116844,
                            'current_address_months_count': 91.47064953766863,
                            'customer_age': 34.8747536759133,
                            'days_since_request': 1.0153656270433704,
                            'zip_count_4w': 1576.1575716234652,
                            'bank_branch_count_8w': 174.3052144914355,
                            'date_of_birth_distinct_emails_4w': 9.17839548279521,
                            'credit_risk_score': 138.77993785053812,
                            'bank_months_count': 10.781188418978324,
                            'proposed_credit_limit': 567.947741397605,
                            'device_distinct_emails_8w': 1.03014627861149,
                            'bank_integration': 12.943970744277703,
                            'income_credit_limit_ratio': 0.0018832795126628803},
                            'std': {'income': 0.2902300165003485,
                            'name_email_similarity': 0.29243564444475884,
                            'current_address_months_count': 88.76399577114984,
                            'customer_age': 12.477604473322588,
                            'days_since_request': 5.4101607134746565,
                            'zip_count_4w': 1002.8973173533811,
                            'bank_branch_count_8w': 448.88471035979137,
                            'date_of_birth_distinct_emails_4w': 5.08265253845966,
                            'credit_risk_score': 73.73541008299814,
                            'bank_months_count': 12.241200719369152,
                            'proposed_credit_limit': 528.6851098964391,
                            'device_distinct_emails_8w': 0.20384914080535405,
                            'bank_integration': 15.93797506583502,
                            'income_credit_limit_ratio': 0.001515515820328237}}
    cluster_features = [
            'fraud_bool',
            'income',
            'name_email_similarity',
            'current_address_months_count',
            'customer_age',
            'days_since_request',
            'payment_type',
            'zip_count_4w',
            'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w',
            'employment_status',
            'credit_risk_score',
            'email_is_free',
            'housing_status',
            'phone_home_valid',
            'phone_mobile_valid',
            'bank_months_count', 
            'has_other_cards',
            'proposed_credit_limit',
            'foreign_request',
            'device_os',
            'device_distinct_emails_8w',
            'bank_integration',
            'income_credit_limit_ratio',
            'have_prev_address',
            'have_initial_bal'
                ]
    numerical_features = [
            'income',
            'name_email_similarity',
            'current_address_months_count',
            'customer_age',
            'days_since_request',
            'zip_count_4w',
            'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w',
            'credit_risk_score',
            'bank_months_count', 
            'proposed_credit_limit',
            'device_distinct_emails_8w',
            'bank_integration',
            'income_credit_limit_ratio',
                ]
    df_scaled = df[cluster_features]

    for col in numerical_features:
        mean = standardisation_dict['mean'][col] 
        std = standardisation_dict['std'][col]
        df_scaled[col] = (df_scaled[col] - mean) / std 

    # kmeans model labelling

    kmeans_model = pickle.load(open('../models/kmeans_model.pkl', 'rb'))

    categorical_features = [
            'payment_type',
            'employment_status',
            'email_is_free',
            'housing_status',
            'phone_home_valid',
            'phone_mobile_valid',
            'has_other_cards',
            'foreign_request',
            'device_os',
            'have_prev_address',
            'have_initial_bal',
                ]
    df_cluster = pd.get_dummies(df_scaled, columns=categorical_features)

    labels = kmeans_model.predict(df_cluster)

    df_scaled['kmeans_label'] = labels

    # use labelling for feature engineering based on proportion of fraud cases

    df_cluster = df_scaled

    condition_list = [
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) & (df_cluster['proposed_credit_limit']>=2.474),
        (df_cluster['housing_status']=='BA') & (df_cluster['kmeans_label']==3),
        (df_cluster['kmeans_label']==3) & (df_cluster['housing_status']=='BA'),
        (df_cluster['kmeans_label']==3) & (df_cluster['credit_risk_score']>=0.792),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) &(df_cluster['housing_status']=='BA') &(df_cluster['bank_months_count']<=-0.91),
        (df_cluster['device_os']=='windows') &(df_cluster['kmeans_label'].isin([1])) & (df_cluster['housing_status']=='BA') & (df_cluster['bank_months_count']<=-0.91),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) &(df_cluster['housing_status']=='BA') & (df_cluster['payment_type'].isin(['AC'])),
        (df_cluster['device_os']=='windows') &(df_cluster['kmeans_label'].isin([1])) &(df_cluster['housing_status']=='BA') &(df_cluster['income']>=1),
        (df_cluster['device_os'] == 'windows') &   (df_cluster['kmeans_label'].isin([1])) & (df_cluster['housing_status'].isin(['BA'])) &   (df_cluster['phone_home_valid'].isin([0])),
        (df_cluster['kmeans_label'].isin([1])) & (df_cluster['phone_home_valid']==0) & (df_cluster['date_of_birth_distinct_emails_4w']<=-1.404),
        (df_cluster['kmeans_label']==3) & (df_cluster['date_of_birth_distinct_emails_4w']< -1.4),
        (df_cluster['device_os']=='windows') &(df_cluster['kmeans_label'].isin([1])) & (df_cluster['housing_status']=='BA') &(df_cluster['customer_age']> 1),
        (df_cluster['device_os'] == 'windows') &  (df_cluster['kmeans_label'].isin([1])) & (df_cluster['housing_status'].isin(['BA'])) & (df_cluster['has_other_cards'].isin([0])),
        (df_cluster['kmeans_label']==3) & (df_cluster['device_os']=='windows'),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label']==3),
        (df_cluster['kmeans_label'].isin([1])) & (df_cluster['has_other_cards']==0) & (df_cluster['credit_risk_score']>=2.2),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) & (df_cluster['date_of_birth_distinct_emails_4w'] < -1.215),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) & (df_cluster['credit_risk_score']>=1.6),
        (df_cluster['device_os']=='windows') &(df_cluster['kmeans_label'].isin([1])) &(df_cluster['housing_status']=='BA') & (df_cluster['date_of_birth_distinct_emails_4w']<=-0.035),
        (df_cluster['device_os']=='windows') &(df_cluster['kmeans_label'].isin([1])) & (df_cluster['housing_status']=='BA') & (df_cluster['bank_branch_count_8w']<=0.19),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) & (df_cluster['income']>1),
        (df_cluster['kmeans_label'].isin([1])) & (df_cluster['has_other_cards']==0) & (df_cluster['date_of_birth_distinct_emails_4w']<=-1.404),
        (df_cluster['kmeans_label']==3) & (df_cluster['income']>1),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) & (df_cluster['housing_status']=='BA'),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) & (df_cluster['payment_type'].isin(['AC'])),
       (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) & (df_cluster['bank_months_count'] <= -0.91),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) & (df_cluster['bank_branch_count_8w']<-0.386),
        (df_cluster['credit_risk_score']>=2.2),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) & (df_cluster['phone_home_valid']==0),
        (df_cluster['kmeans_label'].isin([1])) & (df_cluster['phone_home_valid']==0) & (df_cluster['device_os']=='windows'),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])) & (df_cluster['email_is_free']==1),
        (df_cluster['kmeans_label'].isin([1])) & (df_cluster['phone_home_valid']==0) & (df_cluster['housing_status']=='BA'),
       (df_cluster['date_of_birth_distinct_emails_4w']<=-1.404) & (df_cluster['kmeans_label']==1),
        (df_cluster['device_os']=='windows') & (df_cluster['kmeans_label'].isin([1])),
        (df_cluster['kmeans_label']==3),
       (df_cluster['phone_home_valid']==0) & (df_cluster['kmeans_label'].isin([1])),
        (df_cluster['kmeans_label'].isin([1])) & (df_cluster['has_other_cards']==0),
        (df_cluster['kmeans_label']==4) & (df_cluster['housing_status']=='BA'),
       (df_cluster['kmeans_label']==1),
       (df_cluster['device_os']=='windows') ,
        (df_cluster['kmeans_label']==2) & (df_cluster['housing_status']=='BA'),
        (df_cluster['kmeans_label']==0) & (df_cluster['housing_status']=='BA'),
        (df_cluster['kmeans_label']==4),
        (df_cluster['kmeans_label']==2),
        (df_cluster['kmeans_label']==0)
                   ]

    prob_list = [0.802345059,
                0.731907895,
                0.731907895,
                0.72826087,
                0.688747731,
                0.688747731,
                0.68867083,
                0.675824176,
                0.652659812,
                0.636968085,
                0.631578947,
                0.622473246,
                0.621749409,
                0.621428571,
                0.621428571,
                0.620403321,
                0.60994561,
                0.603804348,
                0.600875593,
                0.594287569,
                0.582660433,
                0.581125828,
                0.581081081,
                0.579007743,
                0.576676385,
                0.572434018,
                0.571984436,
                0.57054126,
                0.550396376,
                0.550396376,
                0.530743347,
                0.528811087,
                0.52045607,
                0.461364427,
                0.439457203,
                0.387566845,
                0.365830916,
                0.361952101,
                0.313652839,
                0.307911111,
                0.30556813,
                0.257042254,
                0.134435711,
                0.10993009,
                0.066630107
                ]
    df_cluster['kmeans_prob'] = np.select(condition_list, prob_list, 0)
    df = pd.concat([df, df_cluster[['kmeans_prob']]], axis=1)
    
    return df
    