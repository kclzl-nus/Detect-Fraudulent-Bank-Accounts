import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Data Downloading
# Check if 'Base.csv' exists before downloading and extracting the dataset
if not os.path.exists('../data/raw/Base.csv'):
    # Instantiate the Kaggle API client
    api = KaggleApi()
    # Will Deactivate credentials after course ends
    api.set_config_value('username', 'khanseem')
    api.set_config_value('key', '705a18cbbd7e4d23d72450605243b00f')
    # Authenticate using the configured credentials
    api.authenticate()
    # Download the dataset
    api.dataset_download_files(dataset='sgpjesus/bank-account-fraud-dataset-neurips-2022', path='../data/raw', unzip=True)
    
    # Define the path to the dataset zip file and the directory for extraction
    zip_file_path = '../data/raw/bank-account-fraud-dataset-neurips-2022.zip'
    # Check if the dataset zip file exists
    if os.path.exists(zip_file_path):
        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall('../data/raw/')

# Data Preprocess
data = pd.read_csv('../data/raw/Base.csv')
# Data Cleaning
data['intended_balcon_amount'] = data['intended_balcon_amount'].apply(lambda x: -1 if x < 0 else x)
data = data[(data['current_address_months_count'] >= 0) & (data['session_length_in_minutes'] >= 0) & (data['device_distinct_emails_8w'] >= 0)]

# Convert the variables to the appropriate data types
data['fraud_bool'] = data['fraud_bool'].astype('category')
data['payment_type'] = data['payment_type'].astype('category')
data['employment_status'] = data['employment_status'].astype('category')
data['email_is_free'] = data['email_is_free'].astype('uint8')
data['housing_status'] = data['housing_status'].astype('category')
data['phone_home_valid'] = data['phone_home_valid'].astype('uint8')
data['phone_mobile_valid'] = data['phone_mobile_valid'].astype('uint8')
data['has_other_cards'] = data['has_other_cards'].astype('uint8')
data['foreign_request'] = data['foreign_request'].astype('uint8')
data['source'] = data['source'].astype('category')
data['device_os'] = data['device_os'].astype('category')

# Remove Redundant rows
data.drop(columns=['device_fraud_count'], inplace=True)

# Ratio of 1 to 5
# Separate fraud and non-fraud data
fraud_data = data[data['fraud_bool'] == 1]
non_fraud_data = data[data['fraud_bool'] == 0]

# Undersample non-fraud data to match the size of the fraud data
undersampled_non_fraud_data = resample(non_fraud_data, replace=False, n_samples=5*len(fraud_data), random_state=42)

# Combine fraud and undersampled non-fraud data
undersampled_data = pd.concat([fraud_data, undersampled_non_fraud_data])

#Save to CSV
undersampled_data.to_csv('../data/processed/undersampled_data.csv', index=False)