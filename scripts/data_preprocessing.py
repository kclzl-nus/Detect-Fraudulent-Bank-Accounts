import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def data_preprocessing():
    # Data Downloading
    # Check if 'Base.csv' exists before downloading and extracting the dataset
    if not os.path.exists('../data/raw/Base.csv'):
        print("Base.csv not found. Downloading the dataset from Kaggle...")

        # Instantiate the Kaggle API client
        api = KaggleApi()

        # Authenticate using the configured credentials
        api.authenticate()

        # Set the name of the dataset on Kaggle
        dataset_name = 'sgpjesus/bank-account-fraud-dataset-neurips-2022'
        # Set the destination path where you want to save the dataset
        destination_path = '../data/raw'


        # Create the destination directory if it doesn't exist
        os.makedirs(destination_path, exist_ok=True)
        # Download the dataset
        api.dataset_download_files(dataset_name, path=destination_path, unzip=True)
        
        # Define the path to the dataset zip file and the directory for extraction
        zip_file_path = '../data/raw/bank-account-fraud-dataset-neurips-2022.zip'
        # Check if the dataset zip file exists
        if os.path.exists(zip_file_path):
            # Extract the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall('../data/raw/')

        print("Base.csv downloaded and extracted successfully. Data saved to 'data/raw/Base.csv'.")
    else:
        print("Base.csv already present.")


    print("Data Preprocessing...")

    # Data Preprocess
    data = pd.read_csv('../data/raw/Base.csv')

    # Convert the variables to the appropriate data types
    data['fraud_bool'] = data['fraud_bool'].astype('category')
    data['payment_type'] = data['payment_type'].astype('category')
    data['employment_status'] = data['employment_status'].astype('category')
    data['housing_status'] = data['housing_status'].astype('category')
    data['source'] = data['source'].astype('category')
    data['device_os'] = data['device_os'].astype('category')

    # Remove Redundant rows
    data.drop(columns=['device_fraud_count'], inplace=True)

    # Data Cleaning
    data['intended_balcon_amount'] = data['intended_balcon_amount'].apply(lambda x: -1 if x < 0 else x)
    data = data[(data['current_address_months_count'] >= 0) & (data['session_length_in_minutes'] >= 0) & (data['device_distinct_emails_8w'] >= 0)]

    # Ratio of 1 to 5
    # Separate fraud and non-fraud data
    fraud_data = data[data['fraud_bool'] == 1]
    non_fraud_data = data[data['fraud_bool'] == 0]

    # Undersample non-fraud data to match the size of the fraud data
    undersampled_non_fraud_data = resample(non_fraud_data, replace=False, n_samples=5*len(fraud_data), random_state=42)

    # Combine fraud and undersampled non-fraud data
    undersampled_data = pd.concat([fraud_data, undersampled_non_fraud_data])

    ## Save to CSV
    # Create the destination directory if it doesn't exist
    os.makedirs('../data/processed', exist_ok=True)

    # save the undersampled data to a CSV file
    undersampled_data.to_csv('../data/processed/undersampled_data.csv', index=False)

    print("Data Preprocessing completed successfully. Data saved to 'data/processed/undersampled_data.csv'.")

    print("Splitting the dataset into train-test subset...")

    ## Splitting the dataset into train-test subset
    X = undersampled_data.drop(columns=['fraud_bool'])
    y = undersampled_data['fraud_bool']

    # Splitting the dataset into training and test sets with an 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the training and test sets to CSV files
    X_train.to_csv('../data/processed/X_train.csv', index=False)
    X_test.to_csv('../data/processed/X_test.csv', index=False)
    y_train.to_csv('../data/processed/y_train.csv', index=False)
    y_test.to_csv('../data/processed/y_test.csv', index=False)

    print("Train-test split completed successfully. Data saved to 'data/processed/X_train.csv', 'data/processed/X_test.csv', 'data/processed/y_train.csv', and 'data/processed/y_test.csv'.")