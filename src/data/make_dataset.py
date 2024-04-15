# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.utils import resample


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # download dataset from kaggle + store in '../../data/raw'
    # Check if 'Base.csv' exists before downloading and extracting the dataset
    if not os.path.exists('../data/raw/Base.csv'):
        # Initialize Kaggle API
        api = KaggleApi()
        # Authenticate with your Kaggle API credentials
        api.authenticate()
        # Set the name of the dataset on Kaggle
        dataset_name = 'sgpjesus/bank-account-fraud-dataset-neurips-2022'
        # Set the destination path where you want to save the dataset
        destination_path = '../data/raw'


        # Create the destination directory if it doesn't exist
        os.makedirs(destination_path, exist_ok=True)
        # Download the dataset
        api.dataset_download_files(dataset_name, path=destination_path, unzip=True)

    # read in dataset
    data = pd.read_csv('../data/raw/Base.csv')

    # convert variable to appropriate type
    data['fraud_bool'] = data['fraud_bool'].astype('category')
    data['payment_type'] = data['payment_type'].astype('category')
    data['employment_status'] = data['employment_status'].astype('category')
    data['housing_status'] = data['housing_status'].astype('category')
    data['source'] = data['source'].astype('category')
    data['device_os'] = data['device_os'].astype('category')

    # clean data
    ## remove redundant rows
    data.drop(columns=['device_fraud_count'], inplace=True)

    ## Change all negative values to -1 for `intended_balcon_amount`
    data['intended_balcon_amount'] = data['intended_balcon_amount'].apply(lambda x: -1 if x < 0 else x)

    ##  Drop rows with missing values in columns: `current_address_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`
    data = data[(data['current_address_months_count'] >= 0) & (data['session_length_in_minutes'] >= 0) & (data['device_distinct_emails_8w'] >= 0)]

    # undersample imbalanced dataset + store in '../../data/processed'
    ## Separate fraud and non-fraud data
    fraud_data = data[data['fraud_bool'] == 1]
    non_fraud_data = data[data['fraud_bool'] == 0]

    ## Undersample non-fraud data to match the size of the fraud data
    undersampled_non_fraud_data = resample(non_fraud_data, replace=False, n_samples=5*len(fraud_data), random_state=42)

    ## Combine fraud and undersampled non-fraud data
    undersampled_data = pd.concat([fraud_data, undersampled_non_fraud_data])

    ## Check if 'Base.csv' exists before downloading and extracting the dataset
    if not os.path.exists('../data/processed/undersampled_data.csv'):
        ## create the destination directory if it doesn't exist
        os.makedirs('../data/processed', exist_ok=True)

        ## save the undersampled data to a CSV file
        undersampled_data.to_csv('../data/processed/undersampled_data.csv', index=False)

    # split dataset into train, validation and test set + store in '../../data/processed'


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
