# DSA4263 Group Project
Detecting fraudulent bank accounts and gaining insights into characteristics of these fraudulent accounts and generate actionable insights to detect and prevent these kind of fraud.


## About the Project
### Abstract

### Data
This project uses the Bank Account Fraud (BAF) Dataset Suit from Kaggle. For more information:
+ Kaggle:  https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?select=Base.csv
+ Datasheet: https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf

## Getting started
First, clone the repo:
```
git clone https://github.com/kclzl-nus/Detect-Fraudulent-Bank-Accounts.git
```
<br/>

Then inside the top-level directory of the repo, run 
```
pip install -r requirements.txt
```
to allow imports from other python scripts.

<br/>
Next, is to setup your KaggleAPI Token:

1. Navigate to https://www.kaggle.com/ and log into your account.
2. In Kaggle, navigate to 'Your Profile' > 'Settings' > 'Account' tab > 'API' section and select 'Create New Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials.
3. Move the `kaggle.json` file into the appropriate folder:
    + Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json`
    + Mac, Linux, OSX, and other UNIX-based OS: `~/.kaggle/kaggle.json`
<br/>

Now you are all ready to explore the project:
+ `notebooks/main.ipynb`: Notebook for project skeleton
+ `1.0-data-preprocessing.ipynb`
+ `2.0-exploratory-data-analysis.ipynb`
+ Hypothesis & Feature Engineering:
    + `3.0-hypothesis-device-activity.ipynb`
    + `3.1-hypothesis-name-email-similarity.ipynb`
    + `3.2-hypothesis-session-length.ipynb`
    + `3.3-hypothesis-clustering.ipynb`
    + `3.4-hypothesis-income-credit-limit-ratio.ipynb`
    + `3.5-hypothesis-bank-integration.ipynb`
+ Models:
    + `4.0-baseline-model.ipynb`
    + `4.1-challenger-model.ipynb`


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
