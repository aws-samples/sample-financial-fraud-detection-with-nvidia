# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Credit Card Transaction Data Cleanup and Prep
#
# This source code shows the steps for cleanup and preparing the credit card transaction data for training models with Training NIM.
#
# ### The dataset:
#  * IBM TabFormer: https://github.com/IBM/TabFormer
#  * Released under an Apache 2.0 license
#
# Contains 24M records with 15 fields, one field being the "is fraud" label which we use for training.
#
# ### Goals
# The goal is to:
#  * Cleanup the data
#    * Make field names just single word
#      * while field names are not used within the GNN, it makes accessing fields easier during cleanup
#    * Encode categorical fields
#      * use one-hot encoding for fields with less than 8 categories
#      * use binary encoding for fields with more than 8 categories
#    * Create a continuous node index across users, merchants, and transactions
#      * having node ID start at zero and then be contiguous is critical for creation of Compressed Sparse Row (CSR) formatted data without wasting memory.
#  * Produce:
#    * For XGBoost:
#      * Training   - all data before 2018
#      * Validation - all data during 2018
#      * Test.      - all data after 2018
#    * For GNN
#      * Training Data
#        * Edge List
#        * Feature data
#    * Test set - all data after 2018
#
#
#
# ### Graph formation
# Given that we are limited to just the data in the transaction file, the ideal model would be to have a bipartite graph of Users to Merchants where the edges represent the credit card transaction and then perform Link Classification on the Edges to identify fraud. Unfortunately the current version of cuGraph does not support GNN Link Prediction. That limitation will be lifted over the next few release at which time this code will be updated. Luckily, there is precedence for viewing transactions as nodes and then doing node classification using the popular GraphSAGE GNN. That is the approach this code takes. The produced graph will be a tri-partite graph where each transaction is represented as a node.
#
# <img src="../img/3-partite.jpg" width="35%"/>
#
#
# ### Features
# For the XGBoost approach, there is no need to generate empty features for the Merchants. However, for GNN processing, every node needs to have the same set of feature data. Therefore, we need to generate empty features for the User and Merchant nodes.
#
# -----

# #### Import the necessary libraries.  In this case will be use cuDF and perform most of the data prep in GPU
#


import json
import os

import cudf
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.linalg import block_diag
from category_encoders import BinaryEncoder
from scipy.stats import pointbiserialr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


COL_USER = "User"
COL_CARD = "Card"
COL_AMOUNT = "Amount"
COL_MCC = "MCC"
COL_TIME = "Time"
COL_DAY = "Day"
COL_MONTH = "Month"
COL_YEAR = "Year"

COL_MERCHANT = "Merchant"
COL_STATE = "State"
COL_CITY = "City"
COL_ZIP = "Zip"
COL_ERROR = "Errors"
COL_CHIP = "Chip"
COL_FRAUD = "Fraud"
COL_TRANSACTION_ID = "Tx_ID"
COL_MERCHANT_ID = "Merchant_ID"
COL_USER_ID = "User_ID"

UNKNOWN_STRING_MARKER = "XX"
UNKNOWN_ZIP_CODE = 0

COL_GRAPH_SRC = "src"
COL_GRAPH_DST = "dst"
COL_GRAPH_WEIGHT = "wgt"
MERCHANT_AND_USER_COLS = [COL_MERCHANT, COL_CARD, COL_MCC]


# https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient
# Use Point-biserial correlation coefficient(rpb) to check if the numerical columns are important to predict if a transaction is fraud


def cramers_v(x, y):
    """ "
    Compute correlation of categorical field x with target y.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r's_V
    """
    confusion_matrix = cudf.crosstab(x, y).to_numpy()
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))


def create_feature_mask(columns):
    # Dictionary to store mapping from original column to mask value
    mask_mapping = {}
    mask_values = []
    current_mask = 0

    for col in columns:
        # For encoded columns, assume the base is before the underscore
        if "_" in col:
            base_feature = col.split("_")[0]
        else:
            base_feature = col  # For non-encoded columns, use the column name directly

        # Assign a new mask value if this base feature hasn't been seen before
        if base_feature not in mask_mapping:
            mask_mapping[base_feature] = current_mask
            current_mask += 1

        # Append the mask value for this column
        mask_values.append(mask_mapping[base_feature])

    # Convert list to numpy array for further processing if needed
    feature_mask = np.array(mask_values)

    return mask_mapping, feature_mask


def preprocess_data(tabformer_base_path):

    # Whether we should under-sample majority class (i.e. non-fraud transactions)
    under_sample = True

    # Ration of fraud and non-fraud transactions in case we under-sample the majority class
    fraud_ratio = 0.1

    tabformer_raw_file_path = os.path.join(
        tabformer_base_path, "raw", "card_transaction.v1.csv"
    )
    tabformer_xgb = os.path.join(tabformer_base_path, "xgb")
    tabformer_gnn = os.path.join(tabformer_base_path, "gnn")

    if not os.path.exists(tabformer_xgb):
        os.makedirs(tabformer_xgb)
    if not os.path.exists(tabformer_gnn):
        os.makedirs(tabformer_gnn)

    # Read the dataset

    data = cudf.read_csv(tabformer_raw_file_path)

    # ##### Save a few transactions before any operations on data

    # # Write a few raw transactions for model's inference
    # out_path = os.path.join(tabformer_xgb, "example_transactions.csv")
    # data.tail(10).to_pandas().to_csv(out_path, header=True, index=False)

    _ = data.rename(
        columns={
            "Merchant Name": COL_MERCHANT,
            "Merchant State": COL_STATE,
            "Merchant City": COL_CITY,
            "Errors?": COL_ERROR,
            "Use Chip": COL_CHIP,
            "Is Fraud?": COL_FRAUD,
        },
        inplace=True,
    )

    # #### Handle missing values
    # * Zip codes are numeral, replace missing zip codes by 0
    # * State and Error are string, replace missing values by marker 'XX'

    # Make sure that 'XX' doesn't exist in State and Error field before we replace missing values by 'XX'
    assert UNKNOWN_STRING_MARKER not in set(data[COL_STATE].unique().to_pandas())
    assert UNKNOWN_STRING_MARKER not in set(data[COL_ERROR].unique().to_pandas())

    # Make sure that 0 or 0.0 doesn't exist in Zip field before we replace missing values by 0
    assert float(0) not in set(data[COL_ZIP].unique().to_pandas())
    assert 0 not in set(data[COL_ZIP].unique().to_pandas())

    # Replace missing values with markers
    data[COL_STATE] = data[COL_STATE].fillna(UNKNOWN_STRING_MARKER)
    data[COL_ERROR] = data[COL_ERROR].fillna(UNKNOWN_STRING_MARKER)
    data[COL_ZIP] = data[COL_ZIP].fillna(UNKNOWN_ZIP_CODE)

    # There shouldn't be any missing values in the data now.
    assert data.isnull().sum().sum() == 0

    # ### Clean up the Amount field
    # * Drop the "$" from the Amount field and then convert from string to float
    # * Look into spread of Amount and choose right scaler for it

    # Drop the "$" from the Amount field and then convert from string to float
    data[COL_AMOUNT] = data[COL_AMOUNT].str.replace("$", "").astype("float")

    # #### Change the 'Fraud' values to be integer where
    #   * 1 == Fraud
    #   * 0 == Non-fraud

    fraud_to_binary = {"No": 0, "Yes": 1}
    data[COL_FRAUD] = data[COL_FRAUD].map(fraud_to_binary).astype("int8")

    # Remove ',' in error descriptions
    data[COL_ERROR] = data[COL_ERROR].str.replace(",", "")

    # Split the time column into hours and minutes and then cast to int32
    T = data[COL_TIME].str.split(":", expand=True)
    T[0] = T[0].astype("int32")
    T[1] = T[1].astype("int32")

    # replace the 'Time' column with the new columns
    data[COL_TIME] = (T[0] * 60) + T[1]
    data[COL_TIME] = data[COL_TIME].astype("int32")

    # Delete temporary DataFrame
    del T

    # #### Convert Merchant column to str type
    data[COL_MERCHANT] = data[COL_MERCHANT].astype("str")
    max_nr_cards_per_user = len(data[COL_CARD].unique())

    # Combine User and Card to generate unique numbers
    data[COL_CARD] = data[COL_USER] * len(data[COL_CARD].unique()) + data[COL_CARD]
    data[COL_CARD] = data[COL_CARD].astype("int")

    # Collect unique merchant, card and MCC in a dataframe and fit a binary transformer
    data = data.to_pandas()

    data_ids = pd.DataFrame()

    nr_unique_card = data[COL_CARD].unique().shape[0]
    nr_unique_merchant = data[COL_MERCHANT].unique().shape[0]
    nr_unique_mcc = data[COL_MCC].unique().shape[0]
    nr_elements = max(nr_unique_merchant, nr_unique_card)

    data_ids[COL_CARD] = [data[COL_CARD][0]] * nr_elements
    data_ids[COL_MERCHANT] = [data[COL_MERCHANT][0]] * nr_elements
    data_ids[COL_MCC] = [data[COL_MCC][0]] * nr_elements

    data_ids.loc[np.arange(nr_unique_card), COL_CARD] = data[COL_CARD].unique()
    data_ids.loc[np.arange(nr_unique_merchant), COL_MERCHANT] = data[
        COL_MERCHANT
    ].unique()
    data_ids.loc[np.arange(nr_unique_mcc), COL_MCC] = data[COL_MCC].unique()

    data_ids = data_ids[MERCHANT_AND_USER_COLS].astype("category")

    id_bin_encoder = Pipeline(
        steps=[
            ("binary", BinaryEncoder(handle_missing="value", handle_unknown="value"))
        ]
    )

    id_transformer = ColumnTransformer(
        transformers=[
            ("binary", id_bin_encoder, MERCHANT_AND_USER_COLS),
        ],
        remainder="passthrough",
    )

    pd.set_option("future.no_silent_downcasting", True)
    id_transformer = id_transformer.fit(data_ids)

    preprocessed_id_data_raw = id_transformer.transform(
        data[MERCHANT_AND_USER_COLS].astype("category")
    )

    # transformed column names
    columns_of_transformed_id_data = list(
        map(
            lambda name: name.split("__")[1],
            list(id_transformer.get_feature_names_out(MERCHANT_AND_USER_COLS)),
        )
    )

    # data type of transformed columns
    id_col_type_mapping = {}
    for col in columns_of_transformed_id_data:
        if col.split("_")[0] in MERCHANT_AND_USER_COLS:
            id_col_type_mapping[col] = "int8"

    assert data_ids.isnull().sum().sum() == 0

    preprocessed_id_data = pd.DataFrame(
        preprocessed_id_data_raw, columns=columns_of_transformed_id_data
    )

    del data_ids
    del preprocessed_id_data_raw

    data = pd.concat(
        [data.reset_index(drop=True), preprocessed_id_data.reset_index(drop=True)],
        axis=1,
    )

    # ##### Compute correlation of different fields with target
    sparse_factor = 1
    columns_to_compute_corr = [
        COL_CARD,
        COL_CHIP,
        COL_ERROR,
        COL_STATE,
        COL_CITY,
        COL_ZIP,
        COL_MCC,
        COL_MERCHANT,
        COL_USER,
        COL_DAY,
        COL_MONTH,
        COL_YEAR,
    ]
    for c1 in columns_to_compute_corr:
        for c2 in [COL_FRAUD]:
            coff = 100 * cramers_v(data[c1][::sparse_factor], data[c2][::sparse_factor])
            print("Correlation ({}, {}) = {:6.2f}%".format(c1, c2, coff))

    # ### Correlation of target with numerical columns

    for col in [COL_TIME, COL_AMOUNT]:
        r_pb, p_value = pointbiserialr(
            # data[COL_FRAUD].to_pandas(), data[col].to_pandas()
            data[COL_FRAUD],
            data[col],
        )
        print("r_pb ({}) = {:3.2f} with p_value {:3.2f}".format(col, r_pb, p_value))

    numerical_predictors = [COL_AMOUNT]
    nominal_predictors = [
        COL_ERROR,
        COL_CARD,
        COL_CHIP,
        COL_CITY,
        COL_ZIP,
        COL_MCC,
        COL_MERCHANT,
    ]

    predictor_columns = numerical_predictors + nominal_predictors
    target_column = [COL_FRAUD]

    # #### Remove duplicates non-fraud data points

    # Remove duplicates data points
    fraud_data = data[data[COL_FRAUD] == 1]
    data = data[data[COL_FRAUD] == 0]

    data = data.drop_duplicates(subset=nominal_predictors)
    data = pd.concat([data, fraud_data])

    # ### Split the data into
    # The data will be split into thee groups based on event date
    #  * Training   - all data before 2018
    #  * Validation - all data during 2018
    #  * Test.      - all data after 2018

    if under_sample:
        fraud_df = data[data[COL_FRAUD] == 1]
        non_fraud_df = data[data[COL_FRAUD] == 0]
        nr_non_fraud_samples = min(
            (len(data) - len(fraud_df)), int(len(fraud_df) / fraud_ratio)
        )
        data = pd.concat(
            [fraud_df, non_fraud_df.sample(nr_non_fraud_samples, random_state=42)]
        )

    predictor_columns = list(set(predictor_columns) - set(MERCHANT_AND_USER_COLS))
    nominal_predictors = list(set(nominal_predictors) - set(MERCHANT_AND_USER_COLS))

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    training_idx = data[COL_YEAR] < 2018
    validation_idx = data[COL_YEAR] == 2018
    test_idx = data[COL_YEAR] > 2018

    # ### Scale numerical columns and encode categorical columns of training data

    # As some of the encoder we want to use is not available in cuml, we can use pandas for now.
    # Move training data to pandas for preprocessing
    pdf_training = data[training_idx][predictor_columns + target_column]

    # Use one-hot encoding for columns with <= 8 categories, and binary encoding for columns with more categories
    columns_for_binary_encoding = []
    columns_for_one_hot_encoding = []
    for col in nominal_predictors:
        if len(data[col].unique()) <= 8:
            columns_for_one_hot_encoding.append(col)
        else:
            columns_for_binary_encoding.append(col)

    assert (training_idx.sum() + validation_idx.sum() + test_idx.sum()) == data.shape[0]

    # Mark categorical column as "category"
    pdf_training[nominal_predictors] = pdf_training[nominal_predictors].astype(
        "category"
    )

    # encoders to encode categorical columns and scalers to scale numerical columns

    bin_encoder = Pipeline(
        steps=[
            ("binary", BinaryEncoder(handle_missing="value", handle_unknown="value"))
        ]
    )
    one_hot_encoder = Pipeline(steps=[("onehot", OneHotEncoder())])

    robust_scaler = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("robust", RobustScaler()),
        ],
    )

    # compose encoders and scalers in a column transformer
    transformer = ColumnTransformer(
        transformers=[
            ("binary", bin_encoder, columns_for_binary_encoding),
            ("onehot", one_hot_encoder, columns_for_one_hot_encoding),
            ("robust", robust_scaler, [COL_AMOUNT]),
        ],
        remainder="passthrough",
    )

    # Fit column transformer with training data

    pd.set_option("future.no_silent_downcasting", True)
    transformer = transformer.fit(pdf_training[predictor_columns])

    # transformed column names
    columns_of_transformed_data = list(
        map(
            lambda name: name.split("__")[1],
            list(transformer.get_feature_names_out(predictor_columns)),
        )
    )

    # data type of transformed columns
    type_mapping = {}
    for col in columns_of_transformed_data:
        if col.split("_")[0] in nominal_predictors:
            type_mapping[col] = "int8"
        elif col in numerical_predictors:
            type_mapping[col] = "float"
        elif col in target_column:
            type_mapping[col] = data.dtypes.to_dict()[col]

    # transform training data
    preprocessed_training_data = transformer.transform(pdf_training[predictor_columns])

    # Convert transformed data to panda DataFrame
    preprocessed_training_data = pd.DataFrame(
        preprocessed_training_data, columns=columns_of_transformed_data
    )

    # Transform test data using the transformer fitted on training data
    pdf_test = data[test_idx][predictor_columns + target_column]
    pdf_test[nominal_predictors] = pdf_test[nominal_predictors].astype("category")

    preprocessed_test_data = transformer.transform(pdf_test[predictor_columns])
    preprocessed_test_data = pd.DataFrame(
        preprocessed_test_data, columns=columns_of_transformed_data
    )

    # Transform validation data using the transformer fitted on training data
    pdf_validation = data[validation_idx][predictor_columns + target_column]
    pdf_validation[nominal_predictors] = pdf_validation[nominal_predictors].astype(
        "category"
    )

    preprocessed_validation_data = transformer.transform(
        pdf_validation[predictor_columns]
    )
    preprocessed_validation_data = pd.DataFrame(
        preprocessed_validation_data, columns=columns_of_transformed_data
    )

    preprocessed_id_data_train = pd.DataFrame(
        id_transformer.transform(data[training_idx][MERCHANT_AND_USER_COLS]),
        columns=columns_of_transformed_id_data,
    )
    preprocessed_training_data = pd.concat(
        [preprocessed_training_data, preprocessed_id_data_train], axis=1
    )

    # ## Write out the data for XGB

    # Copy target column
    preprocessed_training_data[COL_FRAUD] = pdf_training[COL_FRAUD].values
    preprocessed_training_data = preprocessed_training_data.astype(type_mapping)

    assert preprocessed_training_data.columns[-1] == COL_FRAUD
    assert (
        set(preprocessed_training_data.columns)
        - set(
            columns_of_transformed_data + columns_of_transformed_id_data + target_column
        )
        == set()
    )
    assert (
        set(
            columns_of_transformed_data + columns_of_transformed_id_data + target_column
        )
        - set(preprocessed_training_data.columns)
        == set()
    )

    ## Training data
    out_path = os.path.join(tabformer_xgb, "training.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    preprocessed_training_data.to_csv(
        out_path,
        header=True,
        index=False,
        columns=preprocessed_training_data.columns,
    )

    preprocessed_id_data_val = pd.DataFrame(
        id_transformer.transform(data[validation_idx][MERCHANT_AND_USER_COLS]),
        columns=columns_of_transformed_id_data,
    )
    preprocessed_validation_data = pd.concat(
        [preprocessed_validation_data, preprocessed_id_data_val], axis=1
    )

    # Copy target column
    preprocessed_validation_data[COL_FRAUD] = pdf_validation[COL_FRAUD].values
    preprocessed_validation_data = preprocessed_validation_data.astype(type_mapping)

    assert preprocessed_validation_data.columns[-1] == COL_FRAUD
    assert (
        set(preprocessed_validation_data.columns)
        - set(
            columns_of_transformed_data + columns_of_transformed_id_data + target_column
        )
        == set()
    )
    assert (
        set(
            columns_of_transformed_data + columns_of_transformed_id_data + target_column
        )
        - set(preprocessed_validation_data.columns)
        == set()
    )

    ## validation data
    out_path = os.path.join(tabformer_xgb, "validation.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    preprocessed_validation_data.to_csv(
        out_path,
        header=True,
        index=False,
        columns=preprocessed_validation_data.columns,
    )
    # preprocessed_validation_data.to_parquet(out_path, index=False, compression='gzip')

    preprocessed_id_data_test = pd.DataFrame(
        id_transformer.transform(data[test_idx][MERCHANT_AND_USER_COLS]),
        columns=columns_of_transformed_id_data,
    )
    preprocessed_test_data = pd.concat(
        [preprocessed_test_data, preprocessed_id_data_test], axis=1
    )

    # Copy target column
    preprocessed_test_data[COL_FRAUD] = pdf_test[COL_FRAUD].values
    preprocessed_test_data = preprocessed_test_data.astype(type_mapping)

    assert preprocessed_test_data.columns[-1] == COL_FRAUD
    assert (
        set(preprocessed_test_data.columns)
        - set(
            columns_of_transformed_data + columns_of_transformed_id_data + target_column
        )
        == set()
    )
    assert (
        set(
            columns_of_transformed_data + columns_of_transformed_id_data + target_column
        )
        - set(preprocessed_test_data.columns)
        == set()
    )

    ## test data
    out_path = os.path.join(tabformer_xgb, "test.csv")
    preprocessed_test_data.to_csv(
        out_path,
        header=True,
        index=False,
        columns=preprocessed_test_data.columns,
    )

    # Delete dataFrames that are not needed anymore
    del pdf_training
    del pdf_validation
    del pdf_test
    del preprocessed_training_data
    del preprocessed_validation_data
    del preprocessed_test_data

    # ### GNN Data

    # #### Setting Vertex IDs
    # In order to create a graph, the different vertices need to be assigned unique vertex IDs. Additionally, the IDs needs to be consecutive and positive.
    #
    # There are three nodes groups here: Transactions, Users, and Merchants.
    #
    # This IDs are not used in training, just used for graph processing.

    # Use the same training data as used for XGBoost

    data_all = data.copy()
    data = pd.concat([data[training_idx], data[validation_idx]])
    data.reset_index(inplace=True, drop=True)

    # The number of transaction is the same as the size of the list, and hence the index value
    data[COL_TRANSACTION_ID] = data.index

    merchant_name_to_id = dict(
        zip(data[COL_MERCHANT].unique(), np.arange(len(data[COL_MERCHANT].unique())))
    )

    data[COL_MERCHANT_ID] = data[COL_MERCHANT].map(merchant_name_to_id)

    # ##### NOTE: the 'User' and 'Card' columns of the original data were used to crate updated 'Card' column
    # * You can use user or card as nodes

    id_to_consecutive_id = dict(
        zip(data[COL_CARD].unique(), np.arange(len(data[COL_CARD].unique())))
    )

    # Convert Card to consecutive IDs
    data[COL_USER_ID] = data[COL_CARD].map(id_to_consecutive_id)

    NR_USERS = data[COL_USER_ID].max() + 1
    NR_MXS = data[COL_MERCHANT_ID].max() + 1
    NR_TXS = data[COL_TRANSACTION_ID].max() + 1

    # Check the the transaction, merchant and user ids are consecutive
    id_range = data[COL_TRANSACTION_ID].min(), data[COL_TRANSACTION_ID].max()
    print(f"Transaction ID range {id_range}")
    id_range = data[COL_MERCHANT_ID].min(), data[COL_MERCHANT_ID].max()
    print(f"Merchant ID range {id_range}")
    id_range = data[COL_USER_ID].min(), data[COL_USER_ID].max()
    print(f"User ID range {id_range}")

    # #### Create Edge in COO format

    U_2_M = cudf.DataFrame()
    U_2_M[COL_GRAPH_SRC] = data[COL_USER_ID]
    U_2_M[COL_GRAPH_DST] = data[COL_MERCHANT_ID]

    Edge = cudf.concat([U_2_M])

    # Write out Edge data
    out_path = os.path.join(tabformer_gnn, "edges/user_to_merchant.csv")

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    Edge.to_csv(out_path, header=True, index=False)

    # ### Now the feature data
    # Feature data needs to be is sorted in order, where the row index corresponds to the node ID
    #
    # The data is comprised of three sets of features
    # * Transactions
    # * Merchants
    # * Users

    # #### To get feature vectors of Transaction, transform the training data using pre-fitted transformer

    transaction_feature_df = pd.DataFrame(
        transformer.transform(data[predictor_columns]),
        columns=columns_of_transformed_data,
    ).astype(type_mapping)

    transaction_feature_df[COL_FRAUD] = data[COL_FRAUD]

    data_merchant = data[[COL_MERCHANT, COL_MCC, COL_CARD]].drop_duplicates(
        subset=[COL_MERCHANT]
    )
    data_merchant[COL_MERCHANT_ID] = data_merchant[COL_MERCHANT].map(
        merchant_name_to_id
    )
    data_merchant_sorted = data_merchant.sort_values(by=COL_MERCHANT_ID)

    data_user = data[[COL_MERCHANT, COL_MCC, COL_CARD]].drop_duplicates(
        subset=[COL_CARD]
    )
    data_user[COL_USER_ID] = data_user[COL_CARD].map(id_to_consecutive_id)
    data_user_sorted = data_user.sort_values(by=COL_USER_ID)

    user_feature_columns = []
    mx_feature_columns = []
    for c in columns_of_transformed_id_data:
        if c.startswith("Card"):
            user_feature_columns.append(c)
        else:
            mx_feature_columns.append(c)

    preprocessed_merchant_data = pd.DataFrame(
        id_transformer.transform(data_merchant_sorted[MERCHANT_AND_USER_COLS]),
        columns=columns_of_transformed_id_data,
    )[mx_feature_columns]

    preprocessed_user_data = pd.DataFrame(
        id_transformer.transform(data_user_sorted[MERCHANT_AND_USER_COLS]),
        columns=columns_of_transformed_id_data,
    )[user_feature_columns]

    # User features

    out_path = os.path.join(tabformer_gnn, "nodes/user.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    preprocessed_user_data.to_csv(
        out_path, header=True, index=False, columns=user_feature_columns
    )

    # Merchant features

    out_path = os.path.join(tabformer_gnn, "nodes/merchant.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    preprocessed_merchant_data.to_csv(
        out_path, header=True, index=False, columns=mx_feature_columns
    )

    # User to merchant edge labels

    out_path = os.path.join(tabformer_gnn, "edges/user_to_merchant_label.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    transaction_feature_df[[COL_FRAUD]].to_csv(
        out_path, header=True, index=False, columns=[COL_FRAUD]
    )

    # User to merchant edge features

    out_path = os.path.join(tabformer_gnn, "edges/user_to_merchant_attr.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    transaction_feature_df[columns_of_transformed_data].to_csv(
        out_path, header=True, index=False, columns=columns_of_transformed_data
    )

    # # Node feature matrix
    # U = preprocessed_user_data.values
    # M = preprocessed_merchant_data.values
    # T = transaction_feature_df[columns_of_transformed_data].values

    # combined_cols = (
    #     user_feature_columns + mx_feature_columns + columns_of_transformed_data
    # )

    # node_feature_df = pd.DataFrame(block_diag(U, M, T), columns=combined_cols)

    # assert COL_FRAUD not in (
    #     list(preprocessed_user_data.columns)
    #     + list(preprocessed_merchant_data.columns)
    #     + columns_of_transformed_data
    # )

    # # Write out node feature matrix
    # # out_path = os.path.join(tabformer_gnn, "nodes/node.csv")
    # # if not os.path.exists(os.path.dirname(out_path)):
    # #     os.makedirs(os.path.dirname(out_path))
    # # node_feature_df.to_csv(out_path, header=True, index=False, columns=combined_cols)

    # ## Node labels

    # # Initialize with all zeros
    # node_label_df = pd.DataFrame(
    #     np.zeros(len(node_feature_df), dtype=int), columns=[COL_FRAUD]
    # )

    # # Copy the label of transactions to corresponding indices
    # node_label_df.iloc[NR_USERS + NR_MXS : NR_USERS + NR_MXS + NR_TXS, 0] = (
    #     transaction_feature_df[COL_FRAUD].values
    # )

    # # # Write out node labels
    # # out_path = os.path.join(tabformer_gnn, "nodes/node_label.csv")
    # # if not os.path.exists(os.path.dirname(out_path)):
    # #     os.makedirs(os.path.dirname(out_path))
    # # node_label_df.to_csv(out_path, header=True, index=False, columns=[COL_FRAUD])

    # assert data[COL_FRAUD].sum() == node_label_df[COL_FRAUD].sum()

    # # Write NUM_TRANSACTION_NODES in info.json file
    # # with open(
    # #     os.path.join(tabformer_gnn, "nodes/offset_range_of_training_node.json"), "w"
    # # ) as json_file:
    # #     json.dump(
    # #         {"start": int(NR_USERS + NR_MXS), "end": int(NR_USERS + NR_MXS + NR_TXS)},
    # #         json_file,
    # #         indent=4,
    # #     )

    ## Test data

    data = data_all[test_idx].copy()

    data.reset_index(inplace=True, drop=True)

    # The number of transaction is the same as the size of the list, and hence the index value
    data[COL_TRANSACTION_ID] = data.index

    merchant_name_to_id = dict(
        zip(data[COL_MERCHANT].unique(), np.arange(len(data[COL_MERCHANT].unique())))
    )

    data[COL_MERCHANT_ID] = data[COL_MERCHANT].map(merchant_name_to_id)

    # ##### NOTE: the 'User' and 'Card' columns of the original data were used to crate updated 'Card' column
    # * You can use user or card as nodes

    id_to_consecutive_id = dict(
        zip(data[COL_CARD].unique(), np.arange(len(data[COL_CARD].unique())))
    )

    # Convert Card to consecutive IDs
    data[COL_USER_ID] = data[COL_CARD].map(id_to_consecutive_id)

    # Check the the transaction, merchant and user ids are consecutive
    id_range = data[COL_TRANSACTION_ID].min(), data[COL_TRANSACTION_ID].max()
    print(f"Transaction ID range {id_range}")
    id_range = data[COL_MERCHANT_ID].min(), data[COL_MERCHANT_ID].max()
    print(f"Merchant ID range {id_range}")
    id_range = data[COL_USER_ID].min(), data[COL_USER_ID].max()
    print(f"User ID range {id_range}")

    NR_USERS = data[COL_USER_ID].max() + 1
    NR_MXS = data[COL_MERCHANT_ID].max() + 1
    NR_TXS = data[COL_TRANSACTION_ID].max() + 1

    # #### Test Edges in COO format

    # User to Merchant edges
    U_2_M = cudf.DataFrame()
    U_2_M[COL_GRAPH_SRC] = data[COL_USER_ID]
    U_2_M[COL_GRAPH_DST] = data[COL_MERCHANT_ID]

    Edge = cudf.concat([U_2_M])

    # Write out Edge data
    out_path = os.path.join(tabformer_gnn, "test_gnn/edges/user_to_merchant.csv")

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    Edge.to_csv(out_path, header=True, index=False)

    # ### Now the feature data
    # Feature data needs to be is sorted in order, where the row index corresponds to the node ID
    #
    # The data is comprised of three sets of features
    # * Transactions
    # * Merchants
    # * Users

    transaction_feature_df = pd.DataFrame(
        transformer.transform(data[predictor_columns]),
        columns=columns_of_transformed_data,
    ).astype(type_mapping)

    transaction_feature_df[COL_FRAUD] = data[COL_FRAUD]

    data_merchant = data[[COL_MERCHANT, COL_MCC, COL_CARD]].drop_duplicates(
        subset=[COL_MERCHANT]
    )
    data_merchant[COL_MERCHANT_ID] = data_merchant[COL_MERCHANT].map(
        merchant_name_to_id
    )
    data_merchant_sorted = data_merchant.sort_values(by=COL_MERCHANT_ID)

    data_user = data[[COL_MERCHANT, COL_MCC, COL_CARD]].drop_duplicates(
        subset=[COL_CARD]
    )
    data_user[COL_USER_ID] = data_user[COL_CARD].map(id_to_consecutive_id)
    data_user_sorted = data_user.sort_values(by=COL_USER_ID)

    preprocessed_merchant_data = pd.DataFrame(
        id_transformer.transform(data_merchant_sorted[MERCHANT_AND_USER_COLS]),
        columns=columns_of_transformed_id_data,
    )[mx_feature_columns]

    preprocessed_user_data = pd.DataFrame(
        id_transformer.transform(data_user_sorted[MERCHANT_AND_USER_COLS]),
        columns=columns_of_transformed_id_data,
    )[user_feature_columns]

    ## feature matrix

    out_path = os.path.join(tabformer_gnn, "test_gnn/nodes/user.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    preprocessed_user_data.to_csv(
        out_path, header=True, index=False, columns=user_feature_columns
    )

    out_path = os.path.join(tabformer_gnn, "test_gnn/nodes/merchant.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    preprocessed_merchant_data.to_csv(
        out_path, header=True, index=False, columns=mx_feature_columns
    )

    out_path = os.path.join(tabformer_gnn, "test_gnn/edges/user_to_merchant_label.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    transaction_feature_df[[COL_FRAUD]].to_csv(
        out_path, header=True, index=False, columns=[COL_FRAUD]
    )

    out_path = os.path.join(tabformer_gnn, "test_gnn/edges/user_to_merchant_attr.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    transaction_feature_df[columns_of_transformed_data].to_csv(
        out_path, header=True, index=False, columns=columns_of_transformed_data
    )

    # U = preprocessed_user_data.values
    # M = preprocessed_merchant_data.values
    # T = transaction_feature_df[columns_of_transformed_data].values

    # combined_cols = (
    #     user_feature_columns + mx_feature_columns + columns_of_transformed_data
    # )

    # node_feature_df = pd.DataFrame(block_diag(U, M, T), columns=combined_cols)

    # assert COL_FRAUD not in (
    #     list(preprocessed_user_data.columns)
    #     + list(preprocessed_merchant_data.columns)
    #     + columns_of_transformed_data
    # )

    # # Write out node features
    # # out_path = os.path.join(tabformer_gnn, "test_gnn/nodes/node.csv")
    # # if not os.path.exists(os.path.dirname(out_path)):
    # #     os.makedirs(os.path.dirname(out_path))
    # # node_feature_df.to_csv(out_path, header=True, index=False, columns=combined_cols)

    # ## Node labels

    # # Initialize with all zeros
    # node_label_df = pd.DataFrame(
    #     np.zeros(len(node_feature_df), dtype=int), columns=[COL_FRAUD]
    # )

    # # Copy the label of transactions to corresponding indices
    # node_label_df.iloc[NR_USERS + NR_MXS : NR_USERS + NR_MXS + NR_TXS, 0] = (
    #     transaction_feature_df[COL_FRAUD].values
    # )

    # # Write out node labels
    # # out_path = os.path.join(tabformer_gnn, "test_gnn/nodes/node_label.csv")
    # # if not os.path.exists(os.path.dirname(out_path)):
    # #     os.makedirs(os.path.dirname(out_path))
    # # node_label_df.to_csv(out_path, header=True, index=False, columns=[COL_FRAUD])

    # assert set(node_label_df) - set(node_feature_df) == set([COL_FRAUD])
    # assert node_label_df[COL_FRAUD][0 : NR_USERS + NR_MXS].sum() == 0
    # assert (
    #     test_idx.sum() + training_idx.sum() + validation_idx.sum() == data_all.shape[0]
    # )
    # assert COL_FRAUD not in set(node_feature_df.columns)
    # assert COL_FRAUD in set(node_label_df.columns)
    # assert set(node_feature_df.columns) == set(combined_cols)

    return create_feature_mask(
        user_feature_columns + mx_feature_columns + columns_of_transformed_data
    )
