# # Credit Card Transaction Data Cleanup and Prep
#
# This notebook shows the steps for cleanup and preparing the credit card transaction data for follow on GNN training with GraphSAGE.
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
import pickle

import cudf
import numpy as np
import pandas as pd
import scipy.stats as ss
from category_encoders import BinaryEncoder
from scipy.stats import pointbiserialr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


# -------
# #### Define some arguments


def proprocess_data(tabformer_base_path):

    # Whether the graph is undirected
    make_undirected = True

    # Whether to spread features across Users and Merchants nodes
    spread_features = False

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

    # --------
    # #### Load and understand the data

    # Read the dataset
    data = cudf.read_csv(tabformer_raw_file_path)

    # optional - take a look at the data
    data.head(5)

    data.columns

    # #### Findings
    # * Ordinal categorical fields - 'Year', 'Month', 'Day'
    # * Nominal categorical fields - 'User', 'Card', 'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?'
    # * Target label - 'Is Fraud?'

    # #### Check if are there Null values in the data

    # Check which fields are missing values
    data.isnull().sum()

    # Check percentage of missing values
    100 * data.isnull().sum() / len(data)

    # #### Findings
    # * For many transactions 'Merchant State' and 'Zip' are missing, but it's good that all of the transactions have 'Merchant City' specified.
    # * Over 98% of the transactions are missing data for 'Errors?' fields.

    # ##### Save a few transactions before any operations on data

    # Write a few raw transactions for model's inference notebook
    out_path = os.path.join(tabformer_xgb, "example_transactions.csv")
    data.tail(10).to_pandas().to_csv(out_path, header=True, index=False)

    # #### Let's rename the column names to single words and use variables for column names to make access easier

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

    UNKNOWN_STRING_MARKER = "XX"
    UNKNOWN_ZIP_CODE = 0

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

    data[COL_AMOUNT].describe()

    # #### Let's look into how the Amount differ between fraud and non-fraud transactions

    # Fraud transactions
    data[COL_AMOUNT][data[COL_FRAUD] == "Yes"].describe()

    # Non-fraud transactions
    data[COL_AMOUNT][data[COL_FRAUD] == "No"].describe()

    # #### Findings
    # * 25th percentile = 9.2
    # * 75th percentile =  65
    # * Median is around 30 and the mean is around 43 whereas the max value is over 1200 and min value is -500
    # * Average amount in Fraud transactions > 2x the average amount in Non-Fraud transactions
    #
    # We need to scale the data, and RobustScaler could be a good choice for it.

    # #### Now the "Fraud" field

    # How many different categories are there in the COL_FRAUD column?
    # The hope is that there are only two categories, 'Yes' and 'No'
    data[COL_FRAUD].unique()

    data[COL_FRAUD].value_counts()

    100 * data[COL_FRAUD].value_counts() / len(data)

    # #### Change the 'Fraud' values to be integer where
    #   * 1 == Fraud
    #   * 0 == Non-fraud

    fraud_to_binary = {"No": 0, "Yes": 1}
    data[COL_FRAUD] = data[COL_FRAUD].map(fraud_to_binary).astype("int8")

    data[COL_FRAUD].value_counts()

    # #### The 'City', 'State', and 'Zip' columns

    # City
    data[COL_CITY].unique()

    # State
    data[COL_STATE].unique()

    # Zip
    data[COL_ZIP].unique()

    # #### The 'Chip' column


    data[COL_CHIP].unique()

    # #### The 'Error' column

    data[COL_ERROR].unique()

    # Remove ',' in error descriptions
    data[COL_ERROR] = data[COL_ERROR].str.replace(",", "")

    # #### Findings
    # We can one hot or binary encode columns with fewer categories and binary/hash encode columns with more than 8 categories

    # #### Time
    # Time is captured as hour:minute.
    #
    # We are converting the time to just be the number of minutes.
    #
    # time = (hour * 60) + minutes

    data[COL_TIME].describe()

    # Split the time column into hours and minutes and then cast to int32
    T = data[COL_TIME].str.split(":", expand=True)
    T[0] = T[0].astype("int32")
    T[1] = T[1].astype("int32")

    # replace the 'Time' column with the new columns
    data[COL_TIME] = (T[0] * 60) + T[1]
    data[COL_TIME] = data[COL_TIME].astype("int32")

    # Delete temporary DataFrame
    del T

    # #### Merchant column

    data[COL_MERCHANT]

    # #### Convert the column to str type

    data[COL_MERCHANT] = data[COL_MERCHANT].astype("str")

    # TOver 100,000 merchants
    data[COL_MERCHANT].unique()

    # #### The Card column
    # * "Card 0" for User 1 is different from "Card 0" for User 2.
    # * Combine User and Card in a way such that (User, Card) combination is unique

    data[COL_CARD].unique()

    max_nr_cards_per_user = len(data[COL_CARD].unique())

    # Combine User and Card to generate unique numbers
    data[COL_CARD] = data[COL_USER] * len(data[COL_CARD].unique()) + data[COL_CARD]
    data[COL_CARD] = data[COL_CARD].astype("int")

    # #### Define function to compute correlation of different categorical fields with target

    # https://en.wikipedia.org/wiki/Cram%C3%A9r's_V

    def cramers_v(x, y):
        confusion_matrix = cudf.crosstab(x, y).to_numpy()
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

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

    # https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient
    # Use Point-biserial correlation coefficient(rpb) to check if the numerical columns are important to predict if a transaction is fraud

    for col in [COL_TIME, COL_AMOUNT]:
        r_pb, p_value = pointbiserialr(
            data[COL_FRAUD].to_pandas(), data[col].to_pandas()
        )
        print("r_pb ({}) = {:3.2f} with p_value {:3.2f}".format(col, r_pb, p_value))

    # ### Findings
    # * Clearly, Time is not an important predictor
    # * Amount has 3% correlation with target

    # #### Based on correlation, select a set of columns (aka fields) to predict whether a transaction is fraud

    # As the cross correlation of Fraud with Day, Month, Year is significantly lower,
    # we can skip them for now and add these features later.

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
    data = cudf.concat([data, fraud_data])

    # Percentage of fraud and non-fraud cases
    100 * data[COL_FRAUD].value_counts() / len(data)

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
        data = cudf.concat([fraud_df, non_fraud_df.sample(nr_non_fraud_samples)])

    training_idx = data[COL_YEAR] < 2018
    validation_idx = data[COL_YEAR] == 2018
    test_idx = data[COL_YEAR] > 2018

    data[COL_FRAUD].value_counts()

    # ### Scale numerical columns and encode categorical columns of training data

    # As some of the encoder we want to use is not available in cuml, we can use pandas for now.
    # Move training data to pandas for preprocessing
    pdf_training = data[training_idx].to_pandas()[predictor_columns + target_column]

    # Use one-hot encoding for columns with <= 8 categories, and binary encoding for columns with more categories
    columns_for_binary_encoding = []
    columns_for_onehot_encoding = []
    for col in nominal_predictors:
        print(col, len(data[col].unique()))
        if len(data[col].unique()) <= 8:
            columns_for_onehot_encoding.append(col)
        else:
            columns_for_binary_encoding.append(col)

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
    onehot_encoder = Pipeline(steps=[("onehot", OneHotEncoder())])
    std_scaler = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("standard", StandardScaler()),
        ],
    )
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
            ("onehot", onehot_encoder, columns_for_onehot_encoding),
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
    # Copy target column
    preprocessed_training_data[COL_FRAUD] = pdf_training[COL_FRAUD].values
    preprocessed_training_data = preprocessed_training_data.astype(type_mapping)

    # Save the transformer

    with open(os.path.join(tabformer_base_path, "preprocessor.pkl"), "wb") as f:
        pickle.dump(transformer, f)

    # #### Save transformed training data for XGBoost training

    with open(os.path.join(tabformer_base_path, "preprocessor.pkl"), "rb") as f:
        loaded_transformer = pickle.load(f)

    # Transform test data using the transformer fitted on training data
    pdf_test = data[test_idx].to_pandas()[predictor_columns + target_column]
    pdf_test[nominal_predictors] = pdf_test[nominal_predictors].astype("category")

    preprocessed_test_data = loaded_transformer.transform(pdf_test[predictor_columns])
    preprocessed_test_data = pd.DataFrame(
        preprocessed_test_data, columns=columns_of_transformed_data
    )

    # Copy target column
    preprocessed_test_data[COL_FRAUD] = pdf_test[COL_FRAUD].values
    preprocessed_test_data = preprocessed_test_data.astype(type_mapping)

    # Transform validation data using the transformer fitted on training data
    pdf_validation = data[validation_idx].to_pandas()[predictor_columns + target_column]
    pdf_validation[nominal_predictors] = pdf_validation[nominal_predictors].astype(
        "category"
    )

    preprocessed_validation_data = loaded_transformer.transform(
        pdf_validation[predictor_columns]
    )
    preprocessed_validation_data = pd.DataFrame(
        preprocessed_validation_data, columns=columns_of_transformed_data
    )

    # Copy target column
    preprocessed_validation_data[COL_FRAUD] = pdf_validation[COL_FRAUD].values
    preprocessed_validation_data = preprocessed_validation_data.astype(type_mapping)

    # ## Write out the data for XGB

    ## Training data
    out_path = os.path.join(tabformer_xgb, "training.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    preprocessed_training_data.to_csv(
        out_path,
        header=True,
        index=False,
        columns=columns_of_transformed_data + target_column,
    )
    # preprocessed_training_data.to_parquet(out_path, index=False, compression='gzip')

    preprocessed_training_data.head(5)

    ## validation data
    out_path = os.path.join(tabformer_xgb, "validation.csv")
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    preprocessed_validation_data.to_csv(
        out_path,
        header=True,
        index=False,
        columns=columns_of_transformed_data + target_column,
    )
    # preprocessed_validation_data.to_parquet(out_path, index=False, compression='gzip')

    ## test data
    out_path = os.path.join(tabformer_xgb, "test.csv")
    preprocessed_test_data.to_csv(
        out_path,
        header=True,
        index=False,
        columns=columns_of_transformed_data + target_column,
    )
    # preprocessed_test_data.to_parquet(out_path, index=False, compression='gzip')

    # Write untransformed test data that has only (renamed) predictor columns and target
    out_path = os.path.join(tabformer_xgb, "untransformed_test.csv")
    pdf_test.to_csv(out_path, header=True, index=False)

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
    data = data[training_idx]

    # a lot of process has occurred, sort the data and reset the index
    data = data.sort_values(by=[COL_YEAR, COL_MONTH, COL_DAY, COL_TIME], ascending=False)
    data.reset_index(inplace=True, drop=True)

    # Each transaction gets a unique ID
    COL_TRANSACTION_ID = "Tx_ID"
    COL_MERCHANT_ID = "Merchant_ID"
    COL_USER_ID = "User_ID"

    # The number of transaction is the same as the size of the list, and hence the index value
    data[COL_TRANSACTION_ID] = data.index

    # Get the max transaction ID to compute first merchant ID
    max_tx_id = data[COL_TRANSACTION_ID].max()

    # Convert Merchant string to consecutive integers
    merchant_name_to_id = dict(
        (v, k) for k, v in data[COL_MERCHANT].unique().to_dict().items()
    )
    data[COL_MERCHANT_ID] = data[COL_MERCHANT].map(merchant_name_to_id) + (
        max_tx_id + 1
    )
    data[COL_MERCHANT_ID].min(), data[COL_MERCHANT].max()

    # Again, get the max merchant ID to compute first user ID
    max_merchant_id = data[COL_MERCHANT_ID].max()

    # ##### NOTE: the 'User' and 'Card' columns of the original data were used to crate updated 'Card' colum
    # * You can use user or card as nodes

    # Convert Card to consecutive IDs
    id_to_consecutive_id = dict(
        (v, k) for k, v in data[COL_CARD].unique().to_dict().items()
    )
    data[COL_USER_ID] = data[COL_CARD].map(id_to_consecutive_id) + max_merchant_id + 1
    data[COL_USER_ID].min(), data[COL_USER_ID].max()

    # id_to_consecutive_id = dict((v, k) for k, v in data[COL_USER].unique().to_dict().items())
    # data[COL_USER_ID] = data[COL_USER].map(id_to_consecutive_id) + max_merchant_id + 1
    # data[COL_USER_ID].min(), data[COL_USER].max()

    # Save the max user ID
    max_user_id = data[COL_USER_ID].max()

    # Check the the transaction, merchant and user ids are consecutive
    id_range = data[COL_TRANSACTION_ID].min(), data[COL_TRANSACTION_ID].max()
    print(f"Transaction ID range {id_range}")
    id_range = data[COL_MERCHANT_ID].min(), data[COL_MERCHANT_ID].max()
    print(f"Merchant ID range {id_range}")
    id_range = data[COL_USER_ID].min(), data[COL_USER_ID].max()
    print(f"User ID range {id_range}")

    # Sanity checks
    assert data[COL_TRANSACTION_ID].max() == data[COL_MERCHANT_ID].min() - 1
    assert data[COL_MERCHANT_ID].max() == data[COL_USER_ID].min() - 1
    assert len(data[COL_USER_ID].unique()) == (
        data[COL_USER_ID].max() - data[COL_USER_ID].min() + 1
    )
    assert len(data[COL_MERCHANT_ID].unique()) == (
        data[COL_MERCHANT_ID].max() - data[COL_MERCHANT_ID].min() + 1
    )
    assert len(data[COL_TRANSACTION_ID].unique()) == (
        data[COL_TRANSACTION_ID].max() - data[COL_TRANSACTION_ID].min() + 1
    )

    # ### Write out the data for GNN

    # #### Create the Graph Edge Data file
    # The file is in COO format

    COL_GRAPH_SRC = "src"
    COL_GRAPH_DST = "dst"
    COL_GRAPH_WEIGHT = "wgt"

    # User to Transactions
    U_2_T = cudf.DataFrame()
    U_2_T[COL_GRAPH_SRC] = data[COL_USER_ID]
    U_2_T[COL_GRAPH_DST] = data[COL_TRANSACTION_ID]
    if make_undirected:
        T_2_U = cudf.DataFrame()
        T_2_U[COL_GRAPH_SRC] = data[COL_TRANSACTION_ID]
        T_2_U[COL_GRAPH_DST] = data[COL_USER_ID]
        U_2_T = cudf.concat([U_2_T, T_2_U])
        del T_2_U

    # Transactions to Merchants
    T_2_M = cudf.DataFrame()
    T_2_M[COL_GRAPH_SRC] = data[COL_MERCHANT_ID]
    T_2_M[COL_GRAPH_DST] = data[COL_TRANSACTION_ID]

    if make_undirected:
        M_2_T = cudf.DataFrame()
        M_2_T[COL_GRAPH_SRC] = data[COL_TRANSACTION_ID]
        M_2_T[COL_GRAPH_DST] = data[COL_MERCHANT_ID]
        T_2_M = cudf.concat([T_2_M, M_2_T])
        del M_2_T

    Edge = cudf.concat([U_2_T, T_2_M])
    Edge[COL_GRAPH_WEIGHT] = 0.0
    len(Edge)

    # now write out the data
    out_path = os.path.join(tabformer_gnn, "edges.csv")

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    Edge.to_csv(out_path, header=False, index=False)

    del Edge
    del U_2_T
    del T_2_M

    # ### Now the feature data
    # Feature data needs to be is sorted in order, where the row index corresponds to the node ID
    #
    # The data is comprised of three sets of features
    # * Transactions
    # * Users
    # * Merchants

    # #### To get feature vectors of Transaction nodes, transform the training data using pre-fitted transformer

    node_feature_df = pd.DataFrame(
        loaded_transformer.transform(data[predictor_columns].to_pandas()),
        columns=columns_of_transformed_data,
    ).astype(type_mapping)

    node_feature_df[COL_FRAUD] = data[COL_FRAUD].to_pandas()

    node_feature_df.head(5)

    # #### For graph nodes associated with merchant and user, add feature vectors of zeros

    # Number of graph nodes for users and merchants
    nr_users_and_merchant_nodes = max_user_id - max_tx_id

    if not spread_features:
        # Create feature vector of all zeros for each user and merchant node
        empty_feature_df = cudf.DataFrame(
            columns=columns_of_transformed_data + target_column,
            dtype="int8",
            index=range(nr_users_and_merchant_nodes),
        )
        empty_feature_df = empty_feature_df.fillna(0)
        empty_feature_df = empty_feature_df.astype(type_mapping)

    if not spread_features:
        # Concatenate transaction features followed by features for merchants and user nodes
        node_feature_df = pd.concat(
            [node_feature_df, empty_feature_df.to_pandas()]
        ).astype(type_mapping)

    # User specific columns
    if spread_features:
        user_specific_columns = [COL_CARD, COL_CHIP]
        user_specific_columns_of_transformed_data = []

        for col in node_feature_df.columns:
            if col.split("_")[0] in user_specific_columns:
                user_specific_columns_of_transformed_data.append(col)

    # Merchant specific columns
    if spread_features:
        merchant_specific_columns = [COL_MERCHANT, COL_CITY, COL_ZIP, COL_MCC]
        merchant_specific_columns_of_transformed_data = []

        for col in node_feature_df.columns:
            if col.split("_")[0] in merchant_specific_columns:
                merchant_specific_columns_of_transformed_data.append(col)

    # Transaction specific columns
    if spread_features:
        transaction_specific_columns = list(
            set(numerical_predictors).union(nominal_predictors)
            - set(user_specific_columns).union(merchant_specific_columns)
        )
        transaction_specific_columns_of_transformed_data = []

        for col in node_feature_df.columns:
            if col.split("_")[0] in transaction_specific_columns:
                transaction_specific_columns_of_transformed_data.append(col)

    # #### Construct feature vector for merchants

    if spread_features:
        # Find indices of unique merchants
        idx_df = cudf.DataFrame()
        idx_df[COL_MERCHANT_ID] = data[COL_MERCHANT_ID]
        idx_df = idx_df.sort_values(by=COL_MERCHANT_ID)
        idx_df = idx_df.drop_duplicates(subset=COL_MERCHANT_ID)
        assert (
            data.iloc[idx_df.index][COL_MERCHANT_ID] == idx_df[COL_MERCHANT_ID]
        ).all()

    if spread_features:
        # Copy merchant specific columns, and set the rest to zero
        merchant_specific_feature_df = node_feature_df.iloc[idx_df.index.to_numpy()]
        merchant_specific_feature_df.loc[
            :,
            transaction_specific_columns_of_transformed_data
            + user_specific_columns_of_transformed_data,
        ] = 0.0

    if spread_features:
        # Find indices of unique users
        idx_df = cudf.DataFrame()
        idx_df[COL_USER_ID] = data[COL_USER_ID]
        idx_df = idx_df.sort_values(by=COL_USER_ID)
        idx_df = idx_df.drop_duplicates(subset=COL_USER_ID)
        assert (data.iloc[idx_df.index][COL_USER_ID] == idx_df[COL_USER_ID]).all()

    if spread_features:
        # Copy user specific columns, and set the rest to zero
        user_specific_feature_df = node_feature_df.iloc[idx_df.index.to_numpy()]
        user_specific_feature_df.loc[
            :,
            transaction_specific_columns_of_transformed_data
            + merchant_specific_columns_of_transformed_data,
        ] = 0.0

    # Concatenate features of node, user and merchant
    if spread_features:

        node_feature_df[merchant_specific_columns_of_transformed_data] = 0.0
        node_feature_df[user_specific_columns_of_transformed_data] = 0.0
        node_feature_df = pd.concat(
            [node_feature_df, merchant_specific_feature_df, user_specific_feature_df]
        ).astype(type_mapping)

        # features to save
        node_feature_df = node_feature_df[
            transaction_specific_columns_of_transformed_data
            + merchant_specific_columns_of_transformed_data
            + user_specific_columns_of_transformed_data
            + [COL_FRAUD]
        ]

    # target labels to save
    label_df = node_feature_df[[COL_FRAUD]]

    # Remove target label from feature vectors
    _ = node_feature_df.drop(columns=[COL_FRAUD], inplace=True)

    # #### Write out node features and target labels

    # Write node target label to csv file
    out_path = os.path.join(tabformer_gnn, "labels.csv")

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    label_df.to_csv(out_path, header=False, index=False)
    # label_df.to_parquet(out_path, index=False, compression='gzip')

    # Write node features to csv file
    out_path = os.path.join(tabformer_gnn, "features.csv")

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    node_feature_df[columns_of_transformed_data].to_csv(
        out_path, header=True, index=False
    )
    # node_feature_df.to_parquet(out_path, index=False, compression='gzip')

    # Delete dataFrames
    del data
    del node_feature_df
    del label_df

    if spread_features:
        del merchant_specific_feature_df
        del user_specific_feature_df
    else:
        del empty_feature_df

    # #### Number of transaction nodes in training data

    # Number of transaction nodes, needed for GNN training
    nr_transaction_nodes = max_tx_id + 1
    nr_transaction_nodes

    # #### Maximum number of cards per user

    # Max number of cards per user, needed for inference
    max_nr_cards_per_user

    # #### Save variable for training and inference

    variables_to_save = {
        k: v
        for k, v in globals().items()
        if isinstance(v, (str, int)) and k.startswith("COL_")
    }

    variables_to_save["NUM_TRANSACTION_NODES"] = int(nr_transaction_nodes)
    variables_to_save["MAX_NR_CARDS_PER_USER"] = int(max_nr_cards_per_user)

    # Save the dictionary to a JSON file

    with open(os.path.join(tabformer_base_path, "variables.json"), "w") as json_file:
        json.dump(variables_to_save, json_file, indent=4)

    with open(os.path.join(tabformer_gnn, "info.json"), "w") as json_file:
        json.dump(
            {"NUM_TRANSACTION_NODES": int(nr_transaction_nodes)}, json_file, indent=4
        )
