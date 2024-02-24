import boto3
from src.pylogger import logger
import random
import os
import pandas as pd
from sklearn.model_selection import  train_test_split



def dataset_prep(
                data_dir,
                 file_name):
    # Set up credentials for S3 access
    session = boto3.Session()
    # Create S3 client with the configured session
    s3 = session.client('s3')

    # Define the bucket and object key
    bucket_name = "customer-booking-s3"
    object_key = "customer_booking.csv" # Replace with the actual object key

    # Download the file from S3
    s3.download_file(bucket_name, object_key, data_dir/file_name)

    logger.info(f"Downloading data from {bucket_name} into file {data_dir/file_name}")


def supp(trainset, testset, colm):
    # Compter le nombre d'occurrences de chaque valeur dans la colonne colm
    colm_counts_train = trainset[colm].value_counts()
    colm_counts_test = testset[colm].value_counts()

    # Créer un masque booléen pour sélectionner les valeurs avec plus de 50 occurrences
    mask_train = colm_counts_train[trainset[colm]].values >= 50
    mask_test = colm_counts_test[testset[colm]].values >= 50

    # Supprimer les valeurs
    trainset = trainset[mask_train]
    testset = testset[mask_test]

    # Find similar routes between trainset and testset
    colm_sim = list(set(trainset[colm].unique()) & set(testset[colm].unique()))
    # Filter trainset to contain only lines corresponding to similar colms
    trainset = trainset[trainset[colm].isin(colm_sim)]
    # Filter testset to contain only lines corresponding to similar colms
    testset = testset[testset[colm].isin(colm_sim)]

    logger.info(f"Filter {colm}")
    return trainset, testset

def categorize_flight_time(hour):
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'
    
def encoding(df, fct):

    df['flight_time_of_day'] = df['flight_hour'] // 24  # Extract hour
    df['flight_time_of_day'] = df['flight_time_of_day'].apply(fct)


    # Transform into categorical type
    df['sales_channel'] = df['sales_channel'].astype('category')
    df = pd.get_dummies(df, columns=['sales_channel'], prefix='channel', drop_first=True)

    df['trip_type'] = df['trip_type'].astype('category')
    df = pd.get_dummies(df, columns=['trip_type'], prefix='trip', drop_first=True)

    df['flight_day'] = df['flight_day'].astype('category')
    df = pd.get_dummies(df, columns=['flight_day'], prefix='day', drop_first=True)

    df['route'] = df['route'].astype('category')
    df = pd.get_dummies(df, columns=['route'], prefix='route', drop_first=True)

    df['booking_origin'] = df['booking_origin'].astype('category')
    df = pd.get_dummies(df, columns=['booking_origin'], prefix='booking_origin', drop_first=True)

    df['flight_time_of_day'] = df['flight_time_of_day'].astype('category')
    df = pd.get_dummies(df, columns=['flight_time_of_day'], prefix='flight_time_of_day', drop_first=True)

    logger.info(f"encoding data {df.shape}")
    return df

def encoding_test(df, categorize_flight_time):
    df['flight_time_of_day'] = df['flight_hour'] // 24  # Extract hour
    df['flight_time_of_day'] = df['flight_time_of_day'].apply(categorize_flight_time)
    df['flight_time_of_day'] = df['flight_time_of_day'].astype('category')
    df = pd.get_dummies(df, columns=['flight_time_of_day'], prefix='flight_time_of_day', drop_first=True)
    logger.info(f"encoding api test data {df.shape}")
    return df

def feature_engineering(df):

    df['avg_length_of_stay_per_passenger'] = df['length_of_stay'] / df['num_passengers']
    df['avg_flight_duration_per_passenger'] = df['flight_duration'] / df['num_passengers']

    df['total_purchase_amount'] = df['num_passengers'] * df['purchase_lead']

    # combining num_passengers and length_of_stay to get the total group size.
    df['total_group_size'] = df['num_passengers'] * df['length_of_stay']

    logger.info(f"feature engineering data {df.shape}")
    return df

def imputation(df):
    df = df[(4.67 <= df['flight_duration']) & (df['purchase_lead'] < 200) & (df['length_of_stay'] < 100) & (df['num_passengers'] < 4)]
    df = df[df['flight_duration']<8.83]
    df = df.dropna(axis=0)
    logger.info(f"imputation data {df.shape}")
    return  df
# I am only adding train_test_spliting cz this data is already cleaned up

def save_train_test(train_test_dir_path, train, test):

    train.to_csv(os.path.join(train_test_dir_path, "train.csv"),index = False)
    test.to_csv(os.path.join(train_test_dir_path, "test.csv"),index = False)

    logger.info(f"save trainset and testset into {train_test_dir_path}")
    logger.info(train.shape)
    logger.info(test.shape)


def create_test_dataset(file_path,
                        train_test_dir_path,
                        test_pct,
                        random_state):
    random.seed(random_state)

    data = pd.read_csv(file_path, encoding = "ISO-8859-1")

    # Split the data into training and test sets. (0.75, 0.25) split.
    trainset, testset = train_test_split(data, test_size=test_pct, random_state=0)
    logger.info("Splited data into training and test sets")
    
    trainset, testset = supp(trainset, testset,'route')
    trainset, testset = supp(trainset, testset,'booking_origin')
    trainset = encoding(trainset, categorize_flight_time)
    trainset = feature_engineering(trainset)
    trainset = imputation(trainset)
    testset = encoding(testset, categorize_flight_time)
    testset = feature_engineering(testset)
    testset = imputation(testset)
    save_train_test(train_test_dir_path,train=trainset, test=testset)



def handlapitest(df, dataset_path):
    #dftest
    test_data = pd.read_csv(dataset_path, encoding = "ISO-8859-1")
    test_data = test_data.drop('booking_complete', axis=1)

    # Preprocessing steps
    df_apitest = pd.DataFrame(columns=test_data.columns).astype(test_data.dtypes)
    for col in df.select_dtypes('float64').columns:
        df_apitest[col] = df[col]
    for col in df.select_dtypes('int64').columns:
        df_apitest[col] = df[col]
    # Handle missing columns by adding them and setting them to zero
    missing_columns = set(test_data.select_dtypes('bool').columns) - set(df.select_dtypes('bool').columns)
    missing_columns

    for col in missing_columns:
        df_col = "_".join(col.split('_')[:-1])
        df_value = col.split('_')[-1]

        df_apitest[col] = (df[df_col] == df_value)
    logger.info(f"handl api test data {df_apitest.shape}")
    return df_apitest

    
    
   