import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

def preprocess_and_split_data(df, target_column, test_size=0.2, random_state=42):
    # Preprocesses the DataFrame by performing one-hot encoding, splits the data into training and testing sets,
    # and calculates the average baseline error.

    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df)

    # Separate features and labels
    X = df_encoded.drop([target_column], axis=1)
    y = np.array(df_encoded[target_column])

    # Split the data into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Print the shapes of features and labels for both training and testing sets
    print('Training Features Shape:', train_X.shape)
    print('Training Labels Shape:', train_y.shape)
    print('Testing Features Shape:', test_X.shape)
    print('Testing Labels Shape:', test_y.shape)

    # Calculate baseline error
    mean  = df_encoded['Records'].mean()
    baseline_preds = [mean for i in range(len(test_y))]
    baseline_errors = abs(baseline_preds - test_y)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    data_dmatrix = xgb.DMatrix(data=X, label=y, enable_categorical=True)

    return train_X, test_X, train_y, test_y , data_dmatrix


