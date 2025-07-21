import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_input(input_df):
    """
    Preprocess raw user input to match model-ready format.
    """
    # Load encoders and scaler
    scaler = joblib.load("models/standard_scaler.pkl")
    encoder = joblib.load("models/label_encoder_location.pkl")
    onehot_columns = joblib.load("models/onehot_column_list.pkl")
    feature_order = joblib.load("models/feature_columns_order.pkl")
    IQR_bounds = joblib.load("models/iqr_bounds.pkl")

    # Step 1: Feature Engineering for datetime (if applicable)
    if 'Transaction Date' in input_df.columns:
        input_df['Transaction Date'] = pd.to_datetime(input_df['Transaction Date'])
        input_df['Transaction Year'] = input_df['Transaction Date'].dt.year
        input_df['Transaction Month'] = input_df['Transaction Date'].dt.month
        input_df['Transaction Day'] = input_df['Transaction Date'].dt.day
        input_df.drop(columns=['Transaction Date'], inplace=True)

    # Step 2: Encode 'Customer Location'
    # Map known labels only
    known_locations = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    input_df['Customer Location'] = input_df['Customer Location'].map(known_locations)

    # Replace unknowns with a fallback (e.g., mode or -1)
    input_df['Customer Location'].fillna(-1, inplace=True)

    # Step 3: One-hot encoding only for specific columns
    categorical_cols = ['Payment Method', 'Product Category', 'Device Used']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    for col in onehot_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[onehot_columns + ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days',
                                          'Transaction Year', 'Transaction Month', 'Transaction Day','Transaction Hour', 'Customer Location']]
    
    # Step 4: Winsorization (IQR capping)
    if IQR_bounds:
        for col in ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days']:
            lower, upper = IQR_bounds[col]
            input_df[col] = np.where(input_df[col] < lower, lower, input_df[col])
            input_df[col] = np.where(input_df[col] > upper, upper, input_df[col])


    # Step 5: Log transform Transaction Amount
    input_df['Transaction Amount'] = np.log1p(input_df['Transaction Amount'])

    # Step 6: Standardize numeric features
    numeric_cols = ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Step 7: Reorder columns to match training
    input_df = input_df[feature_order]
    
    return input_df

