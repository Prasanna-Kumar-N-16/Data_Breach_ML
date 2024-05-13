import pandas as pd

def load_data(filepath):
    """
    Load data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    DataFrame or None: DataFrame containing the data if the file is found, otherwise None.
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def preprocess_year(year):
    # Preprocesses the year values.
    return int(year)

def preprocess_source(source):
    # Preprocesses the source values.
    return str(source)

def preprocess_records(records):
    try:
        return int(records)
    except ValueError:
        return None
    
def preprocess_data(df):
    # Drop 'Sources' column
    df.drop(['Sources'], axis=1, inplace=True)
      
    # Rename columns
    df.columns = ['Id', 'Entity', 'Year', 'Records', 'Organization type', 'Method']
    
    # Convert 'Year' to integer
    df['Year'] = df['Year'].astype(str).str[:4].astype(int)
    
    # Create a deep copy for heatmap
    df_heatmap = df.copy(deep=True)
    
    return df, df_heatmap

def text_preprocessing(filepath):
    """
    Preprocesses the text data.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    DataFrame or None: Preprocessed DataFrame if data is loaded successfully, otherwise None.
    """
    df = load_data(filepath)
    if df is None:
        return None,None

    # Filter out unwanted records
    excluded_values = ['2014 and 2015', '2019-2020', '2018-2019']
    df = df[~df['Year'].isin(excluded_values)]
    df = df[~df['Records'].isin(['unknown', 'g20 world leaders', '19 years of data', '63 stores', 'tens of thousands',
                                 'over 5,000,000', 'unknown (client list)', 'millions', '235 gb', '350 clients emails',
                                 'nan', '2.5gb', '250 locations', '500 locations', '10 locations', '93 stores',
                                 'undisclosed', 'source code compromised', '100 terabytes', '54 locations', '200 stores',
                                 '8 locations', '51 locations', 'tbc'])]

    # Convert columns to appropriate data types
    df['Records'] = df['Records'].apply(preprocess_records)
    df['Year'] = df['Year'].apply(preprocess_year)
    df['Sources'] = df['Sources'].apply(preprocess_source)

    # Rename columns
    df.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)

    # Drop rows with NaN values in 'Records' column
    df.dropna(subset=['Records'], inplace=True)

    df.reset_index(drop=True, inplace=True)

    return preprocess_data(df=df)
