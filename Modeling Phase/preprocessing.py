from imports import *

def basic_eda(df):
    print("----------TOP 5 RECORDS--------")
    print(df.head(5))
    print("----------INFO-----------------")
    print(df.info())
    print("----------Describe-------------")
    print(df.describe())
    print("----------Columns--------------")
    print(df.columns)
    print("----------Number of Unique Elements-----------")
    print(df.nunique())
    print("----------Data Types-----------")
    print(df.dtypes)
    print("-------Missing Values----------")
    print(df.isnull().sum())
    print("-------NULL values-------------")
    print(df.isna().sum())
    print("-----Shape Of Data-------------")
    print(df.shape)

def preprocess_cols(df):
    df['name'] = df['name'].str.split(' ').str[0]
    df['mileage'] = df['mileage'].str.split(' ', expand=True)[0]
    df['engine'] = df['engine'].str.split(' ', expand=True)[0]
    df['max_power'] = df['max_power'].str.split(' ', expand=True)[0]
    df = df.replace(r'', np.nan)
    df['engine'] = df['engine'].astype('float')
    df['mileage'] = df['mileage'].astype('float')
    df['max_power'] = df['max_power'].astype('float')
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def adjust_typo_torque(x):
    if type(x) != str:
        return x
    if 'at' in x:
        x = x.replace('at', '@')
        return x
    elif '(kgm@ rpm)' in x:
        x = x.replace('(kgm@ rpm)', '')
        return x
    else:
        return x

def adjust_units_torque(x):
    if type(x) != str:
        return x
    numeric_const_pattern = r"""
    [-+]? # optional sign
    (?:
         (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
         |
         (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
         )
     (?: [Ee] [+-]? \d+ ) ?
     """

    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    x = rx.findall(x)
    x = float(x[0])
    if x <= 50:
        x = 9.8*x
    return x

def adjust_rpm_torque(x):
    if type(x) != str:
        return x
    if ',' in x:
        x = x.replace(',', '') 
    numeric_const_pattern = r"""
    [-+]? # optional sign
    (?:
         (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
         |
         (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
         )
     (?: [Ee] [+-]? \d+ ) ?
     """
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    x = rx.findall(x)
    x = float(x[0])
    return x

def get_cols_with_missing_values(DataFrame):
    missing_na_columns=(DataFrame.isnull().sum())
    return missing_na_columns[missing_na_columns > 0]

def fill_nan(df):
    for col in df.columns:
        if df[col].dtype != 'object':
            if df[col].nunique() < 100:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
    return df

def get_dummy(df, cols):
    df_dummies = pd.get_dummies(data = df[cols], drop_first=True) 
    df.drop(columns=cols, inplace=True)
    df = pd.concat([df, df_dummies], axis=1)
    return df