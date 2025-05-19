import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load raw dataset
df = pd.read_csv('PAKDD2010_Modeling_Data.txt', delimiter='\t', header=None, encoding='ISO-8859-1')

# Assign column names
columns = [ 
    'ID_CLIENT', 'CLERK_TYPE', 'PAYMENT_DAY', 'APPLICATION_SUBMISSION_TYPE', 'QUANT_ADDITIONAL_CARDS',
    'POSTAL_ADDRESS_TYPE', 'SEX', 'MARITAL_STATUS', 'QUANT_DEPENDANTS', 'EDUCATION_LEVEL',
    'STATE_OF_BIRTH', 'CITY_OF_BIRTH', 'NACIONALITY', 'RESIDENCIAL_STATE', 'RESIDENCIAL_CITY',
    'RESIDENCIAL_BOROUGH', 'FLAG_RESIDENCIAL_PHONE', 'RESIDENCIAL_PHONE_AREA_CODE', 'RESIDENCE_TYPE',
    'MONTHS_IN_RESIDENCE', 'FLAG_MOBILE_PHONE', 'FLAG_EMAIL', 'PERSONAL_MONTHLY_INCOME',
    'OTHER_INCOMES', 'FLAG_VISA', 'FLAG_MASTERCARD', 'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS',
    'FLAG_OTHER_CARDS', 'QUANT_BANKING_ACCOUNTS', 'QUANT_SPECIAL_BANKING_ACCOUNTS',
    'PERSONAL_ASSETS_VALUE', 'QUANT_CARS', 'COMPANY', 'PROFESSIONAL_STATE', 'PROFESSIONAL_CITY',
    'PROFESSIONAL_BOROUGH', 'FLAG_PROFESSIONAL_PHONE', 'PROFESSIONAL_PHONE_AREA_CODE',
    'MONTHS_IN_THE_JOB', 'PROFESSION_CODE', 'OCCUPATION_TYPE', 'MATE_PROFESSION_CODE',
    'MATE_EDUCATION_LEVEL', 'FLAG_HOME_ADDRESS_DOCUMENT', 'FLAG_RG', 'FLAG_CPF', 'FLAG_INCOME_PROOF',
    'PRODUCT', 'FLAG_ACSP_RECORD', 'AGE', 'RESIDENCIAL_ZIP_3', 'PROFESSIONAL_ZIP_3', 'TARGET_LABEL_BAD'
]
df.columns = columns

# Drop columns with excessive missing values
cols_to_drop = [
    'PROFESSIONAL_CITY', 'PROFESSIONAL_BOROUGH',
    'MATE_PROFESSION_CODE', 'MATE_EDUCATION_LEVEL'
]
df_cleaned = df.drop(columns=cols_to_drop)

# Fill missing values
df_cleaned['RESIDENCE_TYPE'] = df_cleaned['RESIDENCE_TYPE'].fillna(-1)
df_cleaned['MONTHS_IN_RESIDENCE'] = df_cleaned['MONTHS_IN_RESIDENCE'].fillna(0)
df_cleaned['PROFESSION_CODE'] = df_cleaned['PROFESSION_CODE'].fillna(-1)
df_cleaned['OCCUPATION_TYPE'] = df_cleaned['OCCUPATION_TYPE'].fillna(-1)

# Map binary text to numeric
df_cleaned['SEX'] = df_cleaned['SEX'].map({'M': 0, 'F': 1})
df_cleaned['FLAG_MOBILE_PHONE'] = df_cleaned['FLAG_MOBILE_PHONE'].map({'Y': 1, 'N': 0})
df_cleaned['FLAG_EMAIL'] = df_cleaned['FLAG_EMAIL'].astype(float)
df_cleaned['TARGET_LABEL_BAD'] = df_cleaned['TARGET_LABEL_BAD'].astype(int)

# Label encode all other object columns
label_enc = LabelEncoder()
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df_cleaned[col] = label_enc.fit_transform(df_cleaned[col].astype(str))

# Split features and target
X = df_cleaned.drop(columns=['TARGET_LABEL_BAD'])
y = df_cleaned['TARGET_LABEL_BAD']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Combine scaled features and target into a DataFrame
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled['TARGET_LABEL_BAD'] = y.values

# Export to CSV
df_scaled.to_csv('cleaned.csv', index=False)

# Print confirmation
print("Cleaned dataset saved as cleaned.csv")
