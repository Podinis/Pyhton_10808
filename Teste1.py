import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import polars as pl

# 1. Carregamento do dataset
DATA_PATH = r'C:\Users\HP\Desktop\Formação\Eisnt\UFCD 10808 - Limpeza e transformação de dados em Python\Avaliacao\heart_failure_dataset.csv'
df = pd.read_csv(DATA_PATH, sep=',', na_values=["N/D", "NA"])
print(df.head())

# A. Exploração inicial
print(f'Dimensões do dataset: {df.shape}')
print(f'Tipos de dados:\n{df.dtypes}')
print(f'\nValores ausentes por coluna:\n{df.isnull().sum()}')
print(f'\nEstatísticas descritivas:\n{df.describe()}')

# B. Limpeza de outliers
def remove_outliers(df, columns):
    outliers_count_before = {col: ((df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))).sum() +
                                    (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))).sum()) for col in columns}
    
    for col in columns:
        IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
        lower_bound = df[col].quantile(0.25) - 1.5 * IQR
        upper_bound = df[col].quantile(0.75) + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    outliers_count_after = {col: ((df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))).sum() +
                                   (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))).sum()) for col in columns}
    
    return df, outliers_count_before, outliers_count_after

numeric_columns = df.select_dtypes(include=[np.number]).columns
df_cleaned, outliers_before, outliers_after = remove_outliers(df, numeric_columns)

# C. Tratamento de valores em falta
for col in df_cleaned.select_dtypes(include=[object]).columns:
    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

print(f'Valores ausentes após imputação:\n{df_cleaned.isnull().sum()}')

# D. Engenharia de features
df_cleaned['age_group'] = pd.cut(df_cleaned['age'], bins=[0, 40, 60, 80, 100], labels=['0-40', '40-60', '60-80', '80+'])

# E. Codificação de variáveis categóricas
df_one_hot = pd.get_dummies(df_cleaned, drop_first=True)
label_encoder = LabelEncoder()
cat_cols = df_cleaned.select_dtypes(include=[object]).columns
df_label_encoded = df_cleaned.copy()

for col in cat_cols:
    df_label_encoded[col] = label_encoder.fit_transform(df_label_encoded[col])

# F. Escalonamento
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df_cleaned[numeric_columns] = scaler.fit_transform(df_cleaned[numeric_columns])

# G. Validação final
duplicates = df_cleaned.duplicated().sum()
print(f'Número de duplicados: {duplicates}')
print(f'\nValores ausentes finais:\n{df_cleaned.isnull().sum()}')
print(f'\nColunas numéricas escalonadas:\n{df_cleaned[numeric_columns].head()}')
print(f'\nColunas categóricas codificadas:\n{df_cleaned.select_dtypes(include=[object]).head()}')

# H. Exportação
df_cleaned.to_csv('heart_prepared_pandas.csv', index=False)
df_polars = pl.from_pandas(df_cleaned)
df_polars.write_csv('heart_prepared_polars.csv')
