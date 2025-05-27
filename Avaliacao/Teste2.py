import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

# 1. Carregamento do dataset
DATA_PATH = r'C:\Users\HP\Desktop\Formação\Eisnt\UFCD 10808 - Limpeza e transformação de dados em Python\Avaliacao\heart_failure_dataset.csv'
df = pd.read_csv(DATA_PATH, sep=',', na_values=["N/D", "NA"])

# 2. Definindo a classe para remover outliers
class OutlierRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = X.select_dtypes(include=[np.number]).columns
        return self

    def transform(self, X):
        X_cleaned = X.copy()
        for col in self.columns_:
            IQR = X_cleaned[col].quantile(0.75) - X_cleaned[col].quantile(0.25)
            lower_bound = X_cleaned[col].quantile(0.25) - 1.5 * IQR
            upper_bound = X_cleaned[col].quantile(0.75) + 1.5 * IQR
            X_cleaned = X_cleaned[(X_cleaned[col] >= lower_bound) & (X_cleaned[col] <= upper_bound)]
        return X_cleaned

# 3. Definindo as colunas
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=[object]).columns.tolist()

# 4. Criando a pipeline
numeric_transformer = Pipeline(steps=[
    ('outlier_remover', OutlierRemover()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Criando o modelo final
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# 6. Separando as variáveis independentes e dependentes
X = df.drop('heartdisease', axis=1)
y = df['heartdisease']

# 7. Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Ajustando a pipeline
model.fit(X_train, y_train)

# 9. Fazendo previsões e avaliando o modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 10. Exportando os dados preparados
df_cleaned = model.named_steps['preprocessor'].transform(X)
df_cleaned = pd.DataFrame(df_cleaned)
df_cleaned.to_csv('heart_prepared_pandas.csv', index=False)

# Exportar para Polars
import polars as pl
df_polars = pl.from_pandas(df_cleaned)
df_polars.write_csv('heart_prepared_polars.csv')
