import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 1. Leitura dos Dados
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# 2. Limpeza e Preparação dos Dados
# Lide com valores ausentes e converta variáveis categóricas em numéricas
data['bmi'] = data['bmi'].replace('N/A', pd.NA)
data = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

# 3. Separação dos Dados
X = data.drop(['id', 'stroke'], axis=1)
y = data['stroke']

# 4. Construção do Pipeline
numeric_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# 5. Escolha e Treinamento do Modelo
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# 6. Avaliação do Modelo
print("X:")
print(X)
print("y:")
print(y)
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
