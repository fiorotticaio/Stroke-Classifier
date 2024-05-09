import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ============== 1. Leitura dos Dados ==============
data = pd.read_csv('healthcare-dataset-stroke-data.csv')


# ============== 2. Limpeza e Preparação dos Dados ==============
data['bmi'] = data['bmi'].replace('N/A', pd.NA)
# Essa linha substitui os valores 'N/A' na coluna bmi por pd.NA, que é um valor especial do pandas que indica dados ausentes

data = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
# get_dummies cria colunas para cada valor único em uma variável categórica. Ou seja, se uma variável categórica 
# tiver mais de duas categorias, cada categoria será representada por uma coluna binária separada. Por exemplo, 
# se a variável work_type tiver as categorias "Private", "Self-employed", "Govt_job" e "children", a codificação 
# one-hot criará quatro novas colunas binárias, uma para cada categoria


# ============== 3. Separação dos Dados ==============
X = data.drop(['id', 'stroke'], axis=1)
# Aqui, estamos criando uma variável X que contém todas as características dos pacientes, exceto o id e a indicação 
# de AVC (stroke). Ou seja, estamos removendo essas duas colunas do conjunto de dados

y = data['stroke']
# Esta linha cria uma variável y que contém apenas a indicação de AVC (stroke). Esta será a variável alvo que 
# queremos prever com base nas características dos pacientes contidas em X


# ============== 4. Construção do Pipeline ==============
numeric_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
# Esta linha define uma lista de nomes das características numéricas do conjunto de dados

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    # Este passo substitui os valores ausentes nas características numéricas pela mediana dos valores presentes naquela característica
    ("normalization", MaxAbsScaler()),
    # Este passo normaliza as características numéricas para que todos os valores estejam entre -1 e 1
    ('scaler', StandardScaler())
    # Este passo padroniza (escala) as características numéricas para que tenham uma média zero e uma variância unitária
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])
# Aqui, estamos criando um ColumnTransformer, que permite aplicar transformações específicas a colunas 
# específicas do conjunto de dados. Estamos configurando este ColumnTransformer para aplicar o numeric_transformer 
# (o pipeline que criamos anteriormente) apenas às características numéricas especificadas em numeric_features. 
# Isso nos permite manter outras características (por exemplo, categóricas) intactas durante o pré-processamento


# ============== 5. Escolha e Treinamento do Modelo ==============
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Aplica todas as transformações definidas no preprocessor às características de entrada
    ('classifier', RandomForestClassifier())])
    # Classificador que pertence à família de algoritmos de florestas aleatórias


# ============== 6. Avaliação do Modelo ==============
scores = cross_val_score(clf, X, y, cv=10, scoring="accuracy")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Calcular outras métricas
y_pred = cross_val_predict(clf, X, y, cv=10)
report = classification_report(y, y_pred)

# Imprimir o relatório de classificação
print("Classification Report:")
print(report)