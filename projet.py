# %% [markdown]
# # Projet Machine Learning - Prédiction de Maladies Cardiaques
# Dataset: Heart Disease UCI (https://www.kaggle.com/ronitf/heart-disease-uci)

# %% [code]
# Importation des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Chargement des données
url = 'heart_disease_uci.csv'  # Remplacer par votre lien
df = pd.read_csv(url)

# Nettoyage initial
df.replace('?', np.nan, inplace=True)

# Gestion spécifique de la colonne 'thal'
df['thal'] = df['thal'].fillna(df['thal'].mode()[0])  # Remplacer NaN par le mode
df['thal'] = df['thal'].astype('category')  # Conversion en type catégoriel

# Conversion numérique sécurisée pour 'ca'
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['ca'] = df['ca'].fillna(df['ca'].median())

# %% [code]
# Exploration des données
print("=== Informations du dataset ===")
print(df.info())
print("\n=== Statistiques descriptives ===")
print(df.describe())
print("\n=== Répartition des classes ===")
print(df['num'].value_counts())

plt.figure(figsize=(10,6))
sns.countplot(x='num', data=df)
plt.title('Répartition des classes de maladies cardiaques')
plt.show()

# %% [code]
# Prétraitement des données
# Suppression des colonnes non utiles
df = df.drop(['id', 'dataset'], axis=1)

# Définition des caractéristiques et de la cible
X = df.drop('num', axis=1)
y = df['num'].apply(lambda x: 1 if x > 0 else 0)  # Transformation en problème binaire

# Définition des colonnes
numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Pipeline de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Division des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

# %% [code]
# Construction du modèle
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ))
])

# Hyperparamètres pour l'optimisation
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# %% [code]
# Évaluation du modèle
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("=== Meilleurs hyperparamètres ===")
print(grid_search.best_params_)
print("\n=== Rapport de classification ===")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.show()

# Importance des caractéristiques
encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
feature_names = numeric_features + list(encoder.get_feature_names_out(categorical_features))

importances = best_model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(10,6))
plt.title('Top 10 des caractéristiques importantes')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.show()

# %% [code]
# Exemple de prédiction
sample_data = {
    'age': [50],
    'sex': ['Female'],
    'cp': ['typical angina'],
    'trestbps': [145],
    'chol': [200],
    'fbs': [True],
    'restecg': ['lv hypertrophy'],
    'thalch': [150],
    'exang': [False],
    'oldpeak': [2.3],
    'slope': ['downsloping'],
    'ca': [0],
    'thal': ['fixed defect']
}

sample_df = pd.DataFrame(sample_data)
prediction = best_model.predict(sample_df)
print(f"\nPrédiction pour le patient exemple: {'Malade' if prediction[0] == 1 else 'Sain'}")
 # %%
