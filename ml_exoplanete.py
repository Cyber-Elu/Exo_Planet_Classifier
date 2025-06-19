import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump

def load_data_from_db():
    conn = sqlite3.connect('exoplanetes.db')
    query = "SELECT * FROM exoplanetes"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

def classify_planets(data):
    for col in ['pl_masse', 'pl_radj', 'pl_dens']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    conditions = [
        (data['pl_masse'] < 10) & (data['pl_radj'] < 2) & (data['pl_dens'] > 5),  # Tellurique
        (data['pl_masse'] >= 10) | (data['pl_radj'] >= 2) | (data['pl_dens'] < 5)  # Gazeuse
    ]
    choices = ['Tellurique', 'Gazeuse']
    data['pl_type'] = np.select(conditions, choices, default='Inconnu')
    return data

def preprocess_data(data):
    features = ['pl_masse', 'pl_radj', 'pl_dens', 'pl_orbper', 'pl_orbsmax', 'pl_eqt']
    target = 'pl_type'
    data = data[features + [target]].dropna()
    X = data[features]
    y = data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calculer et afficher les métriques d'évaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model

def predict_and_save_results(model, data):
    conn = sqlite3.connect('exoplanetes.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS planet_classifications (
        nom TEXT,
        type_predit TEXT,
        masse REAL,
        rayon REAL,
        densite REAL,
        periode_orbite REAL,
        distance_etoile REAL,
        temperature_equilibre REAL
    )
    ''')
    predictions = model.predict(data[['pl_masse', 'pl_radj', 'pl_dens', 'pl_orbper', 'pl_orbsmax', 'pl_eqt']])
    data['predicted_type'] = predictions
    for _, row in data.iterrows():
        cursor.execute('''
        INSERT INTO planet_classifications (nom, type_predit, masse, rayon, densite, periode_orbite, distance_etoile, temperature_equilibre)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['pl_name'],
            row['predicted_type'],
            row['pl_masse'],
            row['pl_radj'],
            row['pl_dens'],
            row['pl_orbper'],
            row['pl_orbsmax'],
            row['pl_eqt']
        ))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    data = load_data_from_db()
    data = classify_planets(data)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    # Sauvegarder le modèle
    dump(model, 'exoplanet_model.joblib')

    # Prédire et sauvegarder les résultats
    predict_and_save_results(model, data)
