import requests
import sqlite3
import pandas as pd
from io import StringIO

# URL de l'archive des exoplanètes
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Requête TAP pour récupérer les données des exoplanètes
query = "SELECT * FROM ps"

def fetch_data_with_tap():
    # Paramètres de la requête
    params = {
        'query': query,
        'format': 'csv'
    }

    # Requête GET au service TAP
    response = requests.get(TAP_URL, params=params)

    if response.status_code == 200:
        # Lecture du CSV dans un DataFrame pandas avec gestion des types de données
        data = pd.read_csv(StringIO(response.text), low_memory=False)
        return data
    else:
        print(f"Erreur lors de la récupération des données: {response.status_code}")
        return None

def create_database(data):
    # Connexion à la base de données SQLite
    conn = sqlite3.connect('exoplanetes.db')
    cursor = conn.cursor()

    # Supprimer la table si elle existe déjà
    cursor.execute('DROP TABLE IF EXISTS exoplanetes')

    # Récupérer les noms de colonnes et créer la table
    columns = ', '.join([f'"{col}" TEXT' for col in data.columns])
    cursor.execute(f'CREATE TABLE exoplanetes ({columns})')

    # Validation des changements et fermeture de la connexion
    conn.commit()
    conn.close()

def insert_data_into_database(data):
    # Connexion à la base de données SQLite
    conn = sqlite3.connect('exoplanetes.db')
    cursor = conn.cursor()

    # Préparation de la requête d'insertion
    placeholders = ', '.join(['?'] * len(data.columns))
    columns = ', '.join([f'"{col}"' for col in data.columns])
    sql = f'INSERT INTO exoplanetes ({columns}) VALUES ({placeholders})'

    # Insertion des données dans la table exoplanetes
    for _, row in data.iterrows():
        try:
            cursor.execute(sql, tuple(row))
        except sqlite3.OperationalError as e:
            print(f"Erreur lors de l'insertion des données: {e}")
            break

    # Validation des changements et fermeture de la connexion
    conn.commit()
    conn.close()

if __name__ == "__main__":
    data = fetch_data_with_tap()
    if data is not None:
        create_database(data)
        insert_data_into_database(data)
