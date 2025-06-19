import sqlite3

def create_database():
    # connexion à la bdd ou creation si inexistante
    conn = sqlite3.connect('exoplanetes.db')
    cursor = conn.cursor()

    # création des tables pour stocker les données
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS exoplanetes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nom TEXT,
        masse REAL,
        rayon REAL,
        periode_orbite REAL,
        distance_etoile REAL,
        temperature_equilibre REAL,
        date_decouverte TEXT
    )
    ''')


    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
