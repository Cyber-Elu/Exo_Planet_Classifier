from flask import Flask, render_template
import sqlite3

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('exoplanetes.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    conn = get_db_connection()
    planets = conn.execute('SELECT * FROM planet_classifications').fetchall()
    conn.close()
    return render_template('index.html', planets=planets)

@app.route('/planet/<planet_name>')
def planet(planet_name):
    conn = get_db_connection()
    planet = conn.execute('SELECT * FROM planet_classifications WHERE nom = ?', (planet_name,)).fetchone()
    conn.close()
    if planet is None:
        return "Planète non trouvée", 404
    return render_template('planet.html', planet=planet)

if __name__ == '__main__':
    app.run(debug=True)
