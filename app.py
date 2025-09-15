# app.py

from flask import Flask, render_template, request, jsonify
import osmnx as ox
import networkx as nx
import random
import sqlite3
import time
import joblib
from datetime import datetime
import pandas as pd
import requests # <-- NEW IMPORT for API calls

app = Flask(__name__)

# --- NEW: ADD YOUR API KEY HERE (Phase 5) ---
WEATHER_API_KEY = "237f996616f07b24d9dd1c174b5d6c48" # <-- PASTE YOUR KEY FROM OPENWEATHERMAP
# ----------------------------------------------

# --- Load the pre-trained ML model (Phase 4) ---
try:
    predictor_model = joblib.load('obstacle_predictor.pkl')
    print("Obstacle prediction model loaded successfully.")
except FileNotFoundError:
    print("Warning: obstacle_predictor.pkl not found. Predictive routing will be disabled.")
    predictor_model = None

# --- Database Setup (Phase 3) ---
def init_db():
    conn = sqlite3.connect('database.db')
    print("Opened database successfully")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS obstacles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            reported_at INTEGER NOT NULL,
            description TEXT
        );
    ''')
    print("Table created successfully")
    conn.close()

init_db()

# --- Load map data and add simulated accessibility data ---
place_name = "Navi Mumbai, Maharashtra, India"
print(f"Loading map data for {place_name}...")
G = ox.graph_from_place(place_name, network_type='walk')
print("Map data loaded successfully.")

print("Adding simulated accessibility data to the graph...")
for u, v, key, data in G.edges(keys=True, data=True):
    data['slope'] = abs(random.uniform(0.0, 8.0)) 
    data['surface'] = random.choice(['asphalt', 'concrete', 'gravel'])
print("Accessibility data added.")

def get_recent_obstacles():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    twenty_four_hours_ago = int(time.time()) - 86400
    cursor.execute("SELECT lat, lng FROM obstacles WHERE reported_at > ?", (twenty_four_hours_ago,))
    obstacles = cursor.fetchall()
    conn.close()
    return obstacles

# --- NEW: HELPER FUNCTION TO GET WEATHER (Phase 5) ---
def get_current_weather(api_key, city_name="Navi Mumbai"):
    """Fetches the current primary weather condition."""
    if not api_key or api_key == "237f996616f07b24d9dd1c174b5d6c48":
        print("Weather API key not set. Skipping weather check.")
        return "Unknown"
    
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}"
    
    try:
        response = requests.get(complete_url)
        data = response.json()
        if data.get("cod") != 200:
            print(f"Weather API Error: {data.get('message', 'Unknown Error')}")
            return "Unknown"
        
        # We only care about the main condition (e.g., "Rain", "Clouds", "Clear")
        weather_condition = data["weather"][0]["main"]
        print(f"Current weather in {city_name}: {weather_condition}")
        return weather_condition
    except Exception as e:
        print(f"Could not connect to weather API: {e}")
        return "Unknown"

# --- UPDATED COST FUNCTION WITH WEATHER PENALTY (Phase 5) ---
def calculate_custom_cost(G, u, v, user_profile, obstacles=[], current_weather="Unknown"):
    edge_data = G.get_edge_data(u, v)[0]
    
    node_u_data = G.nodes[u]
    for obs_lat, obs_lng in obstacles:
        dist_to_obstacle = ox.distance.great_circle(node_u_data['y'], node_u_data['x'], obs_lat, obs_lng)
        if dist_to_obstacle < 50:
            return float('inf')
            
    cost = edge_data.get('length', 0)

    if predictor_model:
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        is_weekend = 1 if day_of_week >= 5 else 0
        feature_names = ['hour', 'day_of_week', 'is_rush_hour', 'is_weekend']
        features = pd.DataFrame([[hour, day_of_week, is_rush_hour, is_weekend]], columns=feature_names)
        predicted_prob = predictor_model.predict_proba(features)[0][1]
        cost *= (1 + predicted_prob * 5.0)

    # --- NEW: ADD WEATHER-BASED PENALTY ---
    surface = edge_data.get('surface', 'asphalt')
    if current_weather == "Rain" and surface == "gravel":
        cost *= 3.0 # Add a 3x penalty to gravel paths if it's raining
    # -----------------------------------------

    max_slope = user_profile.get('max_slope', 6)
    slope = edge_data.get('slope', 0)
    if slope > max_slope:
        cost *= 10
    
    disliked_surfaces = user_profile.get('disliked_surfaces', [])
    if surface in disliked_surfaces:
        cost *= 2.0
        
    comfort_pref = user_profile.get('comfort_weight', 0.5) 
    cost += cost * slope * comfort_pref * 0.1

    return cost

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report_obstacle', methods=['POST'])
def report_obstacle():
    # This function remains the same
    # ... (code is unchanged)
    data = request.get_json()
    lat = data['lat']
    lng = data['lng']
    description = data.get('description', 'User-reported obstacle.')
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO obstacles (lat, lng, reported_at, description) VALUES (?, ?, ?, ?)",
        (lat, lng, int(time.time()), description)
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Obstacle reported successfully."})


@app.route('/get_route', methods=['POST'])
def get_route():
    data = request.get_json()
    start_point = data['start']
    end_point = data['end']
    user_profile = data.get('profile', {})
    
    # --- NEW: Get weather ONCE per request ---
    current_weather = get_current_weather(WEATHER_API_KEY)
    # -----------------------------------------
    
    recent_obstacles = get_recent_obstacles()
    start_node = ox.distance.nearest_nodes(G, X=start_point['lng'], Y=start_point['lat'])
    end_node = ox.distance.nearest_nodes(G, X=end_point['lng'], Y=end_point['lat'])

    def heuristic(u, v):
        return ox.distance.great_circle(G.nodes[u]['y'], G.nodes[u]['x'], G.nodes[v]['y'], G.nodes[v]['x'])

    try:
        # --- UPDATED: Pass current_weather to the cost function ---
        route = nx.astar_path(
            G,
            source=start_node,
            target=end_node,
            weight=lambda u, v, d: calculate_custom_cost(G, u, v, user_profile, recent_obstacles, current_weather),
            heuristic=heuristic
        )
        
        route_coords = []
        for node in route:
            point = G.nodes[node]
            route_coords.append([point['y'], point['x']])
            
        return jsonify(route_coords)

    except nx.NetworkXNoPath:
        return jsonify({"error": "No path could be found with the given constraints."}), 400

if __name__ == '__main__':
    app.run(debug=True)