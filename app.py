# app.py

from flask import Flask, render_template, request, jsonify
import osmnx as ox
import networkx as nx
import random
import sqlite3
import time
import joblib
from datetime import datetime # <-- Make sure this is imported
import pandas as pd
import requests 
import json 

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

    # Existing obstacles table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS obstacles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            reported_at INTEGER NOT NULL,
            description TEXT
        );
    ''')

    # New user_profiles table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_name TEXT NOT NULL,
            max_slope REAL DEFAULT 6.0,
            disliked_surfaces TEXT DEFAULT '[]', -- JSON array
            comfort_weight REAL DEFAULT 0.5,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
    ''')

    # New route_history table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS route_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_profile_id INTEGER,
            start_lat REAL NOT NULL,
            start_lng REAL NOT NULL,
            end_lat REAL NOT NULL,
            end_lng REAL NOT NULL,
            route_length REAL,
            route_time REAL,
            weather_condition TEXT,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (user_profile_id) REFERENCES user_profiles (id)
        );
    ''')

    # New ml_training_data table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ml_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            is_rush_hour INTEGER NOT NULL,
            is_weekend INTEGER NOT NULL,
            obstacle_occurred INTEGER NOT NULL, -- 0 or 1
            collected_at INTEGER NOT NULL
        );
    ''')

    print("All tables created successfully")
    conn.close()

init_db()

# --- Load map data and add simulated accessibility data ---
place_name = "Navi Mumbai, Maharashtra, India"
print(f"Loading map data for {place_name}...")
G = ox.graph_from_place(place_name, network_type='walk')
print("Map data loaded successfully.")

print("Adding accessibility data to the graph...")
# For development, use simulated data to avoid API rate limits
# TODO: Replace with real elevation data in production
for u, v, key, data in G.edges(keys=True, data=True):
    data['slope'] = abs(random.uniform(0.0, 8.0))  # Simulated slope
    data['surface'] = random.choice(['asphalt', 'concrete', 'gravel'])  # Simulated surface

    # Handle highway type safely (OSM sometimes has lists)
    highway = data.get('highway', 'unknown')
    if isinstance(highway, list):
        highway = highway[0] if highway else 'unknown'
    data['highway_type'] = highway

    # Add train line detection and penalties
    if 'railway' in data or highway in ['rail', 'light_rail', 'subway']:
        data['is_train_line'] = True
        data['highway_type'] = 'train_line'
    else:
        data['is_train_line'] = False

print("Accessibility data added (using simulated data for development).")

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

# --- NEW: DEM API FUNCTION (Phase 1) ---
def get_elevation(lat, lng):
    """Fetches elevation data for given coordinates using Open-Elevation API."""
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lng}"
        response = requests.get(url, timeout=10)
        data = response.json()

        if 'results' in data and len(data['results']) > 0:
            elevation = data['results'][0]['elevation']
            print(f"Elevation at ({lat}, {lng}): {elevation}m")
            return elevation
        else:
            print(f"Could not get elevation for ({lat}, {lng})")
            return 0.0
    except Exception as e:
        print(f"Error fetching elevation: {e}")
        return 0.0

def calculate_slope(elev1, elev2, distance):
    """Calculate slope percentage between two points."""
    if distance == 0:
        return 0.0
    return abs((elev2 - elev1) / distance) * 100

# --- ADVANCED COST FUNCTION WITH PATH TYPE PENALTIES (Phase 4) ---
def calculate_custom_cost(G, u, v, user_profile, obstacles=[], current_weather="Unknown"):
    edge_data = G.get_edge_data(u, v)[0]

    node_u_data = G.nodes[u]
    for obs_lat, obs_lng in obstacles:
        dist_to_obstacle = ox.distance.great_circle(node_u_data['y'], node_u_data['x'], obs_lat, obs_lng)
        if dist_to_obstacle < 50:
            return float('inf')

    cost = edge_data.get('length', 0)

    # --- ML-based obstacle prediction ---
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

    # --- Path Type Penalties (OSM highway tags) ---
    highway_type = edge_data.get('highway_type', 'unknown')
    path_type_multipliers = {
        'footway': 0.8,      # Prefer pedestrian paths
        'pedestrian': 0.8,   # Prefer pedestrian zones
        'path': 0.9,         # Slightly prefer generic paths
        'residential': 1.2,  # Small penalty for residential streets
        'living_street': 1.1, # Slight penalty for living streets
        'primary': 2.0,      # Heavy penalty for major roads
        'secondary': 1.8,    # High penalty for secondary roads
        'tertiary': 1.5,     # Moderate penalty for tertiary roads
        'motorway': 5.0,     # Extreme penalty for highways
        'trunk': 4.0,        # Very high penalty for trunk roads
        'train_line': float('inf'),  # Completely avoid train lines
        'unknown': 1.0       # No penalty for unknown types
    }
    cost *= path_type_multipliers.get(highway_type, 1.0)

    # --- Enhanced Weather Penalties ---
    surface = edge_data.get('surface', 'asphalt')
    weather_penalties = {
        "Rain": {
            "gravel": 4.0,     # Very slippery
            "concrete": 1.5,   # Somewhat slippery
            "asphalt": 1.2     # Less slippery but still wet
        },
        "Snow": {
            "gravel": 6.0,     # Extremely hazardous
            "concrete": 3.0,   # Very slippery
            "asphalt": 2.0     # Slippery
        },
        "Clear": {
            "gravel": 1.0,     # Normal
            "concrete": 1.0,   # Normal
            "asphalt": 1.0     # Normal
        }
    }

    weather_surface_penalties = weather_penalties.get(current_weather, weather_penalties["Clear"])
    cost *= weather_surface_penalties.get(surface, 1.0)

    # --- Slope penalties ---
    max_slope = user_profile.get('max_slope', 6)
    slope = edge_data.get('slope', 0)
    if slope > max_slope:
        cost *= 10  # Impassable

    # --- Surface preferences ---
    disliked_surfaces = user_profile.get('disliked_surfaces', [])
    if isinstance(disliked_surfaces, str):
        # Handle case where it's stored as JSON string
        try:
            disliked_surfaces = json.loads(disliked_surfaces)
        except:
            disliked_surfaces = []
    # Ensure it's a list
    if not isinstance(disliked_surfaces, list):
        disliked_surfaces = []
    if surface in disliked_surfaces:
        cost *= 2.0

    # --- Comfort weighting ---
    comfort_pref = user_profile.get('comfort_weight', 0.5)
    cost += cost * slope * comfort_pref * 0.1

    return cost

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_profile', methods=['POST'])
def create_profile():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        profile_name = data.get('profile_name')
        max_slope = data.get('max_slope', 6.0)
        disliked_surfaces = data.get('disliked_surfaces', [])
        comfort_weight = data.get('comfort_weight', 0.5)

        if not profile_name:
            return jsonify({"error": "Profile name is required"}), 400

        # Validate inputs
        if not (0 <= max_slope <= 20):
            return jsonify({"error": "Max slope must be between 0 and 20"}), 400
        if not (0 <= comfort_weight <= 1):
            return jsonify({"error": "Comfort weight must be between 0 and 1"}), 400

        current_time = int(time.time())

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_profiles
            (profile_name, max_slope, disliked_surfaces, comfort_weight, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (profile_name, max_slope, json.dumps(disliked_surfaces), comfort_weight, current_time, current_time))

        profile_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Profile created successfully",
            "profile_id": profile_id
        })

    except Exception as e:
        return jsonify({"error": f"Failed to create profile: {str(e)}"}), 500

@app.route('/get_profiles', methods=['GET'])
def get_profiles():
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, profile_name, max_slope, disliked_surfaces, comfort_weight, created_at, updated_at FROM user_profiles ORDER BY updated_at DESC')
        profiles = cursor.fetchall()
        conn.close()

        profile_list = []
        for profile in profiles:
            profile_list.append({
                'id': profile[0],
                'profile_name': profile[1],
                'max_slope': profile[2],
                'disliked_surfaces': json.loads(profile[3]) if profile[3] else [],
                'comfort_weight': profile[4],  # Fixed: was profile[3], should be profile[4]
                'created_at': profile[5],
                'updated_at': profile[6]
            })

        return jsonify({"profiles": profile_list})

    except Exception as e:
        return jsonify({"error": f"Failed to get profiles: {str(e)}"}), 500

@app.route('/update_profile/<int:profile_id>', methods=['PUT'])
def update_profile(profile_id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Get current profile
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user_profiles WHERE id = ?', (profile_id,))
        current_profile = cursor.fetchone()

        if not current_profile:
            conn.close()
            return jsonify({"error": "Profile not found"}), 404

        # Update fields
        updates = {}
        if 'profile_name' in data:
            updates['profile_name'] = data['profile_name']
        if 'max_slope' in data:
            if not (0 <= data['max_slope'] <= 20):
                conn.close()
                return jsonify({"error": "Max slope must be between 0 and 20"}), 400
            updates['max_slope'] = data['max_slope']
        if 'disliked_surfaces' in data:
            updates['disliked_surfaces'] = json.dumps(data['disliked_surfaces'])
        if 'comfort_weight' in data:
            if not (0 <= data['comfort_weight'] <= 1):
                conn.close()
                return jsonify({"error": "Comfort weight must be between 0 and 1"}), 400
            updates['comfort_weight'] = data['comfort_weight']

        if updates:
            set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
            values = list(updates.values()) + [int(time.time()), profile_id]

            cursor.execute(f'UPDATE user_profiles SET {set_clause}, updated_at = ? WHERE id = ?', values)
            conn.commit()

        conn.close()
        return jsonify({"success": True, "message": "Profile updated successfully"})

    except Exception as e:
        return jsonify({"error": f"Failed to update profile: {str(e)}"}), 500

@app.route('/delete_profile/<int:profile_id>', methods=['DELETE'])
def delete_profile(profile_id):
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        # Check if profile exists
        cursor.execute('SELECT id FROM user_profiles WHERE id = ?', (profile_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "Profile not found"}), 404

        # Delete profile
        cursor.execute('DELETE FROM user_profiles WHERE id = ?', (profile_id,))
        conn.commit()
        conn.close()

        return jsonify({"success": True, "message": "Profile deleted successfully"})

    except Exception as e:
        return jsonify({"error": f"Failed to delete profile: {str(e)}"}), 500

@app.route('/report_obstacle', methods=['POST'])
def report_obstacle():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        lat = data.get('lat')
        lng = data.get('lng')
        description = data.get('description', 'User-reported obstacle.')

        if lat is None or lng is None:
            return jsonify({"error": "Latitude and longitude are required"}), 400

        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return jsonify({"error": "Invalid coordinates"}), 400

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO obstacles (lat, lng, reported_at, description) VALUES (?, ?, ?, ?)",
            (lat, lng, int(time.time()), description)
        )
        conn.commit()
        conn.close()

        # Also add to ML training data
        try:
            now = datetime.now()
            hour = now.hour
            day_of_week = now.weekday()
            is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
            is_weekend = 1 if day_of_week >= 5 else 0

            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ml_training_data
                (hour, day_of_week, is_rush_hour, is_weekend, obstacle_occurred, collected_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (hour, day_of_week, is_rush_hour, is_weekend, 1, int(time.time())))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Could not save ML training data: {e}")

        return jsonify({"success": True, "message": "Obstacle reported successfully."})

    except Exception as e:
        return jsonify({"error": f"Failed to report obstacle: {str(e)}"}), 500


def calculate_route_stats(G, route, obstacles, current_weather):
    """Calculate comprehensive statistics for a route."""
    total_length = 0
    total_slope = 0
    max_slope = 0
    surface_counts = {'asphalt': 0, 'concrete': 0, 'gravel': 0}
    highway_counts = {}  # Count different highway types
    obstacle_count = 0
    barrier_count = 0  # Count barriers encountered
    estimated_time = 0  # Estimated walking time in minutes

    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        edge_data = G.get_edge_data(u, v)[0]

        # Length
        length = edge_data.get('length', 0)
        total_length += length

        # Slope
        slope = edge_data.get('slope', 0)
        total_slope += slope * length  # Weighted by length
        max_slope = max(max_slope, slope)

        # Surface
        surface = edge_data.get('surface', 'asphalt')
        surface_counts[surface] = surface_counts.get(surface, 0) + length # Added .get for safety

        # Highway types
        highway_type = edge_data.get('highway_type', 'unknown')
        highway_counts[highway_type] = highway_counts.get(highway_type, 0) + length

        # Estimated walking time (rough calculation: 5 km/h base speed, adjusted for slope and surface)
        base_speed = 5.0  # km/h

        # Slope penalty: reduce speed by 0.5 km/h per 5% slope
        slope_penalty = max(0, slope / 5) * 0.5

        # Surface penalty: different surfaces affect speed
        surface_penalties = {'asphalt': 0, 'concrete': 0.2, 'gravel': 1.0}
        surface_penalty = surface_penalties.get(surface, 0.5)

        # Weather penalty: rain/snow reduces speed
        weather_penalty = 0
        if current_weather in ["Rain", "Snow"]:
            weather_penalty = 1.0 if surface == "gravel" else 0.5

        effective_speed = max(1.0, base_speed - slope_penalty - surface_penalty - weather_penalty)
        time_hours = (length / 1000) / effective_speed  # Convert length to km, divide by speed
        estimated_time += time_hours * 60  # Convert to minutes

        # Obstacles
        node_u_data = G.nodes[u]
        for obs_lat, obs_lng in obstacles:
            dist_to_obstacle = ox.distance.great_circle(node_u_data['y'], node_u_data['x'], obs_lat, obs_lng)
            if dist_to_obstacle < 50:
                obstacle_count += 1
                break

        # Barrier detection (check for barrier tags in OSM data)
        if edge_data.get('barrier') in ['steps', 'kerb', 'bollard', 'gate']:
            barrier_count += 1

    # Calculate averages
    avg_slope = total_slope / total_length if total_length > 0 else 0

    # Weather impact score
    weather_penalty = 0
    if current_weather == "Rain":
        weather_penalty = surface_counts.get('gravel', 0) * 4.0
    elif current_weather == "Snow":
        weather_penalty = total_length * 0.1  # General snow penalty

    # Calculate accessibility score (0-100, higher is better)
    accessibility_score = 100

    # Slope penalties
    if max_slope > 8:
        accessibility_score -= 40
    elif max_slope > 5:
        accessibility_score -= min(30, (max_slope - 5) * 6)
    elif max_slope > 3:
        accessibility_score -= min(15, (max_slope - 3) * 5)

    # Obstacle and barrier penalties
    accessibility_score -= min(25, obstacle_count * 8)
    accessibility_score -= min(25, barrier_count * 10)

    # Surface penalties
    gravel_ratio = surface_counts.get('gravel', 0) / total_length if total_length > 0 else 0
    accessibility_score -= min(20, gravel_ratio * 100)

    # Highway type penalties (prefer pedestrian infrastructure)
    pedestrian_friendly_length = sum(length for ht, length in highway_counts.items()
                                     if ht in ['footway', 'pedestrian', 'path'])
    pedestrian_ratio = pedestrian_friendly_length / total_length if total_length > 0 else 0
    accessibility_score += pedestrian_ratio * 10  # Bonus for pedestrian infrastructure

    return {
        'length': round(total_length, 2),
        'avg_slope': round(avg_slope, 2),
        'max_slope': round(max_slope, 2),
        'surface_breakdown': {k: round(v/total_length*100, 1) if total_length > 0 else 0 for k, v in surface_counts.items()},
        'highway_breakdown': {k: round(v/total_length*100, 1) if total_length > 0 else 0 for k, v in highway_counts.items()},
        'obstacle_count': obstacle_count,
        'barrier_count': barrier_count,
        'weather_penalty': round(weather_penalty, 2),
        'estimated_time': round(estimated_time, 1),
        'accessibility_score': round(max(0, min(100, accessibility_score)), 1)
    }

@app.route('/get_route', methods=['POST'])
def get_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        start_point = data.get('start')
        end_point = data.get('end')
        user_profile = data.get('profile', {})
        route_type = data.get('route_type', 'balanced')  # fastest, comfortable, accessible, balanced
        profile_id = data.get('profile_id')  # Optional: use saved profile

        # If profile_id is provided, load the saved profile
        if profile_id:
            try:
                conn = sqlite3.connect('database.db')
                cursor = conn.cursor()
                cursor.execute('SELECT max_slope, disliked_surfaces, comfort_weight FROM user_profiles WHERE id = ?', (profile_id,))
                profile_data = cursor.fetchone()
                conn.close()

                if profile_data:
                    user_profile = {
                        'max_slope': profile_data[0],
                        'disliked_surfaces': json.loads(profile_data[1]) if profile_data[1] else [],
                        'comfort_weight': profile_data[2]
                    }
                    print(f"Loaded profile {profile_id}: {user_profile}")
            except Exception as e:
                print(f"Could not load profile {profile_id}: {e}")
                # Continue with provided user_profile

        if not start_point or not end_point:
            return jsonify({"error": "Start and end points are required"}), 400

        # --- NEW: Get weather ONCE per request ---
        current_weather = get_current_weather(WEATHER_API_KEY)
        # -----------------------------------------

        recent_obstacles = get_recent_obstacles()

        try:
            start_node = ox.distance.nearest_nodes(G, X=start_point['lng'], Y=start_point['lat'])
            end_node = ox.distance.nearest_nodes(G, X=end_point['lng'], Y=end_point['lat'])

            # Validate that nodes exist and are connected
            if start_node not in G.nodes or end_node not in G.nodes:
                return jsonify({"error": "Start or end point is outside the mapped area"}), 400

        except Exception as e:
            return jsonify({"error": f"Could not find nearest nodes: {str(e)}"}), 400

        def heuristic(u, v):
            try:
                return ox.distance.great_circle(G.nodes[u]['y'], G.nodes[u]['x'], G.nodes[v]['y'], G.nodes[v]['x'])
            except KeyError:
                return 0  # Fallback heuristic

        try:
            # Choose cost function based on route type
            if route_type == 'fastest':
                # Pure distance-based routing with slight preference for better surfaces
                def fastest_cost(u, v, d):
                    edge_data = G.get_edge_data(u, v)[0]
                    cost = edge_data.get('length', 0)

                    # Minor penalty for very bad surfaces to avoid complete inaccessibility
                    surface = edge_data.get('surface', 'asphalt')
                    if surface == 'gravel':
                        cost *= 1.2  # Small penalty

                    # Avoid train lines completely
                    if edge_data.get('is_train_line', False):
                        return float('inf')

                    return cost
                cost_func = fastest_cost
            elif route_type == 'comfortable':
                # Prioritize low slope and good surfaces
                def comfortable_cost(u, v, d):
                    edge_data = G.get_edge_data(u, v)[0]
                    cost = edge_data.get('length', 0)

                    # Check for obstacles first
                    node_u_data = G.nodes[u]
                    for obs_lat, obs_lng in recent_obstacles:
                        dist_to_obstacle = ox.distance.great_circle(node_u_data['y'], node_u_data['x'], obs_lat, obs_lng)
                        if dist_to_obstacle < 50:
                            cost *= 3.0  # Moderate penalty for obstacles

                    # Avoid train lines completely
                    if edge_data.get('is_train_line', False):
                        return float('inf')

                    # Progressive slope penalties
                    slope = edge_data.get('slope', 0)
                    if slope > 8:
                        cost *= 5.0  # Very steep
                    elif slope > 5:
                        cost *= 2.5  # Moderately steep
                    elif slope > 3:
                        cost *= 1.5  # Slightly steep

                    # Surface penalties
                    surface = edge_data.get('surface', 'asphalt')
                    if surface == 'gravel':
                        cost *= 2.5
                    elif surface == 'concrete':
                        cost *= 1.2  # Slight penalty for concrete vs asphalt

                    # Weather impact
                    if current_weather == "Rain" and surface == "gravel":
                        cost *= 4.0

                    return cost
                cost_func = comfortable_cost
            elif route_type == 'accessible':
                # Maximum accessibility - avoid obstacles, slopes, bad surfaces
                def accessible_cost(u, v, d):
                    edge_data = G.get_edge_data(u, v)[0]
                    cost = edge_data.get('length', 0)

                    # Check for obstacles - make them impassable
                    node_u_data = G.nodes[u]
                    for obs_lat, obs_lng in recent_obstacles:
                        dist_to_obstacle = ox.distance.great_circle(node_u_data['y'], node_u_data['x'], obs_lat, obs_lng)
                        if dist_to_obstacle < 50:
                            return float('inf')  # Completely avoid obstacles

                    # Avoid train lines completely
                    if edge_data.get('is_train_line', False):
                        return float('inf')

                    # Very strict slope limits
                    slope = edge_data.get('slope', 0)
                    if slope > 5:  # Maximum 5% slope for accessibility
                        return float('inf')  # Make impassable
                    elif slope > 3:
                        cost *= 3.0  # Heavy penalty
                    elif slope > 2:
                        cost *= 2.0  # Moderate penalty

                    # Strict surface requirements
                    surface = edge_data.get('surface', 'asphalt')
                    if surface == 'gravel':
                        return float('inf')  # Completely avoid gravel
                    elif surface == 'concrete':
                        cost *= 1.5  # Slight penalty for concrete

                    # Weather impact - be extra careful in rain
                    if current_weather == "Rain":
                        if surface == "gravel":
                            return float('inf')  # Never use gravel in rain
                        cost *= 1.5  # General rain penalty

                    return cost
                cost_func = accessible_cost
            else:  # balanced (default) - but since we removed balanced from frontend, this shouldn't be called
                def balanced_cost(u, v, d):
                    return calculate_custom_cost(G, u, v, user_profile, recent_obstacles, current_weather)
                cost_func = balanced_cost

            # Calculate route with improved efficiency for long distances
            try:
                # Use A* with optimized parameters for better performance
                route = nx.astar_path(
                    G,
                    source=start_node,
                    target=end_node,
                    weight=cost_func,
                    heuristic=heuristic
                )
            except nx.NetworkXNoPath:
                # If A* fails, try Dijkstra as fallback for complex paths
                try:
                    route = nx.dijkstra_path(
                        G,
                        source=start_node,
                        target=end_node,
                        weight=cost_func
                    )
                except nx.NetworkXNoPath:
                    raise nx.NetworkXNoPath("No path found with current constraints")

            route_coords = []
            for node in route:
                point = G.nodes[node]
                route_coords.append([point['y'], point['x']])

            # Calculate route statistics
            route_stats = calculate_route_stats(G, route, recent_obstacles, current_weather)

            # Store route in history
            try:
                conn = sqlite3.connect('database.db')
                cursor = conn.cursor()
                
                # --- MODIFIED LINE: Added route_time and its value ---
                cursor.execute('''
                    INSERT INTO route_history
                    (start_lat, start_lng, end_lat, end_lng, route_length, route_time, weather_condition, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (start_point['lat'], start_point['lng'], end_point['lat'], end_point['lng'],
                      route_stats['length'], route_stats['estimated_time'], current_weather, int(time.time())))
                # --- END OF MODIFICATION ---
                      
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Could not save route history: {e}")

            return jsonify({
                'route': route_coords,
                'stats': route_stats,
                'route_type': route_type,
                'weather': current_weather
            })

        except nx.NetworkXNoPath:
            return jsonify({"error": "No path could be found with the given constraints. Try adjusting your preferences or selecting different start/end points."}), 400
        except Exception as e:
            return jsonify({"error": f"Routing calculation failed: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Request processing failed: {str(e)}"}), 500

# --- NEW: DASHBOARD ROUTES ---

@app.route('/dashboard')
def dashboard():
    """Serves the main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/dashboard_stats')
def get_dashboard_stats():
    """Provides aggregate statistics for the dashboard."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()

        # --- MODIFICATION: Updated KPIs to be more insightful ---
        
        # 1. Main KPIs
        cursor.execute("SELECT COUNT(*), COALESCE(AVG(route_length), 0), COALESCE(AVG(route_time), 0) FROM route_history")
        total_routes, avg_distance, avg_time = cursor.fetchone()

        # 2. Routes Today
        today_start = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        cursor.execute("SELECT COUNT(*) FROM route_history WHERE created_at > ?", (today_start,))
        routes_today = cursor.fetchone()[0]

        # 3. Obstacles This Week
        seven_days_ago = int(time.time()) - 7 * 86400
        cursor.execute("SELECT COUNT(*) FROM obstacles WHERE reported_at > ?", (seven_days_ago,))
        obstacles_week = cursor.fetchone()[0]

        # 4. Total Profiles
        cursor.execute("SELECT COUNT(*) FROM user_profiles")
        total_profiles = cursor.fetchone()[0]
        
        # --- END MODIFICATION ---

        # Routes by Day (for chart) - Last 7 days
        cursor.execute("""
            SELECT 
                strftime('%Y-%m-%d', created_at, 'unixepoch') as date, 
                COUNT(*) as count
            FROM route_history
            WHERE created_at > ?
            GROUP BY date
            ORDER BY date ASC
        """, (seven_days_ago,)) # Re-use seven_days_ago
        
        routes_by_day = cursor.fetchall()
        
        # --- REMOVED Weather Query ---

        conn.close()

        # Format chart data
        chart_labels = [row[0] for row in routes_by_day]
        chart_data = [row[1] for row in routes_by_day]
        
        # --- MODIFICATION: Updated KPI dictionary ---
        return jsonify({
            "kpis": {
                "total_routes": total_routes,
                "routes_today": routes_today,
                "avg_distance_km": round(avg_distance / 1000, 2),
                "avg_time_min": round(avg_time, 1),
                "obstacles_week": obstacles_week,
                "total_profiles": total_profiles
            },
            "routes_by_day_chart": {
                "labels": chart_labels,
                "data": chart_data
            }
            # --- REMOVED weather_chart ---
        })

    except Exception as e:
        print(f"Error getting dashboard stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recent_routes')
def get_recent_routes():
    """Provides a list of the 10 most recent routes."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # --- MODIFICATION: Use COALESCE to handle NULL values in old rows ---
        cursor.execute("""
            SELECT start_lat, start_lng, end_lat, end_lng, 
                   COALESCE(route_length, 0) as route_length, 
                   COALESCE(route_time, 0) as route_time, 
                   COALESCE(weather_condition, 'Unknown') as weather_condition, 
                   created_at
            FROM route_history
            ORDER BY created_at DESC
            LIMIT 10
        """)
        # --- END OF MODIFICATION ---
        
        routes = cursor.fetchall()
        conn.close()
        
        recent_routes = []
        for row in routes:
            recent_routes.append({
                "start": f"{round(row[0], 4)}, {round(row[1], 4)}",
                "end": f"{round(row[2], 4)}, {round(row[3], 4)}",
                "length_km": round(row[4] / 1000, 2),
                "time_min": round(row[5], 1),
                "weather": row[6],
                "date": datetime.fromtimestamp(row[7]).strftime('%Y-%m-%d %H:%M')
            })
            
        return jsonify(recent_routes)
        
    except Exception as e:
        print(f"Error getting recent routes: {e}")
        return jsonify({"error": str(e)}), 500

# --- END OF DASHBOARD ROUTES ---

if __name__ == '__main__':
    app.run(debug=True)