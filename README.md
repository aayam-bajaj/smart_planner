# Smart Planner

Accessible route planning with dynamic, multi-objective optimization using Flask, OpenStreetMap, and a modified A* search strategy.

## Overview

Most navigation systems optimize for distance or travel time. In practice, that is often not enough for users who need routes that are safer, smoother, and easier to traverse.

Smart Planner is a web-based accessible navigation system that evaluates road segments using multiple real-world factors such as slope, surface quality, weather conditions, and obstacle risk. The goal is to generate routes that better match user mobility preferences and accessibility needs.

The current implementation is focused on the Navi Mumbai region and combines route computation, user personalization, obstacle reporting, and basic analytics in a single Flask application.

## Problem Statement

Traditional map applications can suggest routes that are technically valid but practically unsuitable for:

- users with mobility limitations
- elderly users
- wheelchair users
- anyone who prefers safer and more comfortable walking routes

Smart Planner addresses that gap by combining geographic data with accessibility-aware routing logic.

## Key Objectives

- Build a personalized navigation system based on user mobility preferences
- Incorporate accessibility factors such as slope, surface type, and obstacle presence
- Use real-time and predictive signals in route planning
- Improve route safety and comfort for vulnerable users
- Provide a foundation for scalable smart-city navigation workflows

## How It Works

### 1. Road Network Modeling

- Walking-network data is sourced from OpenStreetMap through `osmnx`
- The road network is converted into a graph of nodes and edges
- Each edge is enriched with routing attributes used during pathfinding

### 2. Dynamic Cost Function

Each path segment is scored using accessibility-relevant features, including:

- distance
- obstacle risk
- weather conditions
- path characteristics
- slope
- surface type

Conceptually, the cost is modeled as:

```text
C(e) = Length x Risk x Weather x Path x Slope x Surface
```

### 3. User Personalization

Users can configure routing behavior using preferences such as:

- maximum slope tolerance
- comfort weighting
- disliked surface types
- preferred route style

These settings influence how route costs are calculated and which path is ultimately selected.

<img width="1679" height="824" alt="Screenshot 2026-04-01 184634" src="https://github.com/user-attachments/assets/e2447741-a85e-4d8b-a744-ea343426c8b3" />

### 4. Obstacle Risk Prediction

The project includes a logistic regression model that estimates obstacle likelihood using features such as:

- time of day
- day of week
- rush-hour patterns
- weekend behavior

This helps the planner move beyond static routing and incorporate predictive risk into route selection.

### 5. Multi-Objective Pathfinding

Routing is built on a modified A* approach that can evaluate trade-offs across:

- distance
- safety
- comfort
- accessibility

The application supports multiple route strategies, including fastest, comfortable, accessible, and balanced routing.

<img width="1919" height="861" alt="Screenshot 2026-04-01 191638" src="https://github.com/user-attachments/assets/e4308dcf-2137-4162-b328-8857b895d9e9" />
<img width="448" height="856" alt="Screenshot 2026-04-01 184013" src="https://github.com/user-attachments/assets/abddf5f4-d712-4ede-a98b-9314b91b6be7" />

## Features

- Accessibility-focused route planning
- User registration and login
- Personalized route profiles
- Multiple route modes: fastest, comfortable, accessible, and balanced
- Real-time obstacle reporting
- Weather-aware route evaluation
- Map-based interactive interface with Leaflet
- Route history tracking
- Basic dashboard and admin analytics
- ML-backed obstacle risk prediction

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript, Leaflet.js |
| Database | SQLite |
| Map Data | OpenStreetMap, OSMnx |
| Graph Processing | NetworkX |
| Machine Learning | scikit-learn Logistic Regression |
| APIs | Weather API, Elevation API |

## Project Structure

```text
smart_planner/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- database.db
|-- obstacle_predictor.pkl
|-- static/
|   |-- css/
|   `-- js/
|-- templates/
`-- cache/
```

## Results

The project has been tested in the Mumbai and Navi Mumbai region and is designed to produce routes that are more practical than a purely shortest-path approach.

It is intended to reduce exposure to:

- steep slopes
- uncomfortable or unsafe surfaces
- recently reported obstacles
- weather-affected walking conditions

In short, the system prioritizes the most suitable route, not only the shortest one.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Mastercoder0406/smart_planner.git
cd smart_planner
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Application

Before running the app, review configuration values in `app.py`, especially:

- `SECRET_KEY`
- `WEATHER_API_KEY`
- the default map region (`Navi Mumbai, Maharashtra, India`)

### 4. Run the Application

```bash
python app.py
```

### 5. Open in Your Browser

```text
http://localhost:5000
```

## Why This Project Matters

- It moves beyond one-size-fits-all navigation
- It improves route safety and comfort for vulnerable users
- It adapts routing decisions using environmental and contextual data
- It connects digital route planning with real-world accessibility concerns
- It can serve as a foundation for broader smart-city mobility systems

## Future Enhancements

- Computer-vision-based obstacle detection
- Crowd-density and traffic-aware routing
- Voice-assisted navigation for visually impaired users
- Integration with smart-city infrastructure
- Mobile application deployment
- Replacement of simulated accessibility values with richer live data sources

## References

- OpenStreetMap (OSM)
- A* pathfinding algorithm
- Accessibility-aware routing research
- Machine learning methods for obstacle risk prediction
