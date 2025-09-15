// static/js/main.js (Final Version)

// Initialize the map and set its view to Navi Mumbai
const map = L.map('map').setView([19.0330, 73.0297], 13);

// Add a tile layer to the map
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Add a small info box
const info = L.control();
info.onAdd = function (map) {
    this._div = L.DomUtil.create('div', 'info');
    this.update();
    return this._div;
};
info.update = function (props) {
    this._div.innerHTML = '<h4>Smart Route Planner</h4>' + 'Click to select start and end points.';
};
info.addTo(map);

let startMarker = null;
let endMarker = null;
let routeLine = null;

// --- NEW: Event listeners for the interactive controls ---
const slopeSlider = document.getElementById('slope');
const comfortSlider = document.getElementById('comfort');
const slopeValue = document.getElementById('slopeValue');
const comfortValue = document.getElementById('comfortValue');

slopeSlider.addEventListener('input', () => {
    slopeValue.textContent = slopeSlider.value;
});
comfortSlider.addEventListener('input', () => {
    comfortValue.textContent = comfortSlider.value;
});
// --- End of New Listeners ---


map.on('click', function(e) {
    if (!startMarker) {
        startMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
        startMarker.on('dragend', handleRouteCalculation);
    } else if (!endMarker) {
        endMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
        endMarker.on('dragend', handleRouteCalculation);
        handleRouteCalculation();
    } else {
        map.removeLayer(startMarker);
        map.removeLayer(endMarker);
        if (routeLine) {
            map.removeLayer(routeLine);
        }
        startMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
        endMarker = null;
        startMarker.on('dragend', handleRouteCalculation);
    }
});

map.on('contextmenu', async function(e) {
    const description = prompt("Describe the obstacle (e.g., 'Construction blocking sidewalk'):");
    if (description) {
        try {
            const response = await fetch('/report_obstacle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    lat: e.latlng.lat,
                    lng: e.latlng.lng,
                    description: description
                })
            });
            const result = await response.json();
            if (result.success) {
                alert('Obstacle reported! It will be considered in routing for the next 24 hours.');
                L.marker(e.latlng).addTo(map).bindPopup(description).openPopup();
            }
        } catch (error) {
            console.error('Failed to report obstacle:', error);
        }
    }
});


async function handleRouteCalculation() {
    if (!startMarker || !endMarker) return;

    const startPoint = startMarker.getLatLng();
    const endPoint = endMarker.getLatLng();

    if (routeLine) {
        map.removeLayer(routeLine);
        routeLine = null;
    }

    // --- UPDATED: Dynamically create the user profile from the HTML controls ---
    const dislikedSurfaces = [];
    if (document.getElementById('gravel').checked) {
        dislikedSurfaces.push('gravel');
    }

    const userProfile = {
        max_slope: parseFloat(document.getElementById('slope').value),
        disliked_surfaces: dislikedSurfaces,
        comfort_weight: parseFloat(document.getElementById('comfort').value)
    };
    // --- End of Update ---

    try {
        const response = await fetch('/get_route', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                start: { lat: startPoint.lat, lng: startPoint.lng },
                end: { lat: endPoint.lat, lng: endPoint.lng },
                profile: userProfile // Send the dynamic profile
            }),
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            alert('Error: ' + (errorData.error || 'Could not calculate route.'));
            return;
        }

        const routeCoords = await response.json();
        
        routeLine = L.polyline(routeCoords, { color: 'blue' }).addTo(map);
        map.fitBounds(routeLine.getBounds());

    } catch (error) {
        console.error('Failed to fetch route:', error);
        alert('An error occurred while planning the route.');
    }
}