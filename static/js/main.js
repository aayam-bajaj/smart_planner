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
let routeLines = []; // Array to hold multiple route lines
let currentRoutes = {}; // Store current route data
let visibleRoutes = new Set(['fastest', 'comfortable', 'accessible']); // Track which routes are visible

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

// Add event listener for calculate routes button
document.getElementById('calculate-routes').addEventListener('click', calculateAllRoutes);

// Add event listener for save profile button
document.getElementById('save-profile').addEventListener('click', saveCurrentProfile);

// Add event listeners for route selection buttons
document.getElementById('show-fastest').addEventListener('click', () => toggleRouteVisibility('fastest'));
document.getElementById('show-comfortable').addEventListener('click', () => toggleRouteVisibility('comfortable'));
document.getElementById('show-accessible').addEventListener('click', () => toggleRouteVisibility('accessible'));
document.getElementById('show-all').addEventListener('click', showAllRoutes);

// Load profiles on page load
document.addEventListener('DOMContentLoaded', loadProfiles);
// --- End of New Listeners ---


map.on('click', function(e) {
    if (!startMarker) {
        startMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
        startMarker.on('dragend', handleRouteCalculation);
    } else if (!endMarker) {
        endMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
        endMarker.on('dragend', handleRouteCalculation);
        // Don't auto-calculate, wait for button press
    } else {
        // Clear existing markers and routes
        clearAllRoutes();
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


function clearAllRoutes() {
    // Clear all route lines
    routeLines.forEach(line => {
        if (map.hasLayer(line)) {
            map.removeLayer(line);
        }
    });
    routeLines = [];
    currentRoutes = {};
    visibleRoutes = new Set(['fastest', 'comfortable', 'accessible']);

    // Don't clear markers here - they should persist for recalculation
    // Hide route comparison
    document.getElementById('route-comparison').style.display = 'none';
}

async function calculateAllRoutes() {
    if (!startMarker || !endMarker) {
        alert('Please select both start and end points first.');
        return;
    }

    // Recreate markers if they were cleared
    if (!startMarker || !endMarker) {
        alert('Markers were cleared. Please select start and end points again.');
        return;
    }

    clearAllRoutes();

    // Re-add markers after clearing
    if (startMarker) startMarker.addTo(map);
    if (endMarker) endMarker.addTo(map);

    const startPoint = startMarker.getLatLng();
    const endPoint = endMarker.getLatLng();

    // Get user profile
    const dislikedSurfaces = [];
    if (document.getElementById('gravel').checked) {
        dislikedSurfaces.push('gravel');
    }

    const userProfile = {
        max_slope: parseFloat(document.getElementById('slope').value),
        disliked_surfaces: dislikedSurfaces,
        comfort_weight: parseFloat(document.getElementById('comfort').value)
    };

    const routeTypes = ['fastest', 'comfortable', 'accessible'];
    const routeColors = {
        'fastest': '#ff4444',      // Red
        'comfortable': '#44ff44',  // Green
        'accessible': '#4444ff'    // Blue
    };

    const routeNames = {
        'fastest': '🚀 Fastest',
        'comfortable': '🛋️ Most Comfortable',
        'accessible': '♿ Most Accessible'
    };

    let routeStats = [];

    for (const routeType of routeTypes) {
        try {
            const response = await fetch('/get_route', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    start: { lat: startPoint.lat, lng: startPoint.lng },
                    end: { lat: endPoint.lat, lng: endPoint.lng },
                    profile: userProfile,
                    route_type: routeType
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error(`Error calculating ${routeType} route:`, errorData.error);
                continue;
            }

            const data = await response.json();
            const routeCoords = data.route;
            const stats = data.stats;

            // Create route line
            const routeLine = L.polyline(routeCoords, {
                color: routeColors[routeType],
                weight: 4,
                opacity: 0.8
            }).addTo(map);

            routeLines.push(routeLine);
            currentRoutes[routeType] = { line: routeLine, coords: routeCoords, stats: stats };

            routeStats.push({
                type: routeType,
                name: routeNames[routeType],
                color: routeColors[routeType],
                stats: stats
            });

        } catch (error) {
            console.error(`Failed to fetch ${routeType} route:`, error);
        }
    }

    // Display route comparison
    displayRouteComparison(routeStats);

    // Initially show all routes
    updateRouteVisibility();

    // Fit map to show all routes
    if (routeLines.length > 0) {
        const group = new L.featureGroup(routeLines);
        map.fitBounds(group.getBounds());
    }
}

function displayRouteComparison(routeStats) {
    const comparisonDiv = document.getElementById('route-comparison');
    const statsDiv = document.getElementById('route-stats');

    if (routeStats.length === 0) {
        statsDiv.innerHTML = '<p>No routes could be calculated.</p>';
        comparisonDiv.style.display = 'block';
        return;
    }

    let html = '<table style="width: 100%; border-collapse: collapse; font-size: 12px;">';
    html += '<tr style="background: #f0f0f0;"><th style="padding: 5px; text-align: left;">Route</th><th style="padding: 5px;">Distance</th><th style="padding: 5px;">Time</th><th style="padding: 5px;">Avg Slope</th><th style="padding: 5px;">Max Slope</th><th style="padding: 5px;">Obstacles</th><th style="padding: 5px;">Accessibility</th></tr>';

    routeStats.forEach(route => {
        const stats = route.stats;
        html += `<tr style="border-bottom: 1px solid #ddd;">
            <td style="padding: 5px;"><span style="color: ${route.color}; font-weight: bold;">${route.name}</span></td>
            <td style="padding: 5px;">${(stats.length / 1000).toFixed(2)} km</td>
            <td style="padding: 5px;">${stats.estimated_time} min</td>
            <td style="padding: 5px;">${stats.avg_slope.toFixed(1)}%</td>
            <td style="padding: 5px;">${stats.max_slope.toFixed(1)}%</td>
            <td style="padding: 5px;">${stats.obstacle_count}</td>
            <td style="padding: 5px;">${stats.accessibility_score}/100</td>
        </tr>`;
    });

    html += '</table>';

    // Add surface breakdown
    html += '<h5 style="margin-top: 10px; margin-bottom: 5px;">Surface Composition:</h5>';
    routeStats.forEach(route => {
        const stats = route.stats;
        html += `<div style="margin-bottom: 5px;"><span style="color: ${route.color};">${route.name.split(' ')[1]}:</span> `;
        const surfaces = Object.entries(stats.surface_breakdown).filter(([_, pct]) => pct > 0);
        html += surfaces.map(([surface, pct]) => `${surface}: ${pct}%`).join(', ');
        html += '</div>';
    });
    statsDiv.innerHTML = html;
    comparisonDiv.style.display = 'block';
}

async function saveCurrentProfile() {
    const profileName = prompt('Enter a name for this profile:');
    if (!profileName || profileName.trim() === '') {
        return;
    }

    const dislikedSurfaces = [];
    if (document.getElementById('gravel').checked) {
        dislikedSurfaces.push('gravel');
    }

    const profileData = {
        profile_name: profileName.trim(),
        max_slope: parseFloat(document.getElementById('slope').value),
        disliked_surfaces: dislikedSurfaces,
        comfort_weight: parseFloat(document.getElementById('comfort').value)
    };

    try {
        const response = await fetch('/create_profile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(profileData),
        });

        const result = await response.json();
        if (result.success) {
            alert('Profile saved successfully!');
            loadProfiles(); // Refresh the profile list
        } else {
            alert('Error saving profile: ' + result.error);
        }
    } catch (error) {
        console.error('Failed to save profile:', error);
        alert('Failed to save profile. Please try again.');
    }
}

async function loadProfiles() {
    try {
        const response = await fetch('/get_profiles');
        const data = await response.json();

        const profileListDiv = document.getElementById('profile-list');
        if (data.profiles && data.profiles.length > 0) {
            let html = '';
            data.profiles.forEach(profile => {
                html += `<div style="margin-bottom: 8px; padding: 5px; border: 1px solid #ddd; border-radius: 3px;">
                    <div style="font-weight: bold; font-size: 12px;">${profile.profile_name}</div>
                    <div style="font-size: 11px; color: #666;">
                        Max slope: ${profile.max_slope}%, Comfort: ${profile.comfort_weight}
                    </div>
                    <button onclick="loadProfile(${profile.id})" style="font-size: 11px; padding: 2px 6px; margin-right: 5px;">Load</button>
                    <button onclick="deleteProfile(${profile.id})" style="font-size: 11px; padding: 2px 6px; background: #dc3545; color: white; border: none;">Delete</button>
                </div>`;
            });
            profileListDiv.innerHTML = html;
        } else {
            profileListDiv.innerHTML = '<p style="font-size: 12px; color: #666;">No saved profiles yet.</p>';
        }
    } catch (error) {
        console.error('Failed to load profiles:', error);
        document.getElementById('profile-list').innerHTML = '<p style="font-size: 12px; color: #666;">Error loading profiles.</p>';
    }
}

async function loadProfile(profileId) {
    try {
        const response = await fetch('/get_profiles');
        const data = await response.json();

        const profile = data.profiles.find(p => p.id === profileId);
        if (profile) {
            // Load profile settings into the UI
            document.getElementById('slope').value = profile.max_slope;
            document.getElementById('slopeValue').textContent = profile.max_slope;

            document.getElementById('comfort').value = profile.comfort_weight;
            document.getElementById('comfortValue').textContent = profile.comfort_weight;

            // Update checkboxes
            document.getElementById('gravel').checked = profile.disliked_surfaces.includes('gravel');

            alert(`Profile "${profile.profile_name}" loaded successfully!`);
        }
    } catch (error) {
        console.error('Failed to load profile:', error);
        alert('Failed to load profile. Please try again.');
    }
}

async function deleteProfile(profileId) {
    if (!confirm('Are you sure you want to delete this profile?')) {
        return;
    }

    try {
        const response = await fetch(`/delete_profile/${profileId}`, {
            method: 'DELETE',
        });

        const result = await response.json();
        if (result.success) {
            alert('Profile deleted successfully!');
            loadProfiles(); // Refresh the profile list
        } else {
            alert('Error deleting profile: ' + result.error);
        }
    } catch (error) {
        console.error('Failed to delete profile:', error);
        alert('Failed to delete profile. Please try again.');
    }
}

async function handleRouteCalculation() {
    // This function is now only used for marker dragging
    // The actual route calculation is handled by calculateAllRoutes()
}

function toggleRouteVisibility(routeType) {
    if (visibleRoutes.has(routeType)) {
        visibleRoutes.delete(routeType);
    } else {
        visibleRoutes.add(routeType);
    }
    updateRouteVisibility();
}

function showAllRoutes() {
    visibleRoutes = new Set(['fastest', 'comfortable', 'accessible']);
    updateRouteVisibility();
}

function updateRouteVisibility() {
    Object.keys(currentRoutes).forEach(routeType => {
        const routeData = currentRoutes[routeType];
        if (visibleRoutes.has(routeType)) {
            if (!map.hasLayer(routeData.line)) {
                routeData.line.addTo(map);
            }
        } else {
            if (map.hasLayer(routeData.line)) {
                map.removeLayer(routeData.line);
            }
        }
    });

    // Update button styles to show active state
    document.querySelectorAll('.route-btn').forEach(btn => {
        const routeType = btn.dataset.route;
        if (visibleRoutes.has(routeType)) {
            btn.style.opacity = '1';
            btn.style.fontWeight = 'bold';
        } else {
            btn.style.opacity = '0.5';
            btn.style.fontWeight = 'normal';
        }
    });
}