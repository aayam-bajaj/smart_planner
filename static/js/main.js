const map = L.map('map').setView([19.0330, 73.0297], 13);
map.whenReady(() => {
    setTimeout(() => {
        map.invalidateSize();
    }, 100);
});

map.getContainer().style.pointerEvents = 'auto';

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

const info = L.control({ position: 'topright' });
info.onAdd = function () {
    this._div = L.DomUtil.create('div', 'info');
    this.update();
    return this._div;
};
info.update = function (text = 'Click to select start and end points.') {
    this._div.innerHTML = `<div style="background:white;padding:10px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.1);">${text}</div>`;
};
info.addTo(map);

let startMarker = null;
let endMarker = null;
let routeLayers = {
    fastest: null,
    comfortable: null,
    accessible: null
};
let routeDataByType = {};
let visibleRoutes = new Set(['fastest', 'comfortable', 'accessible']);

const slopeSlider = document.getElementById('slope');
const comfortSlider = document.getElementById('comfort');
const slopeValue = document.getElementById('slopeValue');
const comfortValue = document.getElementById('comfortValue');

slopeSlider.addEventListener('input', () => slopeValue.textContent = slopeSlider.value);
comfortSlider.addEventListener('input', () => comfortValue.textContent = comfortSlider.value);

document.getElementById('calculate-routes').addEventListener('click', calculateAllRoutes);
document.getElementById('save-profile').addEventListener('click', saveCurrentProfile);

document.getElementById('show-fastest').addEventListener('click', () => toggleRouteVisibility('fastest'));
document.getElementById('show-comfortable').addEventListener('click', () => toggleRouteVisibility('comfortable'));
document.getElementById('show-accessible').addEventListener('click', () => toggleRouteVisibility('accessible'));
document.getElementById('show-all').addEventListener('click', showAllRoutes);

document.getElementById('zoom-all-btn').addEventListener('click', zoomAllRoutes);
document.getElementById('focus-fastest-btn').addEventListener('click', () => focusRoute('fastest'));
document.getElementById('focus-comfortable-btn').addEventListener('click', () => focusRoute('comfortable'));
document.getElementById('focus-accessible-btn').addEventListener('click', () => focusRoute('accessible'));

document.addEventListener('DOMContentLoaded', () => {
    if (typeof loadProfiles === 'function') {
        loadProfiles();
    }
});

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const res = await fetch("/api/me");
        if (!res.ok) return;

        const me = await res.json();

        if (document.getElementById("slope")) {
            document.getElementById("slope").value = me.max_slope;
            document.getElementById("slopeValue").textContent = me.max_slope;
        }

        if (document.getElementById("comfort")) {
            document.getElementById("comfort").value = me.comfort_weight;
            document.getElementById("comfortValue").textContent = me.comfort_weight;
        }

        if (document.getElementById("gravel")) {
            document.getElementById("gravel").checked = me.disliked_surfaces.includes("gravel");
        }

        const userNameEl = document.getElementById("user-name");
        if (userNameEl) userNameEl.textContent = me.name;
    } catch (e) {
        console.log("User profile not loaded");
    }
});

map.on('click', function (e) {
    console.log("Map clicked at:", e.latlng);

    if (!startMarker) {
        startMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
        startMarker.bindPopup('Start').openPopup();
        info.update('Start selected. Now select end point.');
    } else if (!endMarker) {
        endMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
        endMarker.bindPopup('End').openPopup();
        info.update('End selected. Click "Calculate Routes".');
    } else {
        clearSelectionOnly();
        startMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
        startMarker.bindPopup('Start').openPopup();
        info.update('Start reset. Select end point again.');
    }
});

map.on('contextmenu', async function (e) {
    const description = prompt("Describe the obstacle:");
    if (!description) return;

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
            alert('Obstacle reported successfully.');
            L.marker(e.latlng).addTo(map).bindPopup(description);
        } else {
            alert(result.error || 'Failed to report obstacle.');
        }
    } catch (err) {
        console.error(err);
        alert('Failed to report obstacle.');
    }
});

function clearSelectionOnly() {
    if (startMarker) {
        map.removeLayer(startMarker);
        startMarker = null;
    }
    if (endMarker) {
        map.removeLayer(endMarker);
        endMarker = null;
    }
}

function clearAllRoutes() {
    Object.values(routeLayers).forEach(layer => {
        if (layer && map.hasLayer(layer)) {
            map.removeLayer(layer);
        }
    });
    routeLayers = { fastest: null, comfortable: null, accessible: null };
    routeDataByType = {};
    visibleRoutes = new Set(['fastest', 'comfortable', 'accessible']);
    document.getElementById('route-comparison').style.display = 'none';
    document.getElementById('route-stats').innerHTML = '';
    document.getElementById('segment-list').innerHTML = '';
}

async function calculateAllRoutes() {
    if (!startMarker || !endMarker) {
        alert('Please select both start and end points first.');
        return;
    }

    clearAllRoutes();

    const startPoint = startMarker.getLatLng();
    const endPoint = endMarker.getLatLng();

    const dislikedSurfaces = [];
    if (document.getElementById('gravel').checked) dislikedSurfaces.push('gravel');

    const userProfile = {
        max_slope: parseFloat(document.getElementById('slope').value),
        disliked_surfaces: dislikedSurfaces,
        comfort_weight: parseFloat(document.getElementById('comfort').value)
    };

    const routeTypes = ['fastest', 'comfortable', 'accessible'];
    const routeColors = {
        fastest: '#ef4444',
        comfortable: '#22c55e',
        accessible: '#2563eb'
    };

    const routeNames = {
        fastest: 'Fastest',
        comfortable: 'Most Comfortable',
        accessible: 'Most Accessible'
    };

    const routeStats = [];

    for (const routeType of routeTypes) {
        try {
            const response = await fetch('/get_route', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    start: { lat: startPoint.lat, lng: startPoint.lng },
                    end: { lat: endPoint.lat, lng: endPoint.lng },
                    profile: userProfile,
                    route_type: routeType
                })
            });

            const data = await response.json();

            if (!response.ok) {
                console.error(`Error for ${routeType}:`, data.error);
                continue;
            }

            if (!data.route || data.route.length < 2) {
                console.warn(`No visible route for ${routeType}`);
                continue;
            }

            const polyline = L.polyline(data.route, {
                color: routeColors[routeType],
                weight: 5,
                opacity: 0.9
            }).addTo(map);

            polyline.on('click', () => focusRoute(routeType));

            routeLayers[routeType] = polyline;
            routeDataByType[routeType] = data;

            routeStats.push({
                type: routeType,
                name: routeNames[routeType],
                color: routeColors[routeType],
                stats: data.stats,
                segments: data.segments || [],
                bounds: data.bounds
            });
        } catch (error) {
            console.error(`Failed to calculate ${routeType} route:`, error);
        }
    }

    displayRouteComparison(routeStats);
    updateRouteVisibility();

    clearSelectionOnly();
    zoomAllRoutes();
    info.update('Routes calculated. Use the buttons to zoom or focus a route.');
}

function displayRouteComparison(routeStats) {
    const comparisonDiv = document.getElementById('route-comparison');
    const statsDiv = document.getElementById('route-stats');
    const segmentList = document.getElementById('segment-list');

    if (routeStats.length === 0) {
        statsDiv.innerHTML = '<div>No routes could be calculated.</div>';
        segmentList.innerHTML = '';
        comparisonDiv.style.display = 'block';
        return;
    }

    let html = '';
    routeStats.forEach(route => {
        const s = route.stats;
        html += `
            <div class="route-card" style="border-left: 5px solid ${route.color};">
                <h3>${route.name}</h3>
                <div>Distance: ${(s.length / 1000).toFixed(2)} km</div>
                <div>Time: ${s.estimated_time} min</div>
                <div>Average Slope: ${Number(s.avg_slope).toFixed(1)}%</div>
                <div>Max Slope: ${Number(s.max_slope).toFixed(1)}%</div>
                <div>Obstacles: ${s.obstacle_count}</div>
                <div>Accessibility Score: ${s.accessibility_score}/100</div>
                <div style="margin-top:10px;">
                    <button class="route-btn" onclick="focusRoute('${route.type}')">Zoom To Route</button>
                </div>
            </div>
        `;
    });

    statsDiv.innerHTML = html;
    comparisonDiv.style.display = 'block';
    renderSegmentList(routeStats);
}

function renderSegmentList(routeStats) {
    const segmentList = document.getElementById('segment-list');
    let html = '<h3 style="margin-top:14px;">Segment Details</h3>';

    routeStats.forEach(route => {
        html += `<div class="route-card"><h3>${route.name} Segments</h3>`;

        const segments = route.segments || [];
        if (segments.length === 0) {
            html += '<div class="segment-item">No segment data available.</div>';
        } else {
            segments.slice(0, 8).forEach(seg => {
                html += `
                    <div class="segment-item">
                        <b>Segment ${seg.index}</b><br/>
                        Type: ${seg.highway_type}<br/>
                        Surface: ${seg.surface}<br/>
                        Length: ${seg.length.toFixed(1)} m<br/>
                        Slope: ${seg.slope.toFixed(1)}%
                    </div>
                `;
            });

            if (segments.length > 8) {
                html += `<div class="segment-item">+ ${segments.length - 8} more segments</div>`;
            }
        }

        html += '</div>';
    });

    segmentList.innerHTML = html;
}

function zoomAllRoutes() {
    const layers = Object.values(routeLayers).filter(layer => layer !== null);
    if (layers.length === 0) return;

    const group = L.featureGroup(layers);
    map.fitBounds(group.getBounds(), { padding: [40, 40], maxZoom: 18, animate: true });
}

function focusRoute(routeType) {
    const layer = routeLayers[routeType];
    if (!layer) return;

    Object.entries(routeLayers).forEach(([type, line]) => {
        if (!line) return;
        line.setStyle({
            opacity: type === routeType ? 1 : 0.2,
            weight: type === routeType ? 7 : 4
        });
    });

    map.fitBounds(layer.getBounds(), { padding: [40, 40], maxZoom: 19, animate: true });
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
    zoomAllRoutes();
}

function updateRouteVisibility() {
    Object.keys(routeLayers).forEach(routeType => {
        const layer = routeLayers[routeType];
        if (!layer) return;

        if (visibleRoutes.has(routeType)) {
            if (!map.hasLayer(layer)) layer.addTo(map);
        } else {
            if (map.hasLayer(layer)) map.removeLayer(layer);
        }
    });

    document.querySelectorAll('.route-btn').forEach(btn => {
        const routeType = btn.dataset.route;
        if (!routeType || routeType === 'all') return;
        btn.classList.toggle('active', visibleRoutes.has(routeType));
    });
}

async function saveCurrentProfile() {
    const profileName = prompt('Enter a name for this profile:');
    if (!profileName || profileName.trim() === '') return;

    const dislikedSurfaces = [];
    if (document.getElementById('gravel').checked) dislikedSurfaces.push('gravel');

    const profileData = {
        profile_name: profileName.trim(),
        max_slope: parseFloat(document.getElementById('slope').value),
        disliked_surfaces: dislikedSurfaces,
        comfort_weight: parseFloat(document.getElementById('comfort').value)
    };

    try {
        const response = await fetch('/create_profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profileData)
        });

        const result = await response.json();
        if (result.success) {
            alert('Profile saved successfully!');
            loadProfiles();
        } else {
            alert('Error saving profile: ' + result.error);
        }
    } catch (error) {
        console.error('Failed to save profile:', error);
        alert('Failed to save profile.');
    }
}

async function handleRouteCalculation() {
    // optional auto-recalculate hook
}