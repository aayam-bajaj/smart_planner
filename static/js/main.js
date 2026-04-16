
const map = L.map('map').setView([19.0330, 73.0297], 13);
let markerLayer = L.layerGroup().addTo(map);
let routeLayer = L.layerGroup().addTo(map);

let currentStart = null;
let currentEnd = null;
let searchMode = null; // "start" or "end"


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
let activeRouteType = 'fastest';
let latestRouteStats = [];

function formatDistanceMeters(meters) {
    return `${(meters / 1000).toFixed(2)} km`;
}

function formatMinutes(minutes) {
    return `${Math.round(minutes)} min`;
}

function setActiveRouteCard(routeType) {
    document.querySelectorAll('.route-card-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.route === routeType);
    });
}

function setStartPoint(latlng, label = "Start") {
    if (startMarker) {
        markerLayer.removeLayer(startMarker);
    }

    startMarker = L.marker([latlng.lat, latlng.lng], { draggable: true })
        .addTo(markerLayer)
        .bindPopup(label);

    startMarker.on('dragend', () => {
        const pos = startMarker.getLatLng();
        currentStart = { lat: pos.lat, lng: pos.lng };
    });

    currentStart = { lat: latlng.lat, lng: latlng.lng };
    info.update('Start selected. Now select end point.');
}

function setEndPoint(latlng, label = "End") {
    if (endMarker) {
        markerLayer.removeLayer(endMarker);
    }

    endMarker = L.marker([latlng.lat, latlng.lng], { draggable: true })
        .addTo(markerLayer)
        .bindPopup(label);

    endMarker.on('dragend', () => {
        const pos = endMarker.getLatLng();
        currentEnd = { lat: pos.lat, lng: pos.lng };
    });

    currentEnd = { lat: latlng.lat, lng: latlng.lng };
    info.update('End selected. Click "Calculate Routes".');
}

async function searchPlaces(query) {
    if (!query || query.trim().length < 3) return [];

    const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&q=${encodeURIComponent(query)}&limit=5`;

    const res = await fetch(url, {
        headers: {
            'Accept': 'application/json'
        }
    });

    if (!res.ok) return [];
    return await res.json();
}

function renderSuggestions(containerId, results, kind) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!results.length) {
        container.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    container.innerHTML = results.map((place, index) => `
        <div class="suggestion-item" data-kind="${kind}" data-lat="${place.lat}" data-lon="${place.lon}" data-name="${place.display_name}">
            ${place.display_name}
        </div>
    `).join('');

    container.style.display = 'block';

    container.querySelectorAll('.suggestion-item').forEach(item => {
        item.addEventListener('click', () => {
            const latlng = {
                lat: parseFloat(item.dataset.lat),
                lng: parseFloat(item.dataset.lon)
            };
            const name = item.dataset.name;

            if (kind === 'start') {
                setStartPoint(latlng, `Start: ${name}`);
                document.getElementById('start-search').value = name;
                document.getElementById('start-suggestions').style.display = 'none';
            } else {
                setEndPoint(latlng, `End: ${name}`);
                document.getElementById('end-search').value = name;
                document.getElementById('end-suggestions').style.display = 'none';
            }

            map.setView([latlng.lat, latlng.lng], 16);
        });
    });
}

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


//SeachBOX
function setupSearchBox(inputId, buttonId, suggestionsId, kind) {
    const input = document.getElementById(inputId);
    const button = document.getElementById(buttonId);
    const suggestions = document.getElementById(suggestionsId);

    let debounceTimer = null;

    input.addEventListener('input', () => {
        clearTimeout(debounceTimer);

        debounceTimer = setTimeout(async () => {
            const query = input.value.trim();
            if (query.length < 3) {
                suggestions.style.display = 'none';
                suggestions.innerHTML = '';
                return;
            }

            const results = await searchPlaces(query);
            renderSuggestions(suggestionsId, results, kind);
        }, 350);
    });

    button.addEventListener('click', async () => {
        const query = input.value.trim();
        const results = await searchPlaces(query);

        if (!results.length) {
            alert('No location found.');
            return;
        }

        const first = results[0];
        const latlng = {
            lat: parseFloat(first.lat),
            lng: parseFloat(first.lon)
        };

        if (kind === 'start') {
            setStartPoint(latlng, `Start: ${first.display_name}`);
        } else {
            setEndPoint(latlng, `End: ${first.display_name}`);
        }

        map.setView([latlng.lat, latlng.lng], 16);
        suggestions.style.display = 'none';
        suggestions.innerHTML = '';
    });

    input.addEventListener('focus', () => {
        if (suggestions.children.length > 0) {
            suggestions.style.display = 'block';
        }
    });

    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !suggestions.contains(e.target)) {
            suggestions.style.display = 'none';
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    setupSearchBox('start-search', 'start-search-btn', 'start-suggestions', 'start');
    setupSearchBox('end-search', 'end-search-btn', 'end-suggestions', 'end');
});


map.on('click', function (e) {
    console.log("Map clicked at:", e.latlng);

    // if (!startMarker) {
    //     startMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
    //     startMarker.bindPopup('Start').openPopup();
    //     info.update('Start selected. Now select end point.');
    // } else if (!endMarker) {
    //     endMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
    //     endMarker.bindPopup('End').openPopup();
    //     info.update('End selected. Click "Calculate Routes".');
    // } else {
    //     clearSelectionOnly();
    //     startMarker = L.marker(e.latlng, { draggable: true }).addTo(map);
    //     startMarker.bindPopup('Start').openPopup();
    //     info.update('Start reset. Select end point again.');
    // }
    map.on('click', function (e) {
        if (!startMarker) {
            setStartPoint(e.latlng, 'Start');
        } else if (!endMarker) {
            setEndPoint(e.latlng, 'End');
        } else {
            clearSelectionOnly();
            setStartPoint(e.latlng, 'Start');
        }
    });
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

    

    // const startPoint = startMarker.getLatLng();
    // const endPoint = endMarker.getLatLng();

    const startPoint = currentStart || startMarker?.getLatLng();
    const endPoint = currentEnd || endMarker?.getLatLng();

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

            polyline.on('click', () => {
                activeRouteType = routeType;
                setActiveRouteCard(routeType);
                showRouteDetails(routeType);
                focusRoute(routeType);
            });

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

    activeRouteType = routeStats[0]?.type || 'fastest';
    setActiveRouteCard(activeRouteType);
    showRouteDetails(activeRouteType);

    
    zoomAllRoutes();
    info.update('Routes calculated. Use the buttons to zoom or focus a route.');
}

function displayRouteComparison(routeStats) {
    latestRouteStats = routeStats;

    const comparisonDiv = document.getElementById('route-comparison');
    const statsDiv = document.getElementById('route-stats');

    if (routeStats.length === 0) {
        statsDiv.innerHTML = `<div>No routes found</div>`;
        return;
    }

    statsDiv.innerHTML = `
        <h2>Route Comparison</h2>
        <div class="route-grid">
            ${routeStats.map(route => {
        const s = route.stats;
        return `
                    <button class="route-card-btn" data-route="${route.type}">
                        <h3>${route.name}</h3>
                        <div>📏 ${formatDistanceMeters(s.length)}</div>
                        <div>⏱ ${formatMinutes(s.estimated_time)}</div>
                        <div>📈 ${s.avg_slope.toFixed(1)}%</div>
                        <div>⭐ ${s.accessibility_score}/100</div>
                    </button>
                `;
    }).join('')}
        </div>
    `;

    comparisonDiv.style.display = 'block';

    // click handling
    statsDiv.querySelectorAll('.route-card-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const routeType = btn.dataset.route;
            activeRouteType = routeType;

            setActiveRouteCard(routeType);
            showRouteDetails(routeType);
            focusRoute(routeType);
        });
    });

    // default selection
    activeRouteType = routeStats[0].type;
    setActiveRouteCard(activeRouteType);
    showRouteDetails(activeRouteType);
}

function showRouteDetails(routeType) {
    const segmentList = document.getElementById('segment-list');
    const route = latestRouteStats.find(r => r.type === routeType);

    if (!route) {
        segmentList.innerHTML = `<div>No route selected</div>`;
        return;
    }

    const s = route.stats;
    const segments = route.segments || [];

    segmentList.innerHTML = `
        <h2>${route.name}</h2>
        <div>📏 ${formatDistanceMeters(s.length)} | ⏱ ${formatMinutes(s.estimated_time)}</div>

        <div class="details-grid">
            <div>Accessibility: ${s.accessibility_score}</div>
            <div>Avg slope: ${s.avg_slope.toFixed(1)}%</div>
            <div>Max slope: ${s.max_slope.toFixed(1)}%</div>
            <div>Obstacles: ${s.obstacle_count}</div>
        </div>

        <h3>Segments</h3>

        <div class="segment-list-modern">
            ${segments.slice(0, 10).map(seg => `
                    <div class="segment-row">
                        <b>${seg.index}</b> | ${seg.highway_type}
                        <br/>
                        ${seg.length.toFixed(1)}m | ${seg.slope.toFixed(1)}% | ${seg.surface}
                    </div>
                `).join('')
        }
        </div>
    `;
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