import folium

# Initialize the map, centered on Central Europe
m = folium.Map(location=[46.5, 13.5], zoom_start=6)

folium.TileLayer('openstreetmap').add_to(m)
folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
    attr='Esri', 
    name='Esri Satellite', 
    overlay=False, 
    control=True).add_to(m)

# The city data (city name, latitude, longitude)
mesta = [
    ("Maribor", 46.5596, 15.6381),
    ("Trst", 45.6495, 13.7768),
    ("Zagreb", 45.8150, 15.9819),
    ("Gradec", 47.0707, 15.4395),
    ("Celovec", 46.6247, 14.3088),
    ("Videm", 46.0640, 13.2356),
    ("Pula", 44.8666, 13.8496),
    ("Pordenone", 45.9564, 12.6615),
    ("Szombathely", 47.2307, 16.6218),
    ("Benetke", 45.4408, 12.3155),
    ("Ljubljana", 46.0569, 14.5058)
]

for mesto, lat, lon in mesta:
    popup = folium.Popup(mesto, max_width=300, show=True)
    folium.Marker([lat, lon], popup=popup).add_to(m)

# Display the map
m.save('map.html')
