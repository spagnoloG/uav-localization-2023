import math
import json

def update_bounding_box(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_center = (min_lat + max_lat) / 2
    lon_center = (min_lon + max_lon) / 2

    delta_lat = 20 / 111  # Approximate conversion to degrees for 20 km
    delta_lon = 20 / (111 * math.cos(math.radians(lat_center)))  # Varies based on latitude

    min_lat = lat_center - delta_lat
    max_lat = lat_center + delta_lat
    min_lon = lon_center - delta_lon
    max_lon = lon_center + delta_lon

    return [min_lon, min_lat, max_lon, max_lat]

def update_bboxes_dict(bboxes_dict):
    new_bboxes_dict = {city: update_bounding_box(bbox) for city, bbox in bboxes_dict.items()}
    return new_bboxes_dict

bboxes = {
    "Ljubljana": [14.4, 46.0, 14.6, 46.1],  # min_lon, min_lat, max_lon, max_lat
    "Maribor": [15.6, 46.5, 15.7, 46.6],
    "Koper": [13.5484, 45.5452, 13.6584, 45.6052],
    "Trieste": [13.7386, 45.6333, 13.8486, 45.7033],
    "Graz": [15.4294, 47.0595, 15.5394, 47.1195],
    "Pordenone": [12.6496, 45.9564, 12.7596, 46.0164],
    "Udine": [13.2180, 46.0614, 13.3280, 46.1214],
    "Klagenfurt": [14.2765, 46.6203, 14.3865, 46.6803],
    "Pula": [13.8490, 44.8683, 13.9590, 44.9283],
    "Szombathely": [16.6056, 47.2256, 16.7156, 47.2856],
    "Venice": [12.3155, 45.4408, 12.4255, 45.5008],
}

new_bboxes = update_bboxes_dict(bboxes)
# pretty print with json
print(json.dumps(new_bboxes, indent=4))
