# External Data: Lat/Lon/Alt Dictionary for Cities based on Nominatim, ArcGIS & Open-Meteo

Hi everyone! 👋

To comply with the external data rules of the competition (specifically the "Equally accessible" requirement), I am sharing a pre-generated dictionary containing the Latitude, Longitude, and Altitude for all 509 cities listed in the `Cities.csv` file provided by the host. 

I generated this mapping to help build geographical features (like distance traveled or elevation advantage) without having to query APIs during the notebook run. This is especially useful since internet access might be restricted or APIs might face rate limits during the submission process.

### How it was generated:
1. **Latitude/Longitude**: Fetched primarily using the **Nominatim** geocoding service (based on OpenStreetMap). For cities where Nominatim failed (like small towns or specific regions), I used a fallback to the **ArcGIS** geocoder, which provides excellent coverage.
2. **Altitude**: Fetched using the **Open-Meteo API** (Elevation API) based on the coordinates obtained above.
*All services used are completely free and require no authentication keys.*

### How to use it:
You can simply copy and paste the `CITY_GEO_DICT` directly into your preprocessing or modeling notebooks. No internet connection is needed during the run!

```python
# The format of the dictionary is:
# { CityID: {'Lat': float, 'Lon': float, 'Alt': float} }

# Example usage:
# elevation = CITY_GEO_DICT[3001]['Alt']

CITY_GEO_DICT = {
    # ... (Paste the generated dictionary output from the notebook here) ...
}
```

### Generation Notebook Context
If you would like to reproduce the dictionary yourself or modify the script to pull different geographic features, here is the core logic from my Jupyter Notebook used to generate the data:

```python
!pip install geopy geocoder requests -q
import pandas as pd
import geocoder
from geopy.geocoders import Nominatim
import requests
import time
import json

def get_altitude(lat, lon):
    if lat is None or lon is None:
        return None
    try:
        # Use Open-Meteo Elevation API (Free, no key required)
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return round(response.json().get('elevation', [0.0])[0], 1)
    except:
        pass
    return 0.0

def generate_geo_dictionary():
    df_cities = pd.read_csv("../provided/Cities.csv")
    geolocator = Nominatim(user_agent="mm_geo_fetcher_2026")
    city_dict = {}
    
    for idx, row in df_cities.iterrows():
        city_id, city_name, state_abb = row['CityID'], row['City'], row['State']
        
        country = "USA"
        if str(state_abb).strip().upper() == "MX": country = "Mexico"
        elif str(state_abb).strip().upper() == "PR": country = "Puerto Rico"
            
        query = f"{city_name}, {state_abb}, {country}"
        lat, lon = None, None
        
        try:
            # 1st Attempt: Nominatim
            location = geolocator.geocode(query, timeout=5)
            if location:
                lat, lon = location.latitude, location.longitude
            else:
                # 2nd Attempt: fallback to ArcGIS
                g = geocoder.arcgis(query)
                if g.ok:
                    lat, lon = g.latlng
            
            if lat is not None and lon is not None:
                alt = get_altitude(lat, lon)
                city_dict[int(city_id)] = {"Lat": round(lat, 4), "Lon": round(lon, 4), "Alt": alt}
            else:
                city_dict[int(city_id)] = {"Lat": None, "Lon": None, "Alt": None}
            
            time.sleep(1) # Prevent rate limiting
            
        except Exception as e:
            city_dict[int(city_id)] = {"Lat": None, "Lon": None, "Alt": None}
            time.sleep(2)
            
    return city_dict

city_geo_dict = generate_geo_dictionary()
```

Happy forecasting and good luck to everyone! 🏀
