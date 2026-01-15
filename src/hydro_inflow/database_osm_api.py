import requests
import time
import json

class OverpassAPI:
    def __init__(self, config_obj, path_obj):
        self.config = config_obj.config
        self.path_dict = path_obj.path_dict
        self.country_code = config_obj.country_code
        self.iso_code = config_obj.map[self.country_code]['iso_code']
        self.url = "https://overpass-api.de/api/interpreter"
    

    def get_overpass_query(self, iso_code: str) -> str:

        """Generate an Overpass query for hydropower plants in a given country."""
        # Hydropower plants:
        #   power=plant + plant:source=hydro
        # OR:
        #   generator:source=hydro (sometimes without power=plant)
        query = f"""
                [out:json][timeout:180];
                area["ISO3166-1"="{iso_code}"][admin_level=2]->.area;

                (
                nwr["power"="plant"]["plant:source"="hydro"](area.area);
                nwr["generator:source"="hydro"](area.area);
                );

                out center tags;
                """
        
        return query

    def overpass_request(self, query: str, url: str, max_retries: int = 5):
        
        for attempt in range(1, max_retries + 1):
            r = requests.post(url, data={"data": query}, timeout=300)
            if r.status_code == 200:
                return r.json()
            
            sleep_s = min(60, 2 ** attempt)
            time.sleep(sleep_s)
        r.raise_for_status()

    def overpass_to_geojson(self, data: dict) -> dict:
        """Convert Overpass JSON to GeoJSON."""

        features = []

        for el in data.get("elements", []):
            tags = el.get("tags", {})
            # Determine a representative coordinate:
            # - nodes have lat/lon
            # - ways/relations may have "center" because we used `out center`
            if el.get("type") == "node":
                lon, lat = el["lon"], el["lat"]
            else:
                center = el.get("center")
                if not center:
                    continue
                lon, lat = center["lon"], center["lat"]

            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    **tags,
                    "osm_type": el.get("type"),
                    "osm_id": el.get("id"),
                },
            }
            features.append(feature)

        return {"type": "FeatureCollection", "features": features}
    
    def save_osm_data_main(self):

        file_path = self.path_dict['osm_filepath']
        
        if file_path.exists():
            print(f"File {file_path} already exists. Skipping...")

        else:
            print(f"Retrieving OSM data for country code: {self.country_code}...")
            OVERPASS_QUERY = self.get_overpass_query(self.iso_code)  # Sweden
            data = self.overpass_request(OVERPASS_QUERY, self.url)
            geojson = self.overpass_to_geojson(data)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(geojson, f, ensure_ascii=False)

            print(f"Saved {len(geojson['features'])} hydropower plants to {file_path}")
       