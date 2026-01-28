import os
from glob import glob
import xarray as xr
import cdsapi
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DatabaseGlofasAPI:
    # DATASET = "glofas river discharge: daily resolution"
    REQUEST_MONTH=["01","02","03","04","05","06","07","08","09","10","11","12"]

    def __init__(self, config_obj, path_obj):
        self.config = config_obj.config
        self.path_dict = path_obj.path_dict
        self.country_code = config_obj.country_code
        self.pecd2_code = config_obj.map[self.country_code]['PECD2']
        self.hydro_type = config_obj.hydro_type
        self.var = 'dis24'
        self.osm_method = {
            'hdam' : 'water-storage', 
            'hror' : 'run-of-the-river',
            'pumped':'water-pumped-storage'
        }
        if self.config['pred_years'] == 'None':
            self.pred_years = list(range(2015, 2025))  # default prediction years
        elif self.config['pred_years'] == 'all':
            self.pred_years = list(range(1980, 2025))
        else:
            years = self.config['pred_years'] + list(range(2015, 2025))
            self.pred_years = sorted(set(years))




    def read_cdf(self, files, dim, years, determine_local_points = True):
        paths_all = sorted(glob(files))
        if determine_local_points:
            paths =[p for p in paths_all if int(str(p).split("_")[0].split('\\')[-1]) == 2024 ]  
            #determine accuracte local points by 2024 data

        else:
            paths =[p for p in paths_all if int(str(p).split("_")[0].split('\\')[-1]) in years
                    or int(str(p).split("_")[0].split('\\')[-1])>=2015] # default prediction years should be extracted for training

        #only read necessary years to save memory

        ds = xr.open_mfdataset(
            paths,
            combine="nested",        
            concat_dim=dim,
            parallel=False,
            engine="netcdf4",
            chunks={dim: 200},        # TODO: 
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )
        return ds
        
    def request_data(self, REQUEST_YEAR, REQUEST_MONTH, path): 
        print("Request cdf remotely......")
        for year in REQUEST_YEAR:
            for month in REQUEST_MONTH:
                dataset = "cems-glofas-historical"
                request = {
                    "system_version": ["version_4_0"],
                    "hydrological_model": ["lisflood"],
                    "product_type": ["consolidated"],
                    "variable": ["river_discharge_in_the_last_24_hours"],
                    "hyear": year,
                    "hmonth": month,
                    "hday": [
                        "01", "02", "03","04", "05", "06", "07", "08", "09", "10", "11", "12",
                        "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
                        "25","26","27","28","29","30","31"
                    ],
                    "data_format": "netcdf4",
                    "download_format": "unarchived",
                    "area": [72, -25, 34, 45]  # TODO: change to bounding box of defined area
                }

                client = cdsapi.Client(url="https://ewds.climate.copernicus.eu/api")
                
                client.retrieve(dataset, request, os.path.join(path,f"{year}_{month}_00utc.nc"))
                print(f"Retrieve {year}_{month} successfully") 


    def sjoin_gdf(self, gdf1, gdf2):

        gdf=gdf1.sjoin(gdf2)
        
        if 'index_right' in gdf.columns:
            gdf=gdf.drop(columns='index_right')

        if 'index_left' in gdf.columns:
            gdf=gdf.drop(columns='index_left')

        return gdf

    def create_buffer(self, row, buffer = 0.025):

        lat, lon = row['lat'], row['lon']
        # Find the four nearest grid points (lower/upper lat, lower/upper lon)
        lat_below = lat-buffer
        lat_above = lat+buffer
        lon_below = lon-buffer
        lon_above = lon+buffer

        points = [
            (lat_below, lon_below),
            (lat_below, lon_above),
            (lat_above, lon_below),
            (lat_above, lon_above),
            (lat, lon)
        ]
        return points

    def determine_local_points(self, points, variable_name, ds):
        vals = []
        for plat, plon in points:
            val = ds[variable_name].sel(latitude=plat, longitude=plon, method='nearest').values
            vals.append(val)
        
        vals = pd.DataFrame(vals)
        vals.index= np.arange(5)
        #vals_reshaped = vals.reshape_values('dis24', pd.date_range(data['valid_time'].values[0], data['valid_time'].values[-1], freq='d').shift(-1), points)
        max_point = points[vals.mean(axis=1).idxmax()]
        print("max point", vals.mean(axis=1).idxmax()+1, max_point  )
       # lat_final, lon_final = max_point
    
        return max_point


    def extract_values(self, ds, variable_name, plant_loc):

        lats = xr.DataArray(plant_loc["lat"].to_numpy(), dims="site")
        lons = xr.DataArray(plant_loc["lon"].to_numpy(), dims="site")

        da = ds[variable_name].sel(latitude=lats, longitude=lons, method="nearest")

        return da
    
        
    

    def reshape_values(self, da, time_dim="valid_time"):
        # da dims: (valid_time, site)
        df = da.to_pandas()  # index=valid_time, columns=site
        df.columns = [f"plant_{i}" for i in range(df.shape[1])]
        return df
        

    def __geo_process(self):

        osm_data = gpd.read_file(self.path_dict['osm_filepath'])

        osm_data['plant:method'] = osm_data['plant:method'].fillna('water-storage')
        # Documentation: Assumption 1: fill NaN with water-storage  
        plants = osm_data[osm_data['plant:method'] == self.osm_method[self.hydro_type]]
        plants = plants[plants['geometry'].notna()]
        
        onshore_geo = gpd.read_file(self.path_dict['onshore_filepath'])
        pecd2_geo = onshore_geo[onshore_geo['level']=='PECD2']

        zone = pecd2_geo[pecd2_geo['id'].isin(self.pecd2_code)]
        plants_in_zone = self.sjoin_gdf(plants, zone).reset_index(drop=True)
        #TODO: check the installed capacity in OSM data
        #plants_large =...

        fig, ax = plt.subplots(figsize=(8, 8))
        zone.boundary.plot(ax=ax, color='black')
        plants_in_zone.plot(ax=ax, color='lightgrey', markersize=10)
        #osm_data.plot(ax=ax, color='lightgrey', markersize=5)
        ax.set_title(f'Hydropower plants in {self.country_code} ({self.hydro_type})')
        plt.savefig(self.path_dict['history_data_path'] / f'{self.country_code}_{self.hydro_type}_plants_location.png')
        plt.close()

        return plants_in_zone
    
    def __check_existing_disc(self):
        disc_files = sorted(glob(os.path.join(self.path_dict['history_data_path'], 
                            f"{self.country_code}_{self.hydro_type}_*glofas_discharge.csv")))

        extracted_years = self.pred_years.copy() if isinstance(self.pred_years, list) else list(self.pred_years)
        existing_disc = None

        if disc_files:
            existing_disc = pd.concat(
            [pd.read_csv(f, index_col=0, parse_dates=True) for f in disc_files],
            axis=0
            ).drop_duplicates(keep='first')
            
            # Filter to only keep years in pred_years
            existing_disc = existing_disc[existing_disc.index.year.isin(self.pred_years)]
            
            # Extract years not yet saved locally
            extracted_years = [y for y in self.pred_years if y not in existing_disc.index.year.unique()]

        return existing_disc, extracted_years

    def __check_existing_nc_file(self):
        cds_path = self.path_dict["glofas_cdf_path"]
        if not os.path.exists(cds_path):
            os.makedirs(cds_path)

        files = sorted(glob(os.path.join(cds_path, "*.nc")))
        if len(files) == 0:
            self.request_data(self.pred_years, self.REQUEST_MONTH, cds_path)
        else:
            existing_years = [os.path.splitext(os.path.basename(f))[0].split("_")[0] for f in files]
            existing_years = list(map(int, existing_years))
            missing_years = [y for y in self.pred_years if y not in existing_years]
            #TODO: add missing months check
            if missing_years:
                self.request_data(missing_years, self.REQUEST_MONTH, cds_path)
            else:
                print("GloFAS raw data already exists. Skipping...")

    def __check_existing_loc_file(self):

        loc_file = self.path_dict['history_data_path'] / f'{self.country_code}_{self.hydro_type}_plants_location.xlsx'
        
        if loc_file.exists(): 
            plant_loc = pd.read_excel(loc_file)
            plant_loc.columns = ['lon', 'lat', 'plant:output:electricity']
            print(f"{self.country_code} plant location file loaded.") 

        else:  
            plants_in_zone = self.__geo_process()
            plant_loc = pd.concat([plants_in_zone.geometry.x, plants_in_zone.geometry.y, plants_in_zone['plant:output:electricity']], axis=1)
            plant_loc.columns = ['lon', 'lat', 'plant:output:electricity']

            area_margin = False
            #TODO: if need to check the area margin?
            if area_margin:
                for i in range(plant_loc.shape[0]):
                    points = self.create_buffer(plant_loc.iloc[i], buffer = 0.025)
                    max_point = self.determine_local_points(points, self.var, self.read_cdf(f"{self.path_dict['glofas_cdf_path']}/*.nc", "valid_time", determine_local_points=True))
                    plant_loc.iloc[i] = max_point
            else:
                pass
                
            plant_loc.to_excel(loc_file, index=False)
            print(f"{self.country_code} plant location file saved.")

        return plant_loc



    def save_disc_main(self):

        # Check if data already processed before
        if not self.path_dict['disc_file'].exists():
            existing_disc, extracted_years = self.__check_existing_disc()

            if not extracted_years:
                existing_disc.to_csv(self.path_dict['disc_file'], index=True)
                
            else:
                print(f"Retrieving {self.country_code} Glofas River discharge data for years: {extracted_years}")
                # if need to request cdf by api to local
                self.__check_existing_nc_file()
                #if need to process the location
                plant_loc = self.__check_existing_loc_file()

                data = self.read_cdf(f"{self.path_dict['glofas_cdf_path']}/*.nc", "valid_time", years = extracted_years, determine_local_points=False)
                print("Glofas data read completed.")
                da = self.extract_values(data, self.var, plant_loc)
                da = da.chunk({"valid_time": -1})   
                da = da.compute()                  # one compute
                data_local = self.reshape_values(da)

                if existing_disc is not None:
                    data_local = pd.concat([existing_disc, data_local], axis=0).sort_index()
                else :
                    pass

                data_local.to_csv(self.path_dict['disc_file'], index=True)
                print(f"{self.country_code} Glofas River discharge data: Done")

        else:
            print(f"{self.country_code} Glofas River discharge data already exists. Skipping...")