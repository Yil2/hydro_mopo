import os
from glob import glob
import xarray as xr
import cdsapi
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DatabaseGlofasAPI:
    #DATASET = "glofas river discharge: daily resolution"
    #TODO: change to flexible year and month
    REQUEST_YEAR=["1991","1992","1993","1994","1995","1996","1997","1998","1999","2000",
                "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
                "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020",
                "2021", "2022", "2023","2024"]
    
    REQUEST_MONTH=["01","02","03","04","05","06","07","08","09","10","11","12"]

    def __init__(self, config_obj, path_obj):
        self.config = config_obj.config
        self.path_dict = path_obj.path_dict
        self.country_code = config_obj.country_code
        self.pecd2_code = config_obj.map[self.country_code]['PECD2']
        self.hydro_type = config_obj.hydro_type
        self.osm_method = {
            'hdam' : 'water-storage', 
            'hror' : 'run-of-the-river',
            'pumped':'water-pumped-storage'
        }


    def read_cdf(self, files, dim):
        def process_one_path(path):
            cdf_path=path
            ds = xr.open_dataset(cdf_path, engine="netcdf4")
            ds_copy = ds.load()  # Force load all data into memory
            ds.close()           # Manually close the file
            return ds_copy

        paths=sorted(glob(files))
        dataset = [process_one_path(path) for path in paths]

        return xr.concat(dataset, dim=dim)
        
    def request_data(self, REQUEST_YEAR, REQUEST_MONTH ): 

        path = self.path_dict["glofas_cdf_path"]
        if not os.path.exists(path):
            os.makedirs(path)

        files = sorted(glob(os.path.join(path, "*.nc")))
        if len(files) == 0:
            request_cdf = True
        else:
            request_cdf = False


        if request_cdf:
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
            print("Reading cdf locally......")
            data = self.read_cdf(f"{path}/*.nc", "valid_time")
            return data
        
        else:
            print("Reading cdf locally......")
            data = self.read_cdf(f"{path}/*.nc", "valid_time")
            return data

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



    def extract_values(self, row, variable_name, ds):
        points = self.create_buffer(row)
        vals = []
        for plat, plon in points:
            val = ds[variable_name].sel(latitude=plat, longitude=plon, method='nearest').values
            vals.append(val)
        
        vals = pd.DataFrame(vals)
        #print(vals)
        vals.index= np.arange(5)
        #vals_reshaped = vals.reshape_values('dis24', pd.date_range(data['valid_time'].values[0], data['valid_time'].values[-1], freq='d').shift(-1), points)
        max_point = points[vals.mean(axis=1).idxmax()]
        print("max point", vals.mean(axis=1).idxmax()+1, max_point,  )
        lat_final, lon_final = max_point
    
        return ds[variable_name].sel(latitude=lat_final, longitude=lon_final, method='nearest').values

    def reshape_values(self,var, time_range, grids):
            reshape_dict={}
            i=0
            for row in grids[var]:
                reshape_dict[f"plant_{i}"]=row
                i=i+1
            df_reshaped=pd.DataFrame(reshape_dict, index=pd.to_datetime(time_range))

            return df_reshaped
        

    def geo_process(self):

        osm_data = gpd.read_file(self.path_dict['osm_filepath'])

        osm_data['plant:method'] = osm_data['plant:method'].fillna('water-storage')
        # Documentation: Assumption 1: fill NaN with water-storage
        
        plants = osm_data[osm_data['plant:method'] == self.osm_method[self.hydro_type]]


        plants = plants[plants['geometry'].notna()]
        
        onshore_geo = gpd.read_file(self.path_dict['onshore_filepath'])
        pecd2_geo = onshore_geo[onshore_geo['level']=='PECD2']

        zone = pecd2_geo[pecd2_geo['id'].isin(self.pecd2_code)]
        plants_in_zone = self.sjoin_gdf(plants, zone).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        zone.boundary.plot(ax=ax, color='black')
        plants_in_zone.plot(ax=ax, color='lightgrey', markersize=10)
        #osm_data.plot(ax=ax, color='lightgrey', markersize=5)
        ax.set_title(f'Hydropower plants in {self.country_code} ({self.hydro_type})')
        plt.savefig(self.path_dict['history_data_path'] / f'{self.country_code}_{self.hydro_type}_plants_location.png')
        plt.close()

        return plants_in_zone
    
    def save_disc_main(self):

        if self.path_dict['disc_file'].exists():
            print(f"File {self.path_dict['disc_file']} already exists. Skipping...")
        else:
            print(f"Retriving {self.country_code} Glofas River discharge data...")

            plants_in_zone = self.geo_process()
            data = self.request_data(self.REQUEST_YEAR, self.REQUEST_MONTH)

            plant_loc = pd.concat([plants_in_zone.geometry.x, plants_in_zone.geometry.y], axis=1)
            plant_loc.columns = ['lon', 'lat']

            var = 'dis24'

            plant_loc[var] = plant_loc.apply(
                lambda row: self.extract_values(row, var, data), axis=1)

            data_local = self.reshape_values(var, pd.date_range(data['valid_time'].values[0], data['valid_time'].values[-1], freq='d').shift(-1), plant_loc)

            data_local.to_csv(self.path_dict['disc_file'], index=True)

            print(f"{self.country_code} Glofas River discharge data: Done")
    
