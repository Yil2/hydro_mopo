# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:53:42 2024

@author: yil2
"""

import requests
import pandas as pd
from pathlib import Path

class EsettResponse:
    
    def __init__(self, config_obj):
        
        self.url="https://api.opendata.esett.com/EXP16/MBAOptions"
        self.__local_data_path =  config_obj.config['data_dir']
        self.__MBA_code=['SE1', 'SE2', 'SE3', 'SE4','FI']

    def get_mba_info(self, MBA):
        self.response = requests.get(self.url)    #check the mba code in the string of api request
        #print(response.status_code) # 200: Everything went okay, and the result has been returned (if any).
        self.mba_codelist=self.response.json()
        
        for country in self.mba_codelist:
            for mba in country["mbas"]:
                if mba["name"] == MBA:
                    return mba["code"]
    
    def eSett_request(self, area, country_code):
        assert area in self.__MBA_code, "unsupported area code"
        mba_code= self.get_mba_info(area)
        if mba_code:
            # Build the API URL dynamically
            file_path= Path(__file__).parent / self.__local_data_path / country_code/ f"{country_code}_reservoir generation.csv"
            if file_path.exists():
                hydro=pd.read_csv(file_path)
            else:
                api_url = f"https://api.opendata.esett.com/EXP16/Aggregate?end=2025-01-01T00%3A00%3A00.000Z&mba={mba_code}&mba=string&resolution=hour&start=2017-05-01T00%3A00%3A00.000Z"
                # Make the API request
                data_response = requests.get(api_url)
                data=data_response.json()
                ####store data into csv file
                data=pd.DataFrame(data)
                hydro=data[['timestamp','hydro']]
                #hydro.columns=[['Date CET/CEST','hydro']]
                hydro.to_csv(file_path, index=False)

            hydro=hydro.set_index(hydro['timestamp'], inplace=False, drop=True)
            hydro=hydro.drop(columns='timestamp')

            print(f"Retrieve eSett data: {country_code}_generation--->Finished")
            
        else:
            print(f"MBA '{area}' not found.")   
            file_path=None
            
        return pd.DataFrame(hydro)
            
        
