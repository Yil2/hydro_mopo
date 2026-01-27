# entsoe data request API package 
import sys
from entsoe import EntsoePandasClient
#from entsoe import EntsoeRawClient
from pandas import Timestamp,read_csv
from pathlib import Path
import pandas as pd

###IMPROVEMENT: create the folder


class EntsoeDataProcess:

    def __init__(self, config_obj, **kwargs):
        self.api_key = kwargs['api_key']
        self.client = EntsoePandasClient(api_key=self.api_key)
        # entsoe API parameters
        self.__prstype = {"hydro_pumped_storage"     : "B10",
                          "hydro_river_and_poundage" : "B11",
                          "hydro_water_reservoir"    : "B12"}
        
        self.__area_code = ['DE_50HZ', 'AL', 'DE_AMPRION', 'AT', 'BY', 'BE', 'BA', 'BG', 
                            'CZ_DE_SK', 'HR', 'CWE', 'CY', 'CZ', 'DE_AT_LU', 'DE_LU',
                            'DK', 'DK_1', 'DK_1_NO_1', 'DK_2', 'DK_CA', 'EE', 'FI', 'MK', 
                            'FR', 'DE', 'GR', 'HU', 'IS', 'IE_SEM', 'IE', 'IT', 'IT_SACO_AC', 
                            'IT_CALA', 'IT_SACO_DC', 'IT_BRNN', 'IT_CNOR', 'IT_CSUD', 'IT_FOGN',
                            'IT_GR', 'IT_MACRO_NORTH', 'IT_MACRO_SOUTH', 'IT_MALTA', 'IT_NORD',
                            'IT_NORD_AT', 'IT_NORD_CH', 'IT_NORD_FR', 'IT_NORD_SI', 'IT_PRGP',
                            'IT_ROSN', 'IT_SARD', 'IT_SICI', 'IT_SUD', 'RU_KGD', 'LV', 'LT',
                            'LU', 'LU_BZN', 'MT', 'ME', 'GB', 'GB_IFA', 'GB_IFA2', 'GB_ELECLINK', 
                            'UK', 'NL', 'NO_1', 'NO_1A', 'NO_2', 'NO_2_NSL', 'NO_2A', 'NO_3',
                            'NO_4', 'NO_5', 'NO', 'PL_CZ', 'PL', 'PT', 'MD', 'RO', 'RU', 'SE_1', 
                            'SE_2', 'SE_3', 'SE_4', 'RS', 'SK', 'SI', 'GB_NIR', 'ES', 'SE', 'CH', 
                            'DE_TENNET', 'DE_TRANSNET', 'TR', 'UA', 'UA_DOBTPP', 'UA_BEI', 'UA_IPS', 
                            'XK', 'DE_AMP_LU']
        
        self.__local_data_path = config_obj.config['data_dir']

        self.__time_zone = "UTC"

        self.input_code = config_obj.country_code
        #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # update local data
    # 
    def entsoe_request(self, data_type: str, area: str, start_date: str, end_date: str, country_code: str) -> pd.DataFrame:
        assert area in self.__area_code, "Unsupported area code"

        start_time = pd.to_datetime(str(start_date), format="%Y%m%d", utc=True)
        end_time = pd.to_datetime(str(end_date), format="%Y%m%d", utc=True)

        data_config = {
            "Reservoir rate": {
                "file_suffix": "reservoir rate",
                "query_func": lambda: self.client.query_aggregate_water_reservoirs_and_hydro_storage(area, start=start_time, end=end_time)
            },
            "Reservoir generation": {
                "file_suffix": "reservoir generation",
                "query_func": lambda: self.client.query_generation(area, start=start_time, end=end_time, psr_type=self.__prstype["hydro_water_reservoir"])
            },
            "Pumped Storage": {
                "file_suffix": "pump_generation",
                "query_func": lambda: self.client.query_generation(area, start=start_time, end=end_time, psr_type=self.__prstype["hydro_pumped_storage"])
            },
            "Run of river": {
                "file_suffix": "ror_generation",
                "query_func": lambda: self.client.query_generation(area, start=start_time, end=end_time, psr_type=self.__prstype["hydro_river_and_poundage"])
            }
        }

        if data_type not in data_config:
            raise ValueError("Unsupported data request.")

        config = data_config[data_type]
        file_name = f"{country_code}_{start_date}_{end_date}_{config['file_suffix']}.csv"
        data_path = Path(__file__).parent / self.__local_data_path / country_code / file_name

        if data_path.exists():
            print(f"Retrieve entsoe data: {country_code}_{data_type} ---> Already exist locally")
            requested_data = pd.read_csv(data_path, index_col=0)
            requested_data.index = pd.to_datetime(requested_data.index, errors='coerce', utc=True)
        else:
            requested_data = config["query_func"]()
            #Cases with bad datacolumns for some areas
            if area == 'IT_CSUD':
                dam_gen = data_config["Reservoir generation"]["query_func"]()
            else:
                dam_gen = None

            requested_data = self.__format_process(data_type, self.input_code, requested_data, dam_gen)
            requested_data.to_csv(data_path)

        print(f"Retrieve entsoe data: {country_code}_{data_type} ---> Finished")
        return pd.DataFrame(requested_data)


    # local_data_list 
    def local_data_list (self):
        file_path_list = []
        for file_path in Path(__file__).parent.joinpath(self.__local_data_path).glob("*.csv"):
            file_path_list.append(str(file_path))
            print(file_path.name)
        return file_path_list
    
    # read_local_data
    def read_local_data (self, file_path):
        return read_csv(file_path)
    
    
    def request_price(self, area, start_date, end_date,country_code):
        #client = EntsoeRawClient(api_key=self.api_key)
        assert area in self.__area_code, "unsupported area code"
        
        start = Timestamp(start_date, tz=self.__time_zone)
        end = Timestamp(end_date, tz=self.__time_zone)   
        request_price=self.client.query_day_ahead_prices(area, start=start, end=end)
        
        print(f"Retrieve entsoe data: {country_code}_price--->Finished")
    
        file_name = f"{country_code}_{start_date}_{end_date}_price.csv"
        data_path = Path(__file__).parent / self.__local_data_path / country_code / file_name
        request_price.to_csv(data_path)
        
        return pd.DataFrame(request_price)


    def __format_process(self, data_type, area, data_df, dam_gen=None):
    
        if data_type == "Reservoir generation":
            if area in ['FR','ITCS','ITSI']:
                data_df['Updated Generation'] = (
                data_df[('Hydro Water Reservoir', 'Actual Aggregated')].fillna(0) +
                data_df["Hydro Water Reservoir"].fillna(0)
                )
                data_df= data_df['Updated Generation']
            elif area in ['PT', 'AT']:
                data_df['Updated Generation'] = data_df[('Hydro Water Reservoir', 'Actual Aggregated')].fillna(0)
                data_df= data_df['Updated Generation']
            else:
                pass

        elif data_type == "Run of river":
            ror_filled = data_df.fillna(0)

            if area in ['AT','DK','IE','PT']:
                data_df['ror'] = (ror_filled[('Hydro Run-of-river and poundage', 'Actual Aggregated')] + 
                            ror_filled[('Hydro Run-of-river and poundage', 'Actual Consumption')])
                data_df = pd.DataFrame(data_df['ror'])

            elif area in ['FI','FR','ITSA','SI']:
                data_df['ror'] = (ror_filled[('Hydro Run-of-river and poundage', 'Actual Aggregated')] + 
                            ror_filled['Hydro Run-of-river and poundage'])
                data_df = pd.DataFrame(data_df['ror'])

            elif area == 'ITCS':
                # ITSI has connection error of entsoe, but little reservoir generation, neglect
                dam_gen_filled = dam_gen.fillna(0)

                data_df['ror'] = (ror_filled['Hydro Run-of-river and poundage'] + 
                            dam_gen_filled[('Hydro Water Reservoir', 'Actual Aggregated')] + 
                            dam_gen_filled['Hydro Water Reservoir'])
                
                data_df = pd.DataFrame(data_df['ror'])
            else:
                pass
        else:
            pass

        return data_df