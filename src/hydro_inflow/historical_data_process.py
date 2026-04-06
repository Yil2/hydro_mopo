import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from hydro_inflow.database_eSett_api import EsettResponse
from hydro_inflow.database_entsoe_api import EntsoeDataProcess



class FetchInflow():
    """Fetch Inflow for hdam and hror types"""
    def __init__(self, config_obj, path_obj, args):
        self.esett_country_list = ['SE1', 'SE2', 'SE3', 'SE4', 'FI']
        self.combined_inflow_codes = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'ITSI', 'ITSU', 'ITSA']
        self.pump_country_list = []
            #remove 'BG', 'GR', 'CH'
            #remove 'AT', 'ES', 'FR', 'ITN1','PT'
        self.esett_obj = EsettResponse(config_obj)
        self.entsoe_obj = EntsoeDataProcess(config_obj, api_key=config_obj.config['entsoe_api_token'] )

        

    # Entry point of the api launch
    def api_run(self, path_obj, config_obj, type):
        
        if type == 'hdam':
            self.__hdam_api_run(path_obj, config_obj)
        elif type == 'hror':
            print('Fetching hror data...')
            self.__hror_api_run(path_obj, config_obj)


    def __check_data(self,input_data, freq):
        # check duplication:
        data = input_data[~input_data.index.duplicated(keep='first')]
        # check negative data:

        # data[data < 0] = 0
        # print('Negative data found and set to 0')

        #check missing data:
        if len(data.index) != len(pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)):  
            data = data.resample(freq).asfreq().interpolate(method='linear')
            print('Missing data found and filled by linear interpolation')

        return pd.DataFrame(data)




    # ------------------------- HROR ----------------------------


    def __ror_save(self, path_obj, ror, code):
        history_data_path = path_obj.path_dict['history_data_path']
        ror_path = history_data_path / f"{code}_historical_hror_inflow.csv"
        image_path = history_data_path / f"{code}_historical_ror_inflow.png"
        
        # Save CSV
        ror.to_csv(ror_path, sep=',')

        # Plot and save figure
        ror.plot(figsize=(12, 5), label='Run of River Weekly')
        plt.title(f"ROR generation in {code}")
        plt.xlabel('Time')
        plt.ylabel('ROR generation (MWh)')
        plt.legend()
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

        

    def __hror_api_run(self, path_obj, config_obj):

        code = config_obj.country_code
        generated_data_file = path_obj.path_dict['data_file']
        if generated_data_file.exists():
            print('Local historical ror generation data already exists. Skipping data fetching')
        else:
            print('Local historical ror generation data does not exist. Fetching data from API...')
        
        ror = self.api_request(config_obj, code, "Run of river")
        ror.index = pd.to_datetime(ror.index)
        ror = ror.resample('h').mean()   #15mins to 1 hour
        ror = self.__check_data(ror, 'h')

        self.__ror_save(path_obj, ror, code)

        
        return ror



    # ------------------------- HDAM ----------------------------
    def __hdam_api_run(self, path_obj, config_obj):

        code = config_obj.country_code
        generated_data_file = path_obj.path_dict['data_file']

        if generated_data_file.exists():
            print('Local historical inflow data already exists. Skipping data fetching')
        else:
            print('Local historical inflow data does not exist. Fetching data from API...')

        reservoir_generation = self.api_request(config_obj, code, "Reservoir generation")
        reservoir_generation.index = pd.to_datetime(reservoir_generation.index)
        reservoir_generation = reservoir_generation.resample('h').mean()   #15mins to 1 hour
        reservoir_generation = self.__check_data(reservoir_generation, 'h')

        
        reservoir_rate = self.api_request(config_obj, code, "Reservoir rate")
        reservoir_rate.index = pd.to_datetime(reservoir_rate.index)
        reservoir_rate = reservoir_rate.resample('W-SUN').mean()   #w-sun resample
        reservoir_rate = self.__check_data(reservoir_rate, 'W-SUN')

        
        if code in self.combined_inflow_codes:
            ror_generation = self.__hror_api_run(path_obj, config_obj)
            ror_generation, reservoir_generation, _,_ = self.__time_align(ror_generation, reservoir_generation)
            reservoir_generation = reservoir_generation.add(ror_generation.iloc[:, 0], axis=0)  # add ror generation to reservoir generation to get total generation for inflow calculation
            print('Use Total reservoir generation data for inflow calculation')
            
        self.__inflow_calc_save(config_obj, path_obj, reservoir_generation, reservoir_rate, code)
        

    def api_request(self, config_obj, code, data_type):
        if data_type == "Reservoir generation":
            start_time = config_obj.map[code]['entose_generation']
        else:
            start_time = config_obj.map[code]['entose_reservoir_rate']
        
        # Get API data request start and end date
        start_date, end_date = self.__start_end_date(code, start_time)
        
        request_data = None
        
        # Reservoir generation request
        if start_date is None:
            print(f'Skipping API request for {code} {data_type} as start date is not provided')
        else:
            if data_type == "Reservoir generation":
                if code in self.esett_country_list:
                    try:
                        request_data = self.esett_obj.eSett_request(config_obj.map[code]['eSett'], code)
                    except Exception as e:
                        print(f'Fetching {code} Reservoir generation from eSett API failed: {e}')
                        sys.exit(1)
                else:
                    try:
                        request_data = self.entsoe_obj.entsoe_request(data_type, 
                                                                            config_obj.map[code]['Entsoe'], 
                                                                            start_date, end_date, code)
                    except Exception as e:
                        print(f'Fetching {code} Reservoir generation from ENTSOE API failed: {e}')
                        sys.exit(1)
            
        # Reservoir rate request
            if data_type == "Reservoir rate":
                try:
                    request_data = self.entsoe_obj.entsoe_request("Reservoir rate", 
                                                                    config_obj.map[code]['Entsoe'], 
                                                                    start_date, end_date, code)
                except Exception as e:
                    print(f'Fetching {code} Reservoir rate from ENTSOE API failed: {e}')
                    sys.exit(1)
            if data_type == "Run of river":
                try:
                    request_data = self.entsoe_obj.entsoe_request("Run of river", 
                                                                    config_obj.map[code]['Entsoe'], 
                                                                    start_date, end_date, code)
                except Exception as e:
                    print(f'Fetching {code} Run of river generation from ENTSOE API failed: {e}')
                    sys.exit(1)
            if data_type == 'Pumped Storage':
                try:
                    request_data = self.entsoe_obj.entsoe_request("Pumped Storage", 
                                                                    config_obj.map[code]['Entsoe'], 
                                                                    start_date, end_date, code)
                except Exception as e:
                    print(f'Fetching {code} Pumped Storage generation from ENTSOE API failed: {e}')
                    sys.exit(1)

        return request_data
    


    def __time_align(self, df1, df2):
        df1.index = pd.to_datetime(df1.index, utc=True)
        df2.index = pd.to_datetime(df2.index, utc=True)
        start_time = max(df1.index.min(), df2.index.min())
        end_time = min(df1.index.max(), df2.index.max())
        df1_aligned = df1[(df1.index >= start_time) & (df1.index <= end_time)]
        df2_aligned = df2[(df2.index >= start_time) & (df2.index <= end_time)]
        return df1_aligned, df2_aligned, start_time, end_time
    
    def __save_inflow_fig(self, data:pd.DataFrame, path:str, country_code:str)->None:
        data.plot(figsize=(12,5), label='Inflow Weekly')
        plt.title(f"Inflow in {country_code}")
        plt.xlabel('Time')
        plt.ylabel('Inflow (MWh)')
        plt.legend()
        plt.savefig(path, bbox_inches='tight')

    def __inflow_calc_save(self, config_obj, path_obj, reservoir_generation, reservoir_rate, code):
        """Calculate inflow and save results"""
        # Resample to weekly
        reservoir_generation = reservoir_generation.resample('w-sun').sum().shift(freq="24h").iloc[1:-1] # start from Monday 00:00
        # in case the first and last weeks are not entire weeks, remove first and last weeks directly

        reservoir_rate = reservoir_rate.resample('w-sun').sum().shift(freq="24h").iloc[1:-1] # start from Monday 00:00
        if code in self.pump_country_list: 
            pump = self.api_request(config_obj, code, "Pumped Storage")
            pump.index = pd.to_datetime(pump.index, utc=True)
            pump = pump.resample('h').mean()   #15mins to 1 hour
            pump = pump.iloc[1:]

            if code in ['BG', 'GR', 'CH']:
                generation_pump = pd.to_numeric(pump.iloc[:, 0], errors='coerce').fillna(0)

            if code in ['ES']:  #ES changing the name of generation
                generation_pump = pd.to_numeric(pump.iloc[:, 0], errors='coerce').fillna(0)-pd.to_numeric(pump.iloc[:, 2], errors='coerce').fillna(0)*0.75-pd.to_numeric(pump.iloc[:, 1], errors='coerce').fillna(0)*0.75
            else:
                generation_pump = pd.to_numeric(pump.iloc[:, 0], errors='coerce').fillna(0)-pd.to_numeric(pump.iloc[:, 1], errors='coerce').fillna(0)*0.75
                generation_pump.index = pd.to_datetime(generation_pump.index, utc=True)
                generation_pump = pd.DataFrame(generation_pump)
            
            generation_pump = generation_pump.resample('h').mean()
            generation_pump = generation_pump.resample('w-sun').sum().shift(freq="24h").iloc[1:-1] # start from Monday 00:00
            generation_pump, reservoir_generation, _,_ = self.__time_align(generation_pump, reservoir_generation)
            output_energy = reservoir_generation.iloc[:, 0] + generation_pump.iloc[:, 0]
            reservoir_generation = pd.DataFrame(output_energy, index=reservoir_generation.index, columns=['Reservoir generation'])
            # update reservoir generation by adding pump generation

        reservoir_generation, reservoir_rate, _,_ = self.__time_align(reservoir_generation, reservoir_rate)

        
        #rate_diff = reservoir_rate.diff().fillna(0)
        # if code in ['ES', 'FR', 'GR', 'CH']:
        #     threshold = True
        # else:
        #     threshold = False
        threshold = True
        if threshold:
            threshold=2
            if code == 'CH':
                threshold=1
            #Remove big fluctuation between weeks
            diff=reservoir_rate.diff().diff()
            z_scores=(diff-diff.mean())/diff.std()
            diff_mask=diff.copy()
            diff_mask[z_scores.abs()>threshold]=np.nan

            content_mask=reservoir_rate.copy()
            content_mask[diff_mask.isna()]=np.nan
            content_mask.interpolate(inplace=True)
            content_diff = content_mask.diff().dropna()
        else:
            content_diff = reservoir_rate.diff().dropna()

        reservoir_generation, content_diff, _,_ = self.__time_align(reservoir_generation, content_diff)

        inflow_weekly = content_diff.add(reservoir_generation.iloc[:, 0], axis=0) # add reservoir_generation

        inflow_path =path_obj.path_dict['data_file']
        inflow_weekly.to_csv(inflow_path, sep=',')
        print(f'Save historical inflow for {code}--->Finished')
        
        # Plot and save historical inflow figure
        fig_path = path_obj.path_dict['history_data_path']  / f'{code}_historical_inflow.png'
        self.__save_inflow_fig(inflow_weekly, str(fig_path), code)




    def __start_end_date(self, code, start_time):
        end_date = '20260101'
        
        if code in self.esett_country_list:
            start_date = '20170101'
        elif start_time == '':
            start_date = None
        else:
            start_date = start_time
        
        return start_date, end_date



    

    # ------------------------- HPUMP & PRICE API --------------------------------------------
    def pump_api_run(self, config_obj):

        code = config_obj.country_code
        print('Fetching pumping from API...')
        self.__pump_api_request(config_obj, code)
        
            
    def __pump_api_request(self, config_obj, code):
        pump_start_time = config_obj.map[code]['entose_pumped']
        pump = dates = None

        if pd.notna(pump_start_time):
            start_date = pump_start_time
            end_date = '20260101'
            dates = (start_date, end_date)
            try:
                pump = self.entsoe_obj.entsoe_request("Pumped Storage", config_obj.map[code]['Entsoe'], 
                                                    start_date, end_date, code)
                
            except Exception as e:
                print(f'Fetching {code} PUMP generation from ENTSOE API failed: {e}')
                sys.exit(1)
        else:
            print('Warning: There is no PUMP in this country!')
        return pump, dates


    def price_api_run(self, config_obj):
        code = config_obj.country_code
        print('Fetching price from API...')
        self.__price_request(config_obj, code)
        
    def __price_request(self, config_obj, code):
        start_date = '20150101'
        end_date = '20260101'
        
        try:    
            print(f"Retrieve entsoe data: {code}_price--->Start")
            self.entsoe_obj.request_price(config_obj.map[code]['Entsoe'], start_date, end_date, code)
            
        except Exception as e:
            print(f'Fetching {code} Price from ENTSOE API failed: {e}')
            sys.exit(1)