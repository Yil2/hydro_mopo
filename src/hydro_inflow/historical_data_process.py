import pandas as pd
import sys
import matplotlib.pyplot as plt
from hydro_inflow.database_eSett_api import EsettResponse
from hydro_inflow.database_entsoe_api import EntsoeDataProcess
from hydro_inflow.inflow_calculate import ReadProcessInflow as rpi
from hydro_inflow.data_check import CheckFillData as cfd



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
        self.__api_run(path_obj, config_obj, config_obj.hydro_type)
        

    # Entry point of the api launch
    def __api_run(self, path_obj, config_obj, type):
        
        if type == 'hdam':
            self.__hdam_api_run(path_obj, config_obj)
        elif type == 'hror':
            print('Fetching hror data...')
            self.__hror_api_run(path_obj, config_obj)

    # ------------------------- HROR ----------------------------
    def __hror_api_run(self, path_obj, config_obj):

        code = config_obj.country_code
        generated_data_file = path_obj.path_dict['history_data_path'] / (code + f'_historical_hror_inflow.csv')
        if generated_data_file.exists():
            print('Local historical ror generation data already exists. Skipping data fetching')
            ror = rpi.read_local_data(generated_data_file)
        
        else:
            print('Local historical ror generation data does not exist. Fetching data from API...')
            ror, dates = self.__ror_api_request(config_obj, code)
            if not ror.empty:
                ror = self.__ror_process_data(ror, code, dates, config_obj, ['Run of River Generation'])
                ror = self.__ror_check_data(ror)
                self.__ror_save(path_obj, ror, code)

        
        return ror
     
            
    def __ror_api_request(self, config_obj, code):
        ror_start_time = config_obj.map[code]['entose_ror']
        ror = dates = None

        if pd.notna(ror_start_time):
            start_date = ror_start_time
            end_date = '20250101'
            dates = (start_date, end_date)
            try:
                ror = self.entsoe_obj.entsoe_request("Run of river", config_obj.map[code]['Entsoe'], 
                                                     start_date, end_date, code)
                
                #price = self.entsoe_obj.request_price(config_obj.map[code]['Entsoe'], start_date, end_date, code)
                
            except Exception as e:
                print(f'Fetching {code} ROR generation from ENTSOE API failed: {e}')
                sys.exit(1)
        else:
            print('Warning: There is no ROR inflow in this country!')
        return ror, dates

    def __ror_process_data(self, ror, code, dates, config_obj, cols): 
        ror.columns = cols
        ror = rpi.index_date(ror, 'Run of River Generation')
        return ror

    def __ror_check_data(self, ror):
        start_time = ror.index[0]
        end_time = ror.index[-1]
        date_range = cfd.create_date_range(start_time, end_time, 'h')
        
        ror = cfd.check_duplicate_data(ror)
        ror = cfd.check_missing_data(ror, date_range)  
        ror = cfd.check_negative_data(ror)
        return ror

    def __ror_save(self, path_obj, ror, code):
        history_data_path = path_obj.path_dict['history_data_path']
        ror_path = history_data_path / f"{code}_historical_hror_inflow.csv"
        image_path = history_data_path / f"{code}_historical_ror_inflow.pdf"
        
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

        

    # ------------------------- HDAM ----------------------------



    def __hdam_api_run(self, path_obj, config_obj):

        code = config_obj.country_code
        generated_data_file = path_obj.path_dict['data_file']

        if generated_data_file.exists():
            print('Local historical inflow data already exists. Skipping data fetching')
        else:
            print('Local historical inflow data does not exist. Fetching data from API...')
        reservoir_generation, reservoir_rate = self.__api_request(config_obj, code)
        reservoir_generation, reservoir_rate = self.__process_reservior_data(reservoir_generation, reservoir_rate, code)
        reservoir_generation, reservoir_rate = self.__check_reservior_data(reservoir_generation, reservoir_rate)
        #NO needs to firstly process ROR modelled data and calibrate the RES generation by ROR generation
        if code in self.combined_inflow_codes:
            res_updated_gen_file = path_obj.path_dict['history_data_path'] / (code + f'_calib_reservoir generation.csv')
            reservoir_generation = pd.read_csv(res_updated_gen_file, index_col=0, parse_dates=True)
            print('Use calibrated reservoir generation data for inflow calculation')
            
        self.__inflow_calc_save(path_obj, reservoir_generation, reservoir_rate, code)
        

    def __api_request(self, config_obj, code):
        reser_time = config_obj.map[code]['entose_reservoir_rate']
        gen_time = config_obj.map[code]['entose_generation']

        # Get API data request start and end date
        start_date, end_date, skip_fetch = self.__start_end_date(code, gen_time, reser_time)
        
        reservoir_generation = reservoir_rate = None
        
        # Reservoir generation request
        if code in self.esett_country_list:
            try:
                reservoir_generation = self.esett_obj.eSett_request(config_obj.map[code]['eSett'], code)
            except Exception as e:
                print(f'Fetching {code} Reservoir generation from eSett API failed: {e}')
                sys.exit(1)
        elif not skip_fetch:
            try:
                reservoir_generation = self.entsoe_obj.entsoe_request("Reservoir generation", 
                                                                       config_obj.map[code]['Entsoe'], 
                                                                       start_date, end_date, code)
            except Exception as e:
                print(f'Fetching {code} Reservoir generation from ENTSOE API failed: {e}')
                sys.exit(1)
        
        # Reservoir rate request
        if not skip_fetch:
            try:
                reservoir_rate = self.entsoe_obj.entsoe_request("Reservoir rate", 
                                                                 config_obj.map[code]['Entsoe'], 
                                                                 start_date, end_date, code)
            except Exception as e:
                print(f'Fetching {code} Reservoir rate from ENTSOE API failed: {e}')
                sys.exit(1)

        return reservoir_generation, reservoir_rate

    def __start_end_date(self, code, gen_time, reser_time):
        skip_fetch = False
        end_date = '20260101'
        
        if code in self.esett_country_list:
            start_date = '20170101'
        elif gen_time == '' or reser_time == '':
            print(f'ENTSOE generation time: {gen_time}, ENTSOE reservoir rate time: {reser_time} from country: {code} is empty, skipped fetching')
            skip_fetch = True
            start_date = None
        else:
            start_date = max(int(gen_time), int(reser_time))  
        
        return start_date, end_date, skip_fetch


    def __process_reservior_data(self, reservoir_generation, reservoir_rate, code):
        # Clean column names
        reservoir_generation.columns = ['Reservoir generation']
        reservoir_generation = rpi.index_date(reservoir_generation, 'Reservoir generation')
        
        reservoir_rate.columns = ['Reservoir rate']
        reservoir_rate = rpi.index_date(reservoir_rate, 'Reservoir rate')
        reservoir_rate.index = reservoir_rate.index.normalize()
        
        return reservoir_generation, reservoir_rate

    def __check_reservior_data(self, reservoir_generation, reservoir_rate):
        # Check reservoir rate (some countries do not use midnight time)
        reservoir_rate.index = reservoir_rate.index.normalize()
        start_time = max(reservoir_generation.index.min(), reservoir_rate.index.min())
        end_time = min(reservoir_generation.index.max(), reservoir_rate.index.max())
        date_range = cfd.create_date_range(start_time, end_time, 'W-SUN')
        
        reservoir_rate = cfd.check_duplicate_data(reservoir_rate)
        reservoir_rate = cfd.check_missing_data(reservoir_rate, date_range, 'W-SUN')    
        reservoir_rate = cfd.check_negative_data(reservoir_rate)  # Fill zero and negative values

        # Check reservoir generation
        reservoir_generation = reservoir_generation.resample('h').mean()
        reservoir_generation = cfd.check_duplicate_data(reservoir_generation)
        reservoir_generation = cfd.check_missing_data(reservoir_generation, date_range, 'h')
        reservoir_generation = cfd.check_negative_data(reservoir_generation)
        
        return reservoir_generation, reservoir_rate


    def __inflow_calc_save(self, path_obj, reservoir_generation, reservoir_rate, code):
        """Calculate inflow and save results"""
        # Resample to weekly
        reservoir_generation = reservoir_generation.resample('h').mean()
        reservoir_generation = reservoir_generation.resample('w-sun').sum().shift(freq="24h").iloc[1:-1] # start from Monday 00:00
        # in case the first and last weeks are not entire weeks, remove first and last weeks directly

        reservoir_rate = rpi.resample_data(reservoir_rate, 'Reservoir rate', 'W-MON')


        if code in self.pump_country_list:
            import os
            pump = pd.read_csv(os.path.join(path_obj.path_dict['history_data_path'], code+"_pump.csv"),  index_col=0, parse_dates=True)
            pump.index = pd.to_datetime(pump.index, utc=True)
            pump = pump.iloc[1:]
            # if code in ['BG', 'GR', 'CH']:
            #     generation_pump = pd.to_numeric(pump.iloc[:, 0], errors='coerce').fillna(0)

            if code in ['ES']:  #ES changing the name of generation
                generation_pump = pd.to_numeric(pump.iloc[:, 0], errors='coerce').fillna(0)-pd.to_numeric(pump.iloc[:, 2], errors='coerce').fillna(0)*0.75-pd.to_numeric(pump.iloc[:, 1], errors='coerce').fillna(0)*0.75
            else:
                generation_pump = pd.to_numeric(pump.iloc[:, 0], errors='coerce').fillna(0)-pd.to_numeric(pump.iloc[:, 1], errors='coerce').fillna(0)*0.75
                generation_pump.index = pd.to_datetime(generation_pump.index, utc=True)
                generation_pump = pd.DataFrame(generation_pump)
            
            generation_pump = generation_pump.resample('h').mean()
            generation_pump = generation_pump.resample('w-sun').sum().shift(freq="24h").iloc[1:-1] # start from Monday 00:00
            generation_pump, reservoir_rate, inflow_start, inflow_end = rpi.time_align(generation_pump, reservoir_rate)
            reservoir_generation, reservoir_rate, inflow_start, inflow_end = rpi.time_align(reservoir_generation, reservoir_rate)

            content_diff = reservoir_rate.diff().dropna()
        
            generation_align = reservoir_generation.drop(reservoir_generation.index[0])
            generation_align_pump = generation_pump.drop(generation_pump.index[0])

            inf_original = content_diff.add(generation_align.iloc[:, 0], axis=0)   
            inf_original = inf_original.add(generation_align_pump.iloc[:, 0], axis=0)      
            inf_original[inf_original<0]=None
            inflow_weekly = inf_original.interpolate(method='linear')

        else:

        # Align time series and calculate inflow
            # if code in self.combined_inflow_codes:
            #     # NO1-5 starts from 2022.
            #     reservoir_generation = reservoir_generation[reservoir_generation.index.year >= 2022]
            #     reservoir_rate = reservoir_rate[reservoir_rate.index.year >= 2022]
            # else:
            #     pass

            reservoir_generation, reservoir_rate, inflow_start, inflow_end = rpi.time_align(reservoir_generation, reservoir_rate)
            inflow_weekly = rpi.inflow_calculation(reservoir_generation, reservoir_rate)
        
        #inflow_weekly = cfd.check_negative_data(inflow_weekly)
        # Save historical inflow data
     
        inflow_path =path_obj.path_dict['data_file']
        inflow_weekly.to_csv(inflow_path, sep=',')
        print(f'Save historical inflow for {code}--->Finished')
        
        # Plot and save historical inflow figure
        fig_path = path_obj.path_dict['history_data_path']  / f'{code}_{inflow_start}_{inflow_end}_inflow.pdf'
        rpi.save_inflow_fig(inflow_weekly, str(fig_path), code)


    def price_request(self, config_obj, code):
        start_date = '20150101'
        end_date = '20260101'
        
        try:    
            print(f"Retrieve entsoe data: {code}_price--->Start")
            self.entsoe_obj.request_price(config_obj.map[code]['Entsoe'], start_date, end_date, code)
            
        except Exception as e:
            print(f'Fetching {code} Price from ENTSOE API failed: {e}')
            sys.exit(1)
        else:
            print('The area is not modelled in HROR type')


    # ------------------------- HPUMP ----------------------------
    def pump_api_run(self, config_obj):

        code = config_obj.country_code
        print('Fetching pumping from API...')
        self.__pump_api_request(config_obj, code)
        
            
    def __pump_api_request(self, config_obj, code):
        pump_start_time = config_obj.map[code]['entose_pumped']
        pump = dates = None

        if pd.notna(pump_start_time):
            start_date = pump_start_time
            end_date = '20250101'
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


        
