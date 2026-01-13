import pandas as pd
import sys
import matplotlib.pyplot as plt
from hydro_inflow.database_eSett_api import EsettResponse
from hydro_inflow.database_entsoe_api import EntsoeDataProcess
from hydro_inflow.inflow_cal import ReadProcessInflow as rpi
from hydro_inflow.data_check import CheckFillData as cfd

class FetchInflow():
    """Fetch Inflow for hdam and hror types"""
    def __init__(self, config_obj, path_obj, args):
        self.esett_country_list = ['SE1', 'SE2', 'SE3', 'SE4', 'FI']
        self.esett_obj = EsettResponse(config_obj)
        self.entsoe_obj = EntsoeDataProcess(config_obj, api_key=config_obj.config['entsoe_api_token'])
        self.__api_run(path_obj, config_obj, config_obj.hydro_type, args.equivalent_model)

    # Entry point of the api launch
    def __api_run(self, path_obj, config_obj, type, model_on):
        
        if type == 'hdam':
            self.__hdam_api_run(path_obj, config_obj, model_on)
        elif type == 'hror':
            print('Fetching hror data...')
            self.__hror_api_run(path_obj, config_obj)

    # ------------------------- HROR ----------------------------
    def __hror_api_run(self, path_obj, config_obj):

        code = config_obj.country_code
        generated_data_file = path_obj.path_dict['history_data_path'] / (code + '_historical_ror_generation.csv')

        if generated_data_file.exists():
            print('Local historical ror generation data already exists. Skipping data fetching')
        else:
            print('Local historical ror generation data does not exist. Fetching data from API...')
            ror, dates = self.__ror_api_request(config_obj, code)
            if not ror.empty:
                ror = self.__ror_process_data(ror, code, dates, config_obj)
                ror = self.__ror_check_data(ror)
                self.__ror_save(path_obj, ror, code)

    def __ror_api_request(self, config_obj, code):

        ror_start_time = config_obj.map[code]['entose_ror']
        ror = dates = None

        if pd.notna(ror_start_time):
            end_date='20250101'
            start_date=ror_start_time
            dates = (start_date, end_date)
            # entsoe api
            try:
                ror=self.entsoe_obj.entsoe_request("Run of river", config_obj.map[code]['Entsoe'], start_date, end_date, code)
            except:
                print(f'Fetching {code} ror generation from entose api failed!')
                sys.exit(1)
        else:
            print('Warning: There is no ror inflow in this country!')
        return ror, dates

    def __ror_process_data(self, ror, code, dates, config_obj):

        if code in ['AT','DK','IE','PT']:
            ror['ror']=ror.fillna(0)[('Hydro Run-of-river and poundage',  'Actual Aggregated')]+ror.fillna(0)[('Hydro Run-of-river and poundage',  'Actual Consumption')]
            ror=ror['ror']
            ror=pd.DataFrame(ror)

        elif code in ['FI','FR','ITSA','SI']:
            ror['ror']=ror.fillna(0)[('Hydro Run-of-river and poundage',  'Actual Aggregated')]+ror.fillna(0)[
                'Hydro Run-of-river and poundage']
            ror=pd.DataFrame(ror['ror'])

        elif code in ['ITCS']:   #ITSI has connection error of entsoe, but litte reservoir generation, negelect
            # entsoe api
            try:
                dam_gen=self.entsoe_obj.entsoe_request("Reservoir generation", config_obj.map[code]['Entsoe'], dates[0], dates[1] ,code)
            except:
                print(f'Fetching {code} Reservior generation from entose api failed!')
                sys.exit(1)
            ror['ror']=ror.fillna(0)['Hydro Run-of-river and poundage']+dam_gen.fillna(0)[
                ('Hydro Water Reservoir',  'Actual Aggregated')]+ dam_gen.fillna(0)["Hydro Water Reservoir"]
            ror=pd.DataFrame(ror['ror'])
        else:
            ror=pd.DataFrame(ror)
        ror.columns=['Run of River Generation']
        ror=rpi.index_date(ror, 'Run of River Generation')
        return ror

    def __ror_check_data(self, ror):
        start_time= ror.index[0]
        end_time=ror.index[-1]
        date_range=cfd.create_date_range(start_time, end_time,'h')
        ror=cfd.check_duplicate_data(ror)
        ror=cfd.check_missing_data(ror, date_range)  
        ror=cfd.check_negative_data(ror)
        return ror

    def __ror_save(self, path_obj, ror, code):

        history_data_path = path_obj.path_dict['history_data_path']
        ror_path = path_obj.path_dict['data_file']
        image_path = history_data_path / f"{code}_historical_ror_inflow.pdf"
        ror.to_csv(ror_path,sep=',')

        ror.plot(figsize=(12,5), label='Run of River Weekly')
        plt.title(f"ROR generation in {code}")
        plt.xlabel('Time')
        plt.ylabel('ROR generation (MWh)')
        plt.legend()
        plt.savefig (image_path, bbox_inches='tight')
        plt.close()

    # ------------------------- HDAM ----------------------------

    def __set_year(self, code):
        if code in ['HR']:
            return 2021
        else:
            return 2018

    def __hdam_api_run(self, path_obj, config_obj, model_on):

        code = config_obj.country_code
        year = self.__set_year(code)
        generated_data_file = path_obj.path_dict['data_file']

        if generated_data_file.exists():
            print('Local historical inflow data already exists. Skipping data fetching')
        else:
            print('Local historical inflow data does not exist. Fetching data from API...')
            reservoir_generation, reservoir_rate = self.__api_request(config_obj, code)
            reservoir_generation, reservoir_rate = self.__process_reservior_data(reservoir_generation, reservoir_rate, code)
            reservoir_generation, reservoir_rate = self.__check_reservior_data(reservoir_generation, reservoir_rate)
            self.__inflow_calc_save(path_obj, reservoir_generation, reservoir_rate, code)
        if model_on:  #TODO: remove this funcatinality later
            print('Equivalent model is selected to generate.')
            self.__create_equivalent_model(reservoir_generation, reservoir_rate, path_obj, config_obj, code, year)

    def __api_request(self, config_obj, code):

        reser_time = config_obj.map[code]['entose_reservoir_rate']
        gen_time = config_obj.map[code]['entose_generation']

        # Get Api Data request start and end date
        start_date, end_date, skip_fetch = self.__start_end_date(code, gen_time, reser_time)
        # Reservior request
        if code in self.esett_country_list:
            # esett api
            try:
                reservoir_generation=self.esett_obj.eSett_request(config_obj.map[code]['eSett'], code)
            except:
                print(f'Fetching {code} Reservior generation from eSett api failed!')
                sys.exit(1)
        elif not skip_fetch:
            # entsoe api
            try:
                reservoir_generation=self.entsoe_obj.entsoe_request("Reservoir generation", config_obj.map[code]['Entsoe'], start_date, end_date ,code)
            except:
                print(f'Fetching {code} Reservior generation from entose api failed!')
                sys.exit(1)
        # Generation request
        # entsoe api
        if not skip_fetch:
            try:
                reservoir_rate=self.entsoe_obj.entsoe_request("Reservoir rate", config_obj.map[code]['Entsoe'], start_date, end_date, code)
            except:
                print(f'Fetching {code} Reservior rate from entose api failed!')
                sys.exit(1)

        return reservoir_generation, reservoir_rate

    def __start_end_date(self, code, gen_time, reser_time):
        skip_fetch = False
        if code in self.esett_country_list:
            start_date='20170101'
            end_date='20260101'    
        elif gen_time == '' or reser_time == '':
            print(f'entose_generation time: {gen_time}, entose_reservoir_rate time: {reser_time} from country :[{code}] is empty, skipped fetching')
            skip_fetch = True
        else:
            start_date = max(int(gen_time), int(reser_time))   #TODO: DECIDE IF USE 2015 DATA
            end_date = '20260101'
        return start_date, end_date, skip_fetch


    def __process_reservior_data(self, reservoir_generation, reservoir_rate, code):

        if code in ['FR','ITCS','ITSI']:
            reservoir_generation['New_gen'] = (
                    reservoir_generation["('Hydro Water Reservoir', 'Actual Aggregated')"].fillna(0) +
                    reservoir_generation["Hydro Water Reservoir"].fillna(0)
                    )
            
            reservoir_generation.drop(columns=[col for col in reservoir_generation.columns if col != 'New_gen'], inplace=True)
        elif code in ['PT']:
            reservoir_generation['New_gen'] = reservoir_generation[
                ('Hydro Water Reservoir', 'Actual Aggregated')] #negelect the consumption here as they are so small(1-4MWh)
            reservoir_generation=pd.DataFrame(reservoir_generation['New_gen'])
    
        reservoir_generation.columns=['Reservoir generation']
        reservoir_generation=rpi.index_date(reservoir_generation,'Reservoir generation')
    
        reservoir_rate.columns=['Reservoir rate']
        reservoir_rate=rpi.index_date(reservoir_rate, 'Reservoir rate')
        reservoir_rate.index = reservoir_rate.index.normalize()
        
        return reservoir_generation, reservoir_rate

    def __check_reservior_data(self, reservoir_generation, reservoir_rate):

        start_time= reservoir_generation.index[0]
        end_time=reservoir_generation.index[-1]
        date_range=cfd.create_date_range(start_time, end_time,'h')
        # reservoir_generation=cfd.check_duplicate_data(reservoir_generation)
        # reservoir_generation=cfd.check_missing_data(reservoir_generation, date_range)  
        # reservoir_generation=cfd.check_negative_data(reservoir_generation)
        
        reservoir_rate.index = reservoir_rate.index.normalize()  #some countries do not use midnight time
        start_time= reservoir_rate.index[0]
        end_time=reservoir_rate.index[-1]
        date_range=cfd.create_date_range(start_time, end_time,'W-SUN')
        reservoir_rate=cfd.check_duplicate_data(reservoir_rate)
        reservoir_rate=cfd.check_missing_data(reservoir_rate, date_range,'W-SUN')    
        reservoir_rate=cfd.check_negative_data(reservoir_rate)   #fill the zero and negative values
        
        return reservoir_generation, reservoir_rate

    def __create_equivalent_model(self,reservoir_generation, reservoir_rate, path_obj, config_obj, code, year):
        #------------------------- retrieve price -----------------------------
        if code not in ['ME']:
            if code in ['HR']:
                price=self.entsoe_obj.request_price(config_obj.map[code]['Entsoe'], '20210101', '20220101', code)
            else:
                price=self.entsoe_obj.request_price(config_obj.map[code]['Entsoe'], '20180101', '20190101', code)  #entsoe.py has issues on the 'extra day'problem
            
            start_time= price.index[0]
            end_time=price.index[-1]
            date_range=cfd.create_date_range(start_time, end_time,'h')
            price=cfd.check_duplicate_data(price)
            price=cfd.check_missing_data(price, date_range)
            price=cfd.check_negative_data(price)
            
            #____________________save the electricity price___________________
            his_price_path = path_obj.path_dict['his_price_path']
            his_price_path.mkdir(parents=True, exist_ok=True)
            price_path = his_price_path / f"{code}_{year}_price.csv"

            if code in ['BG']:
                price=price*0.51     #currency transfer is set to be 0.51
            elif code in ['RO']:
                price=price*0.2
            
            price=price.drop(price.index[-1])
            price.index=pd.to_datetime(price.index, utc=True)
            if len(price.index)==8760:
                price.columns=['EUR/MWh']
                price.to_csv(price_path,sep=';')
                print(f'Save {year} price for {code}--->Finished')
            else: 
                print('The historical production data is wrong')
        else:
            print(f'There is no price data for {code}')

        #_____________________________save the max_min M and P data_________________
        init_params_path = path_obj.path_dict['his_eq_path'] / f"{code}_initial_params.csv"
        
        init_params= pd.DataFrame({
            'area': [code],  
            'max_M(MWh)': [max(reservoir_rate['Reservoir rate']) ],
            'min_M(MWh)': [min(reservoir_rate['Reservoir rate']) ],
            'max_P(MWh)': [max(reservoir_generation['Reservoir generation']) ],
            'min_P(MWh)': [min(reservoir_generation['Reservoir generation']) ]
        })

        init_params.to_csv(init_params_path, index=False, sep=';')
        print(f'Save the max and min historical data for {code}--->Finished')

        #_____________________save the reservoir generation data for equivalent model_____________________________
        production_path= path_obj.path_dict['his_eq_path'] / f"{code}_{year}_historical_production.csv"
        
        historical_production=reservoir_generation[reservoir_generation.index.year==year]
        
        if len(historical_production.index)==8760:
            historical_production.to_csv(production_path,sep=';')
            print(f'Save {year} historical production for {code}--->Finished')
        else: 
            print('The historical production data is wrong')

    def __inflow_calc_save(self, path_obj, reservoir_generation, reservoir_rate, code):
        ######################################Inflow calculation############################################
        #TODO: move the  resample before check missing data.
        reservoir_generation = reservoir_generation.resample('h').mean()
        #reservoir_generation=rpi.resample_data(reservoir_generation , 'Reservoir generation','W-SUN')
        reservoir_generation = reservoir_generation.resample('w-sun').sum()
        reservoir_rate=rpi.resample_data(reservoir_rate, 'Reservoir rate',  'W-SUN')
        
        reservoir_generation, reservoir_rate, inflow_start, inflow_end= rpi.time_align(reservoir_generation, reservoir_rate)
        inflow_weekly=rpi.inflow_calculation(reservoir_generation, reservoir_rate)
        inflow_weekly=cfd.check_negative_data(inflow_weekly)

        #____________________save the historical inflow data _____________________________

        history_data_path = path_obj.path_dict['history_data_path']
        inflow_path = path_obj.path_dict['data_file']
        inflow_weekly.to_csv(inflow_path,sep=',')
        print(f'Save historical inflow for {code}--->Finished')
        
        #____________________plot and save the historical inflow data _____________________________
        fig_path = history_data_path / f'{code}_{inflow_start}_{inflow_end}_inflow.pdf'
        rpi.save_inflow_fig(inflow_weekly, str(fig_path), code)
