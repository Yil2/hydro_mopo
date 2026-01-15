from pathlib import Path
import toml
import sys

class ConfigData:

    def __init__(self):
        #FIX:add read only property
        self.__PATH_USER_CONFIG = Path(__file__).parent.parent / 'config_data' / 'user_config.toml'
        self.__PATH_COUNTRY_MAP = Path(__file__).parent.parent / 'config_data' / 'country_map.toml'
        self.config = {}
        self.map = {}
        self.country_code = ''
        self.hydro_type = ''
        self.__load_config()

    def __load_config(self):
        with open(self.__PATH_USER_CONFIG, 'r') as file:
            self.config = toml.load(file)
        with open(self.__PATH_COUNTRY_MAP, 'r') as file:
            self.map = toml.load(file)

    def args_check(self, args):
        # breakpoint()
        if args.input:
            country_code = args.input
        else:
            country_code = self.config['country_code']
        print(f'Selected Counrty is [{country_code}]')

        # Check country code
        if country_code not in self.map:
            print(f'Input country code: {country_code} is not available!')
            sys.exit(1)

        if args.type:
            hydro_type = args.type
        else:
            hydro_type = self.config['hydro_type']

        #Check for hydro_type and country_code matching
        if hydro_type == 'hdam' and self.map[country_code]['hdam_type_support']:
            print(f'Input country code: {country_code} Select hydro type : {hydro_type}')

        elif hydro_type == 'hror' and self.map[country_code]['hror_type_support']:
            print(f'Input country code: {country_code} Select hydro type : {hydro_type}')

        else:
            print(f'Input country code: [{country_code}] Select hydro type : [{hydro_type}] are not supported!')
            sys.exit(1)

        self.country_code = country_code
        self.hydro_type = hydro_type
        


class FetchPath:

    def __init__(self, config_obj):
        self.path_dict = {}
        self.__spine_gen_dir = []
        self.__set_file_path(config_obj)
        self.__create_dir()
        

    def __set_file_path(self, config_obj):
        country_code = config_obj.country_code
        geo_dir = Path(config_obj.config['geo_dir'])
        if config_obj.hydro_type == 'hdam':
            type="hdam"
        elif config_obj.hydro_type == 'hror':
            type="hror"
        else:
            print('Unknown hydro type')
        self.path_dict['onshore_filepath'] = geo_dir / 'onshore.geojson'
        
        data_dir = Path(config_obj.config['data_dir'])
        history_data_path = data_dir / country_code
        self.path_dict['data_file'] = history_data_path / (country_code + f'_historical_{type}_inflow.csv')
        self.path_dict['history_data_path'] = history_data_path
        self.path_dict["glofas_cdf_path"] = config_obj.config['glofas_cdf_path']
        self.path_dict['osm_filepath'] = history_data_path / (country_code + '_hydropower_plants.geojson')
        self.path_dict['disc_file'] = history_data_path / (country_code +'_' + type + '_glofas_discharge.csv')


        method = config_obj.config['algorithm']
        solution_dir = Path(config_obj.config['solution_dir'])
        self.path_dict['pred_data_path'] = solution_dir / str(method) / type
        self.path_dict['his_eq_path'] = solution_dir / 'Historical_production'
        self.path_dict['his_price_path'] = solution_dir / 'price'
        self.path_dict['figs_path_hdam'] = solution_dir / str(method)  / 'figs' / 'hdam'
        self.path_dict['figs_path_hror'] = solution_dir / str(method)  / 'figs' / 'hror'
        
       
        #Path(config_obj.config['cds_path'])

        # Generated directories by this tool
        self.__spine_gen_dir = ['history_data_path', 'pred_data_path', 'his_eq_path', 'his_price_path', 'figs_path_hdam', 'figs_path_hror']

    def __create_dir(self):
        for dir in self.__spine_gen_dir:
            self.path_dict[dir].mkdir(parents=True, exist_ok=True)


