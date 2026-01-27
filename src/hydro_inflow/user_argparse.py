import argparse
import os, sys
from pathlib import Path

class UserArgparser:

    def __init__(self):
        self.description  = ''' This is the Spine tool for EU hydo flow prediction.
                            Run with --help to check all available arguments explanation 
                            User Guidance: 1. Run with [ --config | -c ] to configure all necessary parameters and database paths in your local machine. 
                                        2. Run with [ --run | -r ] to run the tool based on the configuration.
                            '''
        self.PATH_USER_CONFIG = Path(__file__).parent.parent / 'config_data' / 'user_config.toml'

    def __arg_parse(self):
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument('-c', '--config', action='store_true', help='Open configuration toml file by user default editor')
        parser.add_argument('-r', '--run', action='store_true', help='Run the tool with toml configuration')
        #parser.add_argument('-i', '--input', type=str, metavar ='', help='Select bidding country code to run, supperssing the code in the configuration file')
        #parser.add_argument('-t', '--type', choices=['hdam', 'hror'], metavar ='', help='Select hydro type to run, supperssing the type in the configuration file')
        #TODO: add years,  scenario, and if historcial only
        #parser.add_argument('--equivalent_model', default=False, action='store_true', help='Generate an addtional equivalent model with selected country')
        self.args = parser.parse_args()
    
    def __arg_handle(self, args):
        if args.config:
            os.startfile(self.PATH_USER_CONFIG)
            sys.exit(0)
        if not args.run:
            sys.exit(0)

    def parser_run(self):
        self.__arg_parse()
        self.__arg_handle(self.args)
