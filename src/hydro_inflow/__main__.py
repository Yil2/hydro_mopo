#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hydro_inflow.user_argparse import UserArgparser
from hydro_inflow.config_data_handle import ConfigData, FetchPath
from hydro_inflow.historical_data_process import FetchInflow
from hydro_inflow.database_osm_api import OverpassAPI
from hydro_inflow.database_glofas_api import DatabaseGlofasAPI
from hydro_inflow.model_train import ModelTrain

def countries_process(args, config_obj):
    country_code_list = config_obj.config['country_code_list']
    for index in range(len(country_code_list)):
        selected_area_process(args, config_obj, index)


def selected_area_process(args, config_obj, index):
        config_obj.args_check(index)
        path_obj = FetchPath(config_obj)
        FetchInflow(config_obj, path_obj, args)
        #TODO: if retrieve only historical data, then skip below
        osm_database_obj = OverpassAPI(config_obj, path_obj)
        osm_database_obj.save_osm_data_main()
        glofas_database_obj = DatabaseGlofasAPI(config_obj, path_obj)
        glofas_database_obj.save_disc_main()
        model_train_obj = ModelTrain(config_obj, path_obj)
        model_train_obj.modelled_data_main()

def main():
    arg_obj = UserArgparser()
    arg_obj.parser_run() 
    config_obj = ConfigData()
    countries_process(arg_obj.args, config_obj)


if __name__ == "__main__":
    main()
    print("----------------Finished---------------")
