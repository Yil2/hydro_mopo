#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hydro_inflow.user_argparse import UserArgparser
from hydro_inflow.config_data_handle import ConfigData, FetchPath
from hydro_inflow.historical_data_process import FetchInflow
from hydro_inflow.database_osm_api import OverpassAPI
from hydro_inflow.database_glofas_api import DatabaseGlofasAPI
from hydro_inflow.model_train import ModelTrain

def main():
    
    arg_obj = UserArgparser()
    arg_obj.parser_run()
    config_obj = ConfigData()
    config_obj.args_check(arg_obj.args)
    path_obj = FetchPath(config_obj)
    #inflow_obj = FetchInflow(config_obj, path_obj, arg_obj.args)
    #TODO: if retrieve only historical data, then skip below

    osm_database_obj = OverpassAPI(config_obj, path_obj)
    osm_database_obj.save_osm_data_main()
    glofas_database_obj = DatabaseGlofasAPI(config_obj, path_obj)
    glofas_database_obj.save_disc_main()
    model_train_obj = ModelTrain(config_obj, path_obj)
    model_train_obj.modelled_data_main()


if __name__ == "__main__":
    main()
    print("----------------Finished---------------")
