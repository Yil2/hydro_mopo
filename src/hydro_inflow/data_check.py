# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:14:29 2024

@author: yil2
"""

import pandas as pd

class CheckFillData:
    
    def create_date_range(start_date:str, end_date:str, freq='h') ->pd.DataFrame:
        """

        Parameters
        ----------
        start_date : string
            start time of a daterange
        end_date : string
            end time of a daterange
        freq: frequency of this daterange

        Returns
        -------
        a date range with sepecifc frequency

        """
        return pd.date_range(start_date, end_date, freq=freq)
    
    def check_missing_data(input_data:pd.DataFrame, date_range:pd.DatetimeIndex, freq='h')->pd.DataFrame:
        """
        

        Parameters
        ----------
        input_data : pd.DataFrame
            The input_data for data checking
        date_range : pd.DatetimeIndex
            The referenced date range

        Returns
        -------
        The expcted data with correct date range

        """
        if input_data.index[-1]!=date_range[-1]:
            input_data.loc[date_range[-1]]=input_data.iloc[-1]
            missed_length=len(date_range)-len(input_data.index)+1
            
        elif input_data.index[0]!=date_range[0]:
            input_data.loc[date_range[0]]=input_data.iloc[0]
            missed_length=len(date_range)-len(input_data.index)+1
                
        elif len(input_data.index)!=len(date_range):
            missed_length=len(date_range)-len(input_data.index)
            input_data=input_data.resample(freq).asfreq()
            input_data=input_data.interpolate(method='linear')
            #print(f"{missed_length} data points are missing")
        else:
            pass
        
            return pd.DataFrame(input_data)
        
    def check_duplicate_data(input_data:pd.DataFrame)->pd.DataFrame:
        
        """
        Remove duplicate entries from the input DataFrame based on the index.

        Parameters
        ----------
        input_data : pd.DataFrame
            The DataFrame to check for duplicates.

        Returns
        -------
        pd.DataFrame
            A DataFrame with duplicate index entries removed, keeping the first occurrence.
        """
        output_data=input_data[~input_data.index.duplicated(keep='first')]

        return pd.DataFrame(output_data)
    
    def check_negative_data(data:pd.DataFrame)->pd.DataFrame:
        """
        Fill the zero values in the data by the mean of the previous and next values.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.DataFrame
            The data with zero values replaced by the mean of the previous and next values.
        """
        if (data<0).any().any():
            data=data.mask(data<=0)
            pd.set_option('future.no_silent_downcasting', True)
            data_ffilled = data.ffill().infer_objects(copy=False)
            data_bfilled = data.bfill().infer_objects(copy=False)
            mean_filled_value=(data_bfilled+data_ffilled)/2
            data.fillna(mean_filled_value,inplace=True)

        return pd.DataFrame(data)
       
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
