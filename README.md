# Hydropower Inflow Model for European Countries
This is the tool for modelling the European hydropower related energy data: weekly reservoir inflow and daily run-of-river generation in bidding zone level, by historical river discharge. The tool is used for the EU project [MOPO](https://www.tools-for-energy-system-modelling.org/). 
## How to use

### Required API Access :cloud:
In order to run this tool, below API access are required and properly setup:

+ :zap: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)  
Please contact  ENTSO-E Transparency Platform team to get the API key.
+ :ocean: [Global Flood Awareness System](https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical?tab=overview)  
Please follow the [CDSAPI setup](https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+Windows) to setup the CDS Api key.

### Installation



### User Configuration

+ Input area code, hydropower type, years, scenario and method, eg:
    + hydro_type = "hdam"
    + country_code_list  = ["SE1", "SE2"]  
    + pred_years = [2000,2005,2008]
    + scenario = "example"   
    \:point_right:The scenario name is used for naming the model output. 
    + algorithm  = "random forest"    
    \:point_right:The modelling method used for generating required energy data by weather data. 

+ Input paths for saving raw weather data, processed historical data and model output :file_folder:
    + geo_dir = 'Your path for the downloaded onshore.geojson'
    + data_dir= 'Your path saving the historical data'
    + solution_dir = 'Your path saving the modelled results'
    + glofas_cdf_path= 'Your path saving the raw weather data'

+ Input API token
    + entsoe_api_token='Your entsoe api token'

## Data Attribution

This work is built based on below open data. 
+ Hydropower historical data (2015/2019-2024) are from [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)  via the [entsoe-py library](https://github.com/EnergieID/entsoe-py), and [eSett Open Data](https://opendata.esett.com/). Water Reservoirs Stored Energy Value, Electricity price, Reservoir-based hydropower generation, Run-of-river hydropower generation are retrieved from Entso-e for the following areas: [AT, BG, CH, ES, DE-LU, FR, FI, LV, HR, GR, ITCN, ITSA, ITN1, ITSU, NO1, NO2, NO3, NO4, NO5, PT, RO, RS, SE1, SE2, SE3, SE4, SI]. Reservoir-based hydropower generation data for [SE1, SE2, SE3, SE4] are retrieved from eSett Open Data Platform.
+ Detailed hydropower location data are from [OpenStreetMap](https://www.openstreetmap.org)
+ Historical river discharge data (1980-2024) are from [Global Flood Awareness System](https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical?tab=overview) 


## Cite Me :paperclip:
 
