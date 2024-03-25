# deep-mort
A repo for a deep learning mortality modeling project

This project will build off of the deep learning model proposed by Richman and Wuthrich (2019) for mortality forecasting. 

# Data 
Outcome data for age-specific mortality come from the United States Mortality Database (USMDB). Covariate data on behavioral risk factors for mortality come from the Behavioral Risk Factor Survaillance System (BRFSS). USMDB data was relatively straightforward to download. I downloaded 1x1 life tables for males and females for each of the US states from https://usa.mortality.org/ and compiled the data into a .csv file in code/create-usmdb-file.R in this repo. BRFSS was more complicated to access and required several steps, including downloading the CSV versions of the orignal ASCII files from the files hosted on Winston Larsen's AWS account, accessible here: https://github.com/winstonlarson/brfss. I converted the remaining files (years 2015-2022) from .XPT files downloaded from https://www.cdc.gov/brfss/annual_data/annual_data.htm using code/sas2csv.R in this repo. 

