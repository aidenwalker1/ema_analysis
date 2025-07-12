# ema_analysis (Python 3.11)


Compute all emas: reads emas (questions about how someone is feeling) into csv file, also analyzes response rate

-Reads EMA response files from various studies in a given folder.

-Cleans up data and transforms data, removes empty entries

-Saves cleaned up data to ./ema_new.csv

-Also includes functionality to verify the responses, and graph statistical features of the response rates


To run:

-Specify EMA folder locations, run py file.

-Use outputted file for later analysis


Compute all nbacks: reads nbacks (cognitive assessment results) into csv file

-Reads nback response files from various studies in a given folder.

-Cleans up data and transforms data, removes empty entries

-Saves cleaned up data to ./nback.csv


To run:

-Specify nback folder locations, run py file.

-Use outputted file for later analysis


Read and graph data: reads ema and nback data, does analysis and graphs/prints 

To run:

-First run the ema/nback computation files to get the csv files needed

-Run py file

Stat features:

-Provides statistically analysis tools for time-series analysis

-Used as a library


Read ema old: depicrated file
