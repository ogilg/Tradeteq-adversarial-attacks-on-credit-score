#=======================================================================================#
# Author       : Michael Boguslavsky                                                    #
# Company      : Tradeteq                                                               #
# Script Name  : postcodemap.py                                                         #
# Description  : Map postcodes to socio-economic classes using ONS                      #
# Version      : 1.1                                                                    #
#=======================================================================================#
#=======================================================================================#
# Change history:                                                                       #
#                                                                                       #
# 1.0  19/06/17  Michael Boguslavsky - Created                                          #
# 1.1  24/09/19  Agnieszka Rees      - Extended to process 2019 & 2020 data             #
#                                                                                       #
#=======================================================================================#

# Map postcodes to socio-economic classes using Office of National Statistics (ONS)

import pandas as pd
import tqdm
import numpy as np
import sys
import datetime as dt
from time import time
import subprocess
from subprocess import check_output, call
import os

# Include Machine Learning utils
from myutils import *

#0. setup configuration and logging
_modulename = "postcodemap"
_common_cfg = "postcodemap"

# Process command line arguments
if len(sys.argv)>1:
    cfgname = sys.argv[1]
else:
    cfgname="DEFAULT"

cfg = getconfig(_common_cfg, cfgname)
log = setuplog(_modulename, cfg)

# DB config
TTQSQL = getconfig("MSSQL")

LogPath = getLogPath(_modulename)
CfgPath = getCfgPath(_common_cfg)

time_now = dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

log.info("-------------------------------------------------------------------------------------")
log.info("                                S T A R T                   " + time_now )
log.info("-------------------------------------------------------------------------------------")
log.info("Command Line:                  [ " + '  '.join(sys.argv) + " ]")
log.info("Config file:                   " + CfgPath)
log.info("Config used:                   " + cfgname)
log.info("Log file:                      " + LogPath)
log.info("File logging level:            " + cfg["filelogging"])
log.info("Console logging level:         " + cfg["consolelogging"])

log.info("Data input folder:             " + cfg["datafolder"])
log.info("Data output folder:            " + cfg["outpath"])
log.info("nspl_file_in:                  " + cfg["nspl_file_in"])
log.info("nspl_file_out:                 " + cfg["nspl_file_out"])
log.info("save_to_file:                  " + cfg["save_to_file"])
log.info("nspl_bcp_file:                 " + cfg["nspl_bcp_file"])
log.info("-------------------------------------------------------------------------------------")

datafolder = cfg['datafolder']
outpath = cfg['outpath']
NSPL_fn = cfg['nspl_file_in']

# --------------------------------------------------
# Load postcode data
# --------------------------------------------------
nspl_file_in = datafolder + NSPL_fn
print("Loading postcode data from \"{:}\" ...".format(nspl_file_in))
nspl = pd.read_csv(nspl_file_in,
        usecols = ["pcd", "oseast1m", "osnrth1m", "cty", "oa11",
                   "lat", "long", "ru11ind", "oac11",
                   "imd"])

# Remove rows with NaN oac11
nspl = nspl.loc[nspl.oac11.map(lambda x: type(x) == str )].reset_index(drop=True)

# Need to remove space from postcodes to match NSPL to company address postcodes
nspl.pcd = nspl.pcd.str.replace(" ","")

# Preprocess some columns
# Map cty to country code
def cty_map(cty):
    if type(cty) == str:
        country = cty[0]
        if country=="L" or country=="M": #map Channel Islands to England
            country='E'
    else:
        country='E'
    return country


nspl["country"] = nspl.cty.map(cty_map)

## oac11: socio economic class of OA
# split into components
# oac11 have format like 1b2
# first digit is meaningful;
# combination of first digit and letter is meaningful
# full code is meaningful
# letter only, second digit only are not
# thus need to transform to three columns and then binarize each
nspl["oac1"] = nspl.oac11.map(lambda x: x[0])
nspl["oac2"] = nspl.oac11.map(lambda x: x[0:2])


##imd is not comparable across England, Wales, Scotland and NI
#universal umd
#Using regressions of a universal IMD on country imd from UK_indices_of_multiple_deprivation-a_way_to_make_c.pdf
#eyballing from charts:
#England - NI- Wales - Scotland
rO =  np.array([[138., 634], [100., 651.], [110., 651.], [0.,0.]])
rI =  np.array([[706., 41.], [682., 58.], [678., 56.], [1., 1.]])
rA =  np.array([[145., 634.], [111., 606.], [121. ,618. ], [0.,0.]])
rB =  np.array([[583., 191.], [546., 100.], [ 505., 272.], [1., 1.]])
#regression coefficients by country are:
regr = np.zeros([4,2])

for idx, country in enumerate(rO):
    A = (rA[idx] - rO[idx]) / (rI[idx] - rO[idx])
    B = (rB[idx] - rO[idx]) / (rI[idx] - rO[idx])
    regr[idx] = [A[1], (B[1]-A[1])/(B[0]-A[0])]

ctylist = "ENWS"
nspl["imdu"] = np.nan
for i in tqdm.tqdm(range(0,nspl.shape[0])):
    ctyidx = ctylist.index(nspl.loc[i,"country"])
    nspl.set_value(i, "imdu",  regr[ctyidx,0] + regr[ctyidx,1] * nspl.loc[i, "imd"])

# --------------------------------------------------
# Save processed data
# --------------------------------------------------
save_to_file = int(cfg["save_to_file"])

if save_to_file:
    log.info("")
    nspl_file_out = outpath + cfg['nspl_file_out']
    print("Saving data to pickle file \"{:}\" ...".format(nspl_file_out))
    nspl.to_pickle(nspl_file_out)
else:
    log.info("")
    print("Skipping saving data to pickle file \"{:}\"".format(nspl_file_out))
