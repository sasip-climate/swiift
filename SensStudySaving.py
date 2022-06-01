# -*- coding: utf-8 -*-
"""
Created on Wed May 18 18:16:02 2022

@author: tlilia
"""
import config
from tqdm import tqdm

from MF1D_func_saving import MF1D_serie

import dataSaving
from dataSaving import listForPandas

from pars import E, v, rho_w, g, K
from ElasticMaterials import FracToughness

########################
# PARAMETERS TO MODIFY #
########################

DispType = 'Open'
EType = 'Flex'

# Fracture computation
multiFrac = 1  # TODO: Allow multiFrac > 1
assert multiFrac == 1

# Wave Parameters
n_0 = [0.2, 0.3, 0.4, 0.5]
wvlength = [19, 18, 20, 22, 25]

# Ice parameters
h = [0.2, 0.5, 0.7, 1.]
x0 = 10
L = [500]
G = FracToughness(E, v, K)
# TODO: For now, E, nu are taken in pars and G computed -> modify modules

# Type of simulation
growing = True  # Other value would raise error for now #FIXME
reset = False   # Other value would raise error

########################

# Defines the parameters to test on
parameters = {'h': h, 'wvLength': wvlength, 'wvAmpl': n_0, 'initialLength': L,
              'E': [E], 'nu': [v], 'G': [G], 'multiFrac': [multiFrac], 'EnergyType': [EType],
              'DispType': [DispType]}

# Initializes the dataFrame and looks for non computed values
# Note: If you want to recover a data frame from partial progress:
# - go in the config.DataTempDir
# - copy the latest temporary data frame to the config.DataFileName location/filename
df = dataSaving.createAndLoadDataFrame(config.DataFileName, parameters, results=['xc', 'FloeSizes'])
zerosDf = df[~df['computed']]

# Gets the size of the DataFrame
totalSize = df.shape[0]
initialPos = totalSize - zerosDf.shape[0]
addedData = 0

# Dictionary to save results in spyder
results = {}

for index, row in tqdm(zerosDf.iterrows(), total=totalSize, initial=initialPos, desc='Pars Loop'):
    # computes characteristic length
    I = row['h']**3 / (12 * (1 - row['nu']**2))
    xc = (row['E'] * I / (rho_w * g))**0.25

    # runs simulation
    FloeSizes = MF1D_serie(row)

    # saves results in dataFrame
    df.loc[index, ['xc', 'FloeSizes', 'computed']] = [xc, listForPandas(FloeSizes), True]

    # saves results in Spyder variables in case of error
    key = (row['h'], row['wvLength'], row['wvAmpl'])
    results[key] = (xc, FloeSizes)

    # regularly save data in case of error
    if addedData % 10 == 9:
        PartialDFName = config.DataTempDir + 'dataTemp_' + str((initialPos + addedData) // 10) + '.pkl'
        dataSaving.saveDataFrame(df, PartialDFName)

    addedData += 1

dataSaving.saveDataFrame(df, config.DataFileName)
