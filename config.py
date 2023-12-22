#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue June  7 2022

@author: auclaije
"""

import sys
import os

FlexFrac1D = '/home/mokusn/Documents/FlexFrac1D/FlexFrac1D/'
LibPath = 'Libraries/'
sys.path.append(FlexFrac1D)
sys.path.append(FlexFrac1D + LibPath)
sys.path.append(os.getcwd())

FigsDirFloes = 'Figs/Floes/'
FigsDirSumry = 'Figs/Summary/'
FigsDirSpec = 'Figs/Spec/'

DataFileName = 'database/dataSensStudy.pkl'
DataTempDir = 'database/temp/'

# if not os.path.isdir(FigsDirFloes):
#     raise ValueError('Floes folder does not exist')

# if not os.path.isdir(FigsDirSumry):
#     raise ValueError('Summary folder does not exist')

# if not os.path.isdir(DataTempDir):
#     raise ValueError('Temp folder does not exist')
