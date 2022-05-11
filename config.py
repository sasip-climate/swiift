#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue June  7 2022

@author: auclaije
"""
import os

FigsDirFloes = 'Figs/Floes/'
FigsDirSumry = 'Figs/Summary/'

if not os.path.isdir(FigsDirFloes):
    raise ValueError('Floes folder does not exist')

if not os.path.isdir(FigsDirSumry):
    raise ValueError('Summary folder does not exist')
