'''
Student Name: Nur Muhammad Bin Khameed
SRN: 160269044
CO3311 Neural Networks CW1

Main Class

Instructions:
Please install these libraries to run this file:
os, sys, xlrd, openpyxl, numpy, pandas, matplotlib
'''

import kohonen as k
import pandas as pd

# Testing with AND Gate
# n = k.Network(2, 0.1, 0.01, 'AndGateData.xlsx', 'AndGateClasses.xlsx', 1)
# n.execute_network()

# Test with 2 classes
# n = k.Network(2, 0.01, 0.0001, 'TableData.xlsx')
# n.execute_network()

# # # Test with 3 classes
# n = k.Network(3, 0.01, 0.0001, 'TableData.xlsx')
# n.execute_network()

# Test with 4 classes
n = k.Network(2, 0.01, 0.0001, 'TableData.xlsx')
n.execute_network()
