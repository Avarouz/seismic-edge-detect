import h5py
import numpy as np

'''
Takes in a filepath to h5 file
Returns 2d array of data and time between samples (s)
'''

def extract_data(filepath):

    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]  #(1150, 12000)
        dt = f['data'].attrs['dt_s']

    return data, dt