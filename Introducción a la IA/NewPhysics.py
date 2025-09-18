import pandas as pd
import numpy as np
import h5py
import os 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import joblib
import time
import psutil

path = "C:\\Users\\lucki\\Desktop\\AI_Newphysics\\mc_110901.ZPrime500\\mc_110901.ZPrime500.hdf5"

# Open the file as readonly'
f_data = h5py.File(path , 'r')

# List all groups
data = pd.DataFrame(f_data['mini'][:])

background_noise = [
    "mc_110140.stop_wtchan.hdf5",
    "mc_105986.ZZ.hdf5",
    "mc_105987.WZ.hdf5",
    "mc_110090.stop_tchan_top.hdf5",
    "mc_110091.stop_tchan_antitop.hdf5",
    "mc_110119.stop_schan.hdf5",
    "mc_173046.DYtautauM15to40.hdf5",
    "mc_117049.ttbar_had.hdf5",
    "mc_117050.ttbar_lep.hdf5",
    "mc_147770.Zee.hdf5",
    "mc_147771.Zmumu.hdf5",
    "mc_147772.Ztautau.hdf5",
    "mc_167740.WenuWithB.hdf5",
    "mc_167743.WmunuWithB.hdf5",
    "mc_167746.WtaunuWithB.hdf5",
    "mc_173041.DYeeM08to15.hdf5",
    "mc_173042.DYeeM15to40.hdf5",
    "mc_173043.DYmumuM08to15.hdf5",
    "mc_173044.DYmumuM15to40.hdf5",
    "mc_173045.DYtautauM08to15.hdf5"
    ]

signal_files = [
    "mc_110909.ZPrime2500.hdf5",
    "mc_110902.ZPrime750.hdf5",
    "mc_110903.ZPrime1000.hdf5",
    "mc_110905.ZPrime1500.hdf5",
    "mc_110906.ZPrime1750.hdf5",
    "mc_110907.ZPrime2000.hdf5"
    ]

print(data)

# Extract all features excluding the ones specified -> note: these are manully chosen initially because they are irrelevant for training our model.
# However, we still have to do a more refined feature validation as you will see further down
plot_features = [col for col in data.columns if col not in [
    'eventWeight', 
    'mcWeight', 
    'channelNumber', 
    'runNumber', 
    'data_type', 
    'label', 
    'eventNumber', 
    'jet_6_SV0', 
    'jet_7_SV0', 
    'jet_8_SV0', 
    'jet_9_SV0', 
    'jet_8_trueflav', 
    'jet_9_trueflav',
    'lep_4_E',                       
    'lep_4_charge',            
    'lep_4_eta',                     
    'lep_4_etcone20',                
    'lep_4_flag',                   
    'lep_4_phi',                     
    'lep_4_pt',                      
    'lep_4_ptcone30',                
    'lep_4_trackd0pvunbiased',       
    'lep_4_tracksigd0pvunbiased',    
    'lep_4_type',                   
    'lep_4_z0',                     
    'lep_5_E',                       
    'lep_5_charge',                  
    'lep_5_eta',                     
    'lep_5_etcone20',             
    'lep_5_flag',                    
    'lep_5_phi',                     
    'lep_5_pt',                      
    'lep_5_ptcone30',             
    'lep_5_trackd0pvunbiased',      
    'lep_5_tracksigd0pvunbiased',   
    'lep_5_type',                    
    'lep_5_z0',                                            
    'lep_trigMatched',      
    'jet_1_trueflav',                
    'jet_1_truthMatched',            
    'jet_2_trueflav',                
    'jet_2_truthMatched',            
    'jet_3_trueflav',                
    'jet_3_truthMatched',            
    'jet_4_trueflav',                
    'jet_4_truthMatched',            
    'jet_5_trueflav',                
    'jet_5_truthMatched',            
    'jet_6_trueflav',                
    'jet_6_truthMatched',            
    'jet_7_trueflav',                
    'jet_7_truthMatched',            
    'jet_8_trueflav',                
    'jet_8_truthMatched',            
    'jet_9_trueflav',                
    'jet_9_truthMatched',            
    'runNumber',                     
    'scaleFactor_BTAG',              
    'scaleFactor_ELE',               
    'scaleFactor_JVFSF',             
    'scaleFactor_MUON',              
    'scaleFactor_PILEUP',            
    'scaleFactor_TRIGGER',           
    'scaleFactor_ZVERTEX']]

# Create separate figures for each group of features
for i in range(0, len(plot_features), 3):
    # Determine the number of subplots needed
    n_subplots = min(6, len(plot_features) - i)
    fig, axs = plt.subplots(1, n_subplots, figsize=(28, 2))
    # If there's only one subplot, axs will not be an array, so we wrap it in one
    if n_subplots == 1:
        axs = [axs]
    for j in range(n_subplots):
        feature_name = plot_features[i+j]
        axs[j].hist(data[feature_name], bins=50, color='blue', alpha=0.7)
        axs[j].set_xlabel('Value')
        axs[j].set_ylabel('Frequency')
        axs[j].set_title(feature_name)
    plt.show()


        