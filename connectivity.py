# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:30:34 2022
@author: Khadijeh
khadijeh.raeisi@unich.it
"""

import numpy as np
import pandas as pd
import mne
from mne_connectivity import spectral_connectivity_epochs
from progress.bar import Bar
from scipy.stats import spearmanr
from functions import infans_compute_graph_metrics, infans_compute_graph_metrics_per_epoch, graph_preparation_for_regression
import pickle as pkl
from mne_features import feature_extraction


def initialize():
    """
    This function initializes the variables needed for the analysis.
    """
    annotation_file = pd.read_pickle('labels_raw_rec_name.pkl') # contains sleep annotation_file and recordings' name
    recording_list = list(annotation_file.keys()) # contains recordings' name
    indice = np.load(rf"local_adress_here.npy",allow_pickle=True)
    freq_bands = {
        "delta1"    : [0.5, 2],
        "delta2"    : [2, 4],
        "theta"     : [4, 8],
        "alpha"     : [8, 16],
        "beta"      : [16, 32],
        "broadband" : [0.5, 32]
    }
    f_low, f_high = zip(*freq_bands.values())
    no_channels = 8
    fs = 100
    segment_length = 30 #sec
    # Availble metrics for connectivity analysis: ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'wpli', 'wpli2_debiased']
    methods = ['imcoh', 'plv'] 
    mne.set_log_level(verbose = 'CRITICAL', add_frames = 0)
    return annotation_file, recording_list, indice, freq_bands, f_low, f_high, no_channels, fs, segment_length, methods


def calculate_connectivity_matrices():
    """
    This function calculates the connectivity matrices and graph measures.
    """
    annotation_file, recording_list, indice, freq_bands, f_low, f_high, no_channels, fs, segment_length , methods = initialize()
    for i, rec in enumerate(recording_list):
        raw_file = rf'local_adress_here.fif' 
        print(f'\n{16 * "="} {rec} ({i+1}/116) {16 * "="}') 
        # Read the data
        data = mne.io.read_raw_fif(raw_file, verbose = False, preload = True)
        info = data.info
        ann = annotation_file[rec]
        counter = 0     #  the numebr of artifact-free-30s EEG segments in QS_TA
        indices = indice[i] 
        # Allocate memory to connectivity matrices
        num_wins  = len(indices) # number of 30s windows
        con_coh = np.zeros([no_channels, no_channels, len(freq_bands), num_wins]) # 8 * 8 * 6 * num_wins (for coh)
        con_plv = np.zeros([no_channels, no_channels, len(freq_bands), num_wins]) # 8 * 8 * 6 * num_wins (for plv)
        # Allocate memory to univariate features
        feat_mat = np.zeros([len(recording_list),num_wins,152]) #0:56 Complexity feats- 56:136 power feats         
        # 30s Segmentation
        EEG = data.get_data()   
        # indices of clean 30s EEG segments with QS_TA annotation   
        bar = Bar('Connectivity and Feature extraction phase ... ', max = len(indices))

    for n in indices:
        
        win_30s     = EEG[:, n * (fs*segment_length) : (fs*segment_length)] # 30*100 hz
        win_30s_raw = mne.io.RawArray(win_30s, info, first_samp = 0, copy = 'auto', verbose=None) #convert to mne raw 
        x = np.expand_dims(win_30s, 0)
        # Segment 30s window into 3s windows
        # The aim of this part is to have a better estimation of sprctral density and connectivity measures
        epochs_3s = mne.make_fixed_length_epochs(win_30s_raw, duration = 3.0, preload = False);
        epochs_3s.load_data()
        # Spectrum estimation mode: fourier
        #['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'wpli', 'wpli2_debiased']
        con = spectral_connectivity_epochs(epochs_3s, method = methods,
                                       mode = 'fourier', sfreq = 100, fmin = f_low,
                                       fmax = f_high, faverage = True, n_jobs = 1)
        
        con_coh[:,:,:,counter] = con[0].get_data(output = "dense")
        con_plv[:,:,:,counter] = con[1].get_data(output = "dense")           
        counter +=  1       
        bar.next()          
        bar.finish()

    # discard extra connecivity matrices (containing only zeros)                
    con_coh = con_coh[:,:,:,0:counter]
    con_plv = con_plv[:,:,:,0:counter] 
    print(f"noiseless epochs: {counter}/{len(ann)}",)   
    # compute graph-theory-based metrics
    print('Calculation of the graph metrics for the first connectivity method ... ')
    graph_metrics_0 = infans_compute_graph_metrics_per_epoch(con_coh, freq_bands)
    graph_metrics_0_reg = graph_preparation_for_regression(graph_metrics_0)
    print(f'{20 * "="}')   
    print('Calculation of the graph metrics for the second connectivity method ... ')
    graph_metrics_1 = infans_compute_graph_metrics_per_epoch(con_plv, freq_bands)
    graph_metrics_1_reg = graph_preparation_for_regression(graph_metrics_0)
    print(f'{20 * "="}')

    con_mat_0n = {
              "delta1"    : con_coh[:,:,0,:],
              "delta1_std"    : np.std(con_coh[:,:,0,:],2),
              "delta2"    : con_coh[:,:,1,:],
              "delta2_std"    : np.std(con_coh[:,:,1,:],2),
              "theta"     : con_coh[:,:,2,:],
              "theta_std"     : np.std(con_coh[:,:,2,:],2),
              "alpha"     : con_coh[:,:,3,:],
              "alpha_std"     : np.std(con_coh[:,:,3,:],2), 
              "beta"      : con_coh[:,:,4,:],
              "beta_std"      : np.std(con_coh[:,:,4,:],2),
              "broadband" : con_coh[:,:,5,:],
              "broadband_std" : np.std(con_coh[:,:,5,:],2)
              }

    con_mat_1n = {    "delta1"    : con_plv[:,:,0,:],
                  "delta1_std"    : np.std(con_plv[:,:,0,:],2),
                  "delta2"    : con_plv[:,:,1,:],
                  "delta2_std"    : np.std(con_plv[:,:,1,:],2),
                  "theta"     : con_plv[:,:,2,:],
                  "theta_std"     : np.std(con_plv[:,:,2,:],2),
                  "alpha"     : con_plv[:,:,3,:],
                  "alpha_std"     : np.std(con_plv[:,:,3,:],2), 
                  "beta"      : con_plv[:,:,4,:],
                  "beta_std"      : np.std(con_plv[:,:,4,:],2),
                  "broadband" : con_plv[:,:,5,:],
                  "broadband_std" : np.std(con_plv[:,:,5,:],2)}
   
    result_connectivity = {methods[0]:con_mat_0n, methods[1]:con_mat_1n, "features":feat_mat} 
    result_graph        = {methods[0] : graph_metrics_0, methods[1] : graph_metrics_1}  
    result_graph_reg    = {methods[0] : graph_metrics_0_reg, methods[1] : graph_metrics_1_reg}            
    connectivity_all.append(result_connectivity)
    graph_metrics_all.append(result_graph)
    graph_metrics_all_reg.append(result_graph_reg)

    return connectivity_all,  graph_metrics_all
def save_results(connectivity_all, graph_metrics_all):
    """
    This function saves the results to a pickle file.
    """
    with open('connectivity_allepochs_allfeatsplus_60min_imcoh_plv_AS.pickle', 'wb') as f:
        pkl.dump(connectivity_all, f)
    with open('graph_metrics_imcoh_plv_AS.pickle', 'wb') as f:
        pkl.dump(graph_metrics_all, f)

if __name__ == "__main__":
    connectivity_all, graph_metrics_all = calculate_connectivity_matrices()
    save_results(connectivity_all, graph_metrics_all)