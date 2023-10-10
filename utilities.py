"""
Created on Tue Jan 19 13:36:34 2022
@author: Khadijeh
khadijeh.raeisi@unich.it
"""
import numpy as np
import bct
from tqdm import tqdm

def infans_compute_graph_metrics(input_conn, freq_bands):
    """
    This functions gets the connectivity matrices for each recording which are calculated 
    on 30s segments. Then it calculates the graph-theory-based measures for each graph.
    Finally it outputs the mean of each measure over all segments
    
    INPUTS: 
        input_conn: a 4-D numpay array containing connectivity matrices for each segment at each frequency band
        (8 * 8 * 6 * number of 30s segemnts)
        freq_bands: a dictionary containing the name of frequency bands and their ranges.
        
    OUTPUTS:
        results: a dictionary containing the measures for all frequency band
    
    NOTE:
        The following measures are calculated for each graph:
            strength: is the sum of weights of links connected to the node.
            
            betweenness centrality: is the fraction of all shortest paths in the network that contain a given node.
                                    Nodes with high values of betweenness centrality participate in a large number
                                    of shortest paths.
            
            path length: is the average shortest path length in the network.
            
            global efficiency: is the average inverse shortest path length in the network.
                
            radius: the minimum among all the maximum distances between a node to all other nodes is considered as the radius
                
            diameter: is the largest number of vertices which must be traversed in order to travel from one vertex to another
                      when paths which backtrack, detour, or loop are excluded from consideration
                
            local efficiency: is the global efficiency computed on the neighborhood of the node, and is related to the 
            clustering coefficient.
                
            clustering coeffcient: is the fraction of triangles around a node (equiv. the fraction of nodes neighbors 
                                                                               that are neighbors of each other)
                
    """
  
    no_conn_mats = input_conn.shape[-1] # number of 30s segments / connectivity matrices for each recording
    results      = {} # a dictionary containing the measures for all frequency band 
    
    for i , freq in enumerate(freq_bands.keys()):
        
        freqband_result   = {} # a dictionary containing the measures for each frequency band             
        strength          = [] # a list containing 1-D arrays for nodes' strengths
        betweenness       = [] # a list containing 1-D arrays for nodes' betweenness centralities
        local_efficiency  = [] # a list containing 1-D arrays for nodes' loacal effieciencies
        clustering_coeff  = [] # a list containing 1-D arrays for nodes' clustering coefficients
        path_length       = [] # a list containing the shortest path length of each graph
        global_efficiency = [] # a list containing the global efficiency of each graph
        radius            = [] # a list containing the radius of each graph
        diameter          = [] # a list containing the diameter of each graph
    
        for n in tqdm(range(no_conn_mats)):
            print(n/no_conn_mats)
            
            # select connectivity matrix
            temp_conn = input_conn[:,:,i,n]
            
            # make symmetric
            temp_conn = temp_conn + np.transpose(temp_conn)
                   
            # calculate the strength for each node            
            temp_strength = bct.strengths_und(temp_conn) # 1-D numpy array
            strength.append(temp_strength)
            
            # calculate the betweenness centrality
            L       = bct.weight_conversion(temp_conn, "lengths") # convert weights to lengths
            temp_bc = bct.betweenness_wei(L) # 1-D numpy array
            betweenness.append(temp_bc)
            
            # calculate the path length, global efficiency, radius, and diameter           
            D       = bct.distance_wei(L) # calculate the distance between nodes
            CPL     = bct.charpath(D[0], include_diagonal = False) # the distance matrix must be fed
            temp_pl = CPL[0] # a float number
            temp_ge = CPL[1] # a float number
            temp_ra = CPL[3] # a float number
            temp_di = CPL[4] # a float number
            path_length.append(temp_pl)
            global_efficiency.append(temp_ge)
            radius.append(temp_ra)
            diameter.append(temp_di)
            
            # calculate the local efficiency
            # Note: all weights must be between 0 and 1
            temp_le = bct.efficiency_wei(temp_conn, local = True) # 1-D numpy array
            local_efficiency.append(temp_le)
            
            # calculate the clustering coefficient for each node
            temp_cluscoeff = bct.clustering_coef_wu(temp_conn) # 1-D numpy array
            clustering_coeff.append(temp_cluscoeff)
            
            
        # take an average over connectivity matrices and save  
        print(len(strength))
        freqband_result.update({"strength"          : np.array(strength)})
        freqband_result.update({"local efficiency"  : np.mean(np.array(local_efficiency), 0)})
        freqband_result.update({"clustering coeff"  : np.mean(np.array(clustering_coeff), 0)})
        freqband_result.update({"global efficiency" : np.mean(np.array(global_efficiency), 0)})
        freqband_result.update({"path length"       : np.mean(np.array(path_length), 0)})
        freqband_result.update({"radius"            : np.mean(np.array(radius), 0)})
        freqband_result.update({"diameter"          : np.mean(np.array(diameter), 0)})
        
        # save results for each frequency band
        results.update({freq : freqband_result})
    
    return results


def infans_compute_graph_metrics_per_epoch(input_conn, freq_bands):
    """
    This functions gets the connectivity matrices for each recording which are calculated 
    on 30s segments. Then it calculates the graph-theory-based measures for each graph.
    Finally it outputs for each segment
            
    """
  
    no_conn_mats = input_conn.shape[-1] # number of 30s segments / connectivity matrices for each recording
    results      = {} # a dictionary containing the measures for all frequency band 
    
    for i , freq in enumerate(freq_bands.keys()):
        
        freqband_result   = {} # a dictionary containing the measures for each frequency band             
        strength          = [] # a list containing 1-D arrays for nodes' strengths
        betweenness       = [] # a list containing 1-D arrays for nodes' betweenness centralities
        local_efficiency  = [] # a list containing 1-D arrays for nodes' loacal effieciencies
        clustering_coeff  = [] # a list containing 1-D arrays for nodes' clustering coefficients
        path_length       = [] # a list containing the shortest path length of each graph
        global_efficiency = [] # a list containing the global efficiency of each graph
        radius            = [] # a list containing the radius of each graph
        diameter          = [] # a list containing the diameter of each graph
    
        for n in tqdm(range(no_conn_mats)):
            print(n/no_conn_mats)
            
            # select connectivity matrix
            temp_conn = input_conn[:,:,i,n]
            
            # make symmetric
            temp_conn = temp_conn + np.transpose(temp_conn)
                   
            # calculate the strength for each node            
            temp_strength = bct.strengths_und(temp_conn) # 1-D numpy array
            strength.append(temp_strength)
            
            # calculate the betweenness centrality
            L       = bct.weight_conversion(temp_conn, "lengths") # convert weights to lengths
            temp_bc = bct.betweenness_wei(L) # 1-D numpy array
            betweenness.append(temp_bc)
            
            # calculate the path length, global efficiency, radius, and diameter           
            D       = bct.distance_wei(L) # calculate the distance between nodes
            CPL     = bct.charpath(D[0], include_diagonal = False) # the distance matrix must be fed
            temp_pl = CPL[0] # a float number
            temp_ge = CPL[1] # a float number
            temp_ra = CPL[3] # a float number
            temp_di = CPL[4] # a float number
            path_length.append(temp_pl)
            global_efficiency.append(temp_ge)
            radius.append(temp_ra)
            diameter.append(temp_di)
            
            # calculate the local efficiency
            # Note: all weights must be between 0 and 1
            temp_le = bct.efficiency_wei(temp_conn, local = True) # 1-D numpy array
            local_efficiency.append(temp_le)
            
            # calculate the clustering coefficient for each node
            temp_cluscoeff = bct.clustering_coef_wu(temp_conn) # 1-D numpy array
            clustering_coeff.append(temp_cluscoeff)
            
            
        # take an average over connectivity matrices and save  
        freqband_result.update({"strength"          : np.array(strength)})
        freqband_result.update({"local efficiency"  : np.array(local_efficiency)})
        freqband_result.update({"clustering coeff"  : np.array(clustering_coeff)})
        freqband_result.update({"global efficiency" : np.array(global_efficiency)})
        freqband_result.update({"path length"       : np.array(path_length)})
        freqband_result.update({"radius"            : np.array(radius)})
        freqband_result.update({"diameter"          : np.array(diameter)})
        
        # save results for each frequency band
        results.update({freq : freqband_result})
    
    return results


def graph_preparation_for_regression(graph_metrics):
    """
    This function just changes the shape of graph_metrics_0 (or _1) so that it 
    can be easily used for the next step: regression.
    
    please not that it only returens local graph metrics.
    
    input: dict(freq(graph_measures))
    output: np array, shape(number of epochs per subj, no chans*no freq bands) [for ex: 60,8*3]
    
    """
    graph_all_freq =[]
    
    for freq in ['delta1','delta2','theta','alpha','beta']:
        
        g = np.array([graph_metrics[freq][f][:] for f in ["strength","local efficiency", "clustering coeff"]])# (3,60,8)
        graph_per_freq = np.hstack(g) # (60,24) 
        graph_all_freq.append(graph_per_freq) # len:5

    graph_all_freq = np.hstack(np.array(graph_all_freq)) # (60,120)
    
    return(graph_all_freq)


def get_features_names(chan_names,chan_pair_name):
    
    """
    Returns the name of corresponding complexity, connectivity, and graph measures to be used for regression 
    """
    
    freq_bands       = ['delta1','delta2','theta','alpha','beta']
    chan_names       = ['fp1','c3','t3','o1','fp2','c4','t4','o2']
    
    features_complex = list(["hurst_exp"]*8 + ["app_entropy"]*8 + ["samp_entropy"]*8 +
          ["hjorth_mobility_spect"]*8 + ["hjorth_complexity_spect"]*8 +
          ["hjorth_mobility"]*8 + ["hjorth_complexity"]*8 +
          ["higuchi_fd"]*8 + ["katz_fd"]*8 + ["zero_crossings"]*8 +
          ["spect_slope_intercept"]*8 + ["spect_slope_slop"]*8 +
          ["spect_slope_MSE"]*8 +["spect_slope_slop_R2"]*8 +
          ["svd_fisher_info"]*8 +
          ["line_length"]*8 + ["decorr_time"]*8 +
          ["spect_entropy"]*8 + ["svd_entropy"]*8)  
    features_complex_chns =  list(chan_names*8 + chan_names*8 + chan_names*8 +
          chan_names*8 + chan_names*8 +
          chan_names*8 + chan_names*8 +
          chan_names*8 + chan_names*8 + chan_names*8 +
          chan_names*32 + chan_names*8 +
          chan_names*8 + chan_names*8 +
          chan_names*8 + chan_names*8)
    features_complex = [features_complex[i] + " " + features_complex_chns[i] for i in range(len(features_complex))]


    features_con = list(["delta1"]*28 + ["delta2"]*28 + ["theta"]*28 +["alpha"]*28 + ["beta"]*28 )
    chan_pair_name = chan_pair_name*5
    features_con = [features_con[i] + " " + chan_pair_name[i] for i in range(len(features_con))]


    features_g = list(["strength"]*8 + ["local_efficiency"]*8 + ["clustering_coeff"]*8)*5
    features_g_chns = list(chan_names*3)*5
    features_g_freq = list(["delta1"]*24 + ["delta2"]*24 + ["theta"]*24 +["alpha"]*24 + ["beta"]*24 )
    features_gr = [features_g[i] + " " + features_g_chns[i] + " " + features_g_freq[i] for i in range(len(features_g))]
    
    
    
    return(features_complex,features_con,features_gr)