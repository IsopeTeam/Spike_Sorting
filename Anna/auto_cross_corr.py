# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:40:27 2020
@author: Anna

This script allows to plot auto and crosscorrelograms
This script should run in a tridesclous environement where TDC is installed 
"""

#import tridesclous as tdc 
from tridesclous.tools import compute_cross_correlograms
#import numpy as np 
from matplotlib import pyplot as plt 
#import pandas as pd 



#Compute and plot auto-correlations
def auto_corr(spike_indexes, sampling_rate, seg_id, pos_clustlist, clust_id, chan_grp, window_size, bin_size, symmetrize, savedir, closefig):
    
    ccg, bins = compute_cross_correlograms(spike_indexes, clust_id, seg_id, pos_clustlist, sampling_rate, window_size, bin_size, symmetrize)

    #bins = bins * 1000. #to ms
    nbins=len(bins)-1
    bins=bins[:nbins]

    n_clust=len(pos_clustlist) 
    for r in range(0, n_clust):
        for c in range(r, n_clust):
            if r==c:
                count = ccg[r, c, :]         
                fig=plt.figure(figsize=(12,8))
                plt.xlabel('Time (s)') 
                plt.bar(bins, count, width = bin_size, align='edge')
                plt.title('Autocorrelogram_cluster{}'.format(r), fontsize=16)
                
                if savedir != None:
                    plt.savefig('{}/Autocorrelogram_cluster{}_changrp{}.jpg'.format(savedir,r ,chan_grp))
            
    if closefig==True:
        plt.close()
        
    return 0


        
#Compute and plot cross-correlations
def cross_corr(spike_indexes, sampling_rate, seg_id, pos_clustlist, clust_id, chan_grp, window_size, bin_size, symmetrize, savedir, closefig):
    
    ccg, bins = compute_cross_correlograms(spike_indexes, clust_id, seg_id, pos_clustlist, sampling_rate, window_size, bin_size, symmetrize)

    #bins = bins * 1000. #to ms
    nbins=len(bins)-1
    bins=bins[:nbins]

    n_clust=len(pos_clustlist) 
    for r in range(0, n_clust):
        for c in range(r, n_clust):
            if r!=c:
                count = ccg[r, c, :]         
                fig=plt.figure(figsize=(12,8))
                plt.xlabel('Time (s)') 
                plt.bar(bins, count, width = bin_size, align='edge')
                plt.title('Crosscorrelogram_clusters{}&{}'.format(r,c), fontsize=16)
                
                if savedir != None:
                    plt.savefig('{}/Crosscorrelogram_clusters{}&{}_changrp{}.jpg'.format(savedir,r,c,chan_grp))
              
            if closefig==True:
                plt.close()
        
    return 0