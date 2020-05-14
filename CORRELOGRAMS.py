# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:40:27 2020

@author: Anna
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:44:09 2019
@author: Anna

This script allows to plot auto and crosscorrelograms
This script should run in a tridesclous environement where TDC is installed 
Take a look at the DATA INFO section, fill the info, run and let the magic happen. 
"""
#------------------------------------------------------------------------------
#-----------------------------DATA INFO----------------------------------------
#----------------------------FILL BELOW----------------------------------------

#The path of the TDC catalogue file - must be STRING format
path =r'D:/Stage INCI/data/6336 (Atlas - Male)/1500-P13/rbf-20 mW/No stim/tdc_nostim_demo'

#Name of the experiment, protocol... anything to discriminate the date. Will be used
#for datasheets/figure labeling - must be STRING format  
name = '6336(atlasmale)-1500-P13-rbf-20mW-Nostim'

#Where to save datasheets and figures. If None, nothing will be saved  - must be STRING format
savedir = r'D:/Stage INCI/data/6336 (Atlas - Male)/1500-P13/rbf-20 mW/No stim/results/corr'


sampling_rate = 20000 #in Hz

#Stim time ad stim duration in seconds
stim_time = 6
stim_duration = 1
water_time = 2.55
water_duration = 0.15

#Specify the channel group to explore as [#]. Feel free to do them all : [0,1,2,3]
channel_groups=[0]

#If True : close figure automatically (avoids to overhelm screen when looping and debug)
closefig = True

#------------------------------------------------------------------------------
#-----------------------------THE SCRIPT---------------------------------------
#---------------------------DO NOT MODIFY--------------------------------------
import tridesclous as tdc 
from tridesclous.tools import compute_cross_correlograms
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd 

#Load the catalogue
dataio = tdc.DataIO(path)

#Number of segments
n_seg=dataio.nb_segment

#Compute mean episode lenght
sampling_period = 1.0/sampling_rate 
len_trace=0
for seg_num in range(n_seg):  
    len_seg = dataio.get_segment_length(seg_num)
    len_trace += len_seg
ep_len=(len_trace/n_seg)*sampling_period


for chan_grp in channel_groups:
    #Define the constructor and the channel group 
    cc = tdc.CatalogueConstructor(dataio, chan_grp=chan_grp)
    print ('--- Experiment : {} ---'.format(name))
    print ('Catalogue loaded from {}'.format(path))
    print ('----Channel group {}----'.format(chan_grp))
    
    #The cluster list for the spikes 
    clust_id = cc.all_peaks['cluster_label']
    
    #The spike times
    spike_indexes = cc.all_peaks['index']
    spike_times= spike_indexes*sampling_period                                        
    
    #The segment list for the spikes 
    seg_id = cc.all_peaks['segment']       

    #Positive clusters list
    clust_list = np.unique(cc.clusters['cluster_label'])
    mask=(clust_list >= 0)
    pos_clustlist=clust_list[mask]
    n_clust=len(pos_clustlist)    
      
        


bin_size=0.01
window_size=1

ccg, bins = compute_cross_correlograms(spike_indexes, clust_id, seg_id, pos_clustlist, sampling_rate, window_size, bin_size, symmetrize=True,)

        
#bins = bins * 1000. #to ms
nbins=len(bins)-1
bins=bins[:nbins]


plt.style.use('seaborn')

for r in range(0, n_clust):
    for c in range(r, n_clust):
        count = ccg[r, c, :]         
        fig=plt.figure()
        plt.bar(bins, count, width = bin_size, align='edge')
        if r==c:
            plt.title('Autocorrelogram_cluster{}'.format(r))
            if savedir != None:
                plt.savefig('{}/Autocorrelogram_cluster{}_changrp{}.pdf'.format(savedir,r,chan_grp))
        else:  
            plt.title('Crosscorrelogram_clusters{}&{}'.format(r,c))
            if savedir != None:
                plt.savefig('{}/Crosscorrelogram_clusters{}&{}_changrp{}.pdf'.format(savedir,r,c,chan_grp))
        

    if closefig==True:
        plt.close()
        
    