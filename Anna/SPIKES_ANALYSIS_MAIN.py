# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:44:09 2019
@author: Ludo and Fede (modified from Sam Garcia) + Anna
​
This script allows to extract spike times for each cluster from tridesclous catalogue
in excel sheet + figure plot 
+ PSTH
+ auto and cross correlograms
This script should run in a tridesclous environement where TDC is installed 
Take a look at the DATA INFO section, fill the info, run and let the magic happen. 
"""
#------------------------------------------------------------------------------
#-----------------------------DATA INFO----------------------------------------
#----------------------------FILL BELOW----------------------------------------

# Some parameters
depth = '1600'
protocol = 'P15'
power = '7mW'
stim = 'NoStim'

#The path of the TDC catalogue file - must be STRING format
path =r'\\SRV4\Master5.INCI-NSN$\Mes Documents\data\5490(atlas-female)\1600\P15\rbf\No stim\tdc_5490_1600_P15_nostim'
#path =r'C:/Users/ANNA-STAGE-PASTOUCHE/Documents/stageINCI/data/5490(atlas-female)/h5-1300-P13/rbf-stim/tdc_5490_1300P13_stim'

#Name of the experiment, protocol... anything to discriminate the date. Will be used for datasheets/figure labeling - must be STRING format  
name = '5490-1500P13'

#Where to save datasheets and figures. If None, nothing will be saved  - must be STRING format
savedir = r'\\SRV4\Master5.INCI-NSN$\Mes Documents\data\5490(atlas-female)\1600\P15\rbf\No stim\results'
savedir_waveforms = savedir + '\Waveforms'
savedir_allspikes = savedir + '\Allspikes'
savedir_spikesPSTH = savedir + '\Spikes-PSTH'
savedir_autocorr = savedir + '\Auto-corr'
savedir_crosscorr = savedir + '\Cross-corr'

#Recordings sampling rate
sampling_rate = 20000 #in Hz

#Stim time ad stim duration in seconds
stim_time = 1.5
stim_duration = 0.8
water_time = 2.55
water_duration = 0.15

#False if no stim control recordings
STIM=False

#Specify the channel group to explore as [#]. Feel free to do them all : [0,1,2,3]
channel_groups=[0]

#If True : close figure automatically (avoids to overhelm screen when looping and debug)
closefig = True

#The opacity for the waveform rms. 1. = solid, 0. = transparent 
wf_alpha =0.2


#------------------------------------------------------------------------------
#-----------------------------THE SCRIPT---------------------------------------
#---------------------------DO NOT MODIFY--------------------------------------
import tridesclous as tdc 
import numpy as np 
from matplotlib import pyplot as plt 
#import pandas as pd 
from spikes_waves_psth import plotwaveforms, rasterplots_psth, plot_and_save_spikes
from auto_cross_corr import auto_corr, cross_corr

#Plot style
plt.style.use('seaborn')
#plt.style.use('ggplot')


#Extract spikes and clusters info from TDC catalogue---------------------------

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
    spike_index = cc.all_peaks['index']
    spike_times= spike_index*sampling_period
    
    #The segment list for the spikes 
    seg_id = cc.all_peaks['segment']       
        
    #The cluster label for median waveform, the median waveforms and the median rms
    waveforms_label = cc.clusters['cluster_label']
    waveforms = cc.centroids_median
    wf_rms =cc.clusters['waveform_rms']
    
    clusters_catalogue=cc.clusters
    
    #The probe geometry and specs 
    probe_geometry = cc.geometry
    probe_channels = cc.nb_channel
       
    #Positive clusters list
    clust_list = np.unique(waveforms_label)
    mask=(clust_list >= 0)
    pos_clustlist=clust_list[mask]   
    
    
    # Plot and save figures for waveforms--------------------------------------
    plotwaveforms(clusters_catalogue, waveforms_label, waveforms, wf_rms, wf_alpha, chan_grp, probe_geometry, name, savedir_waveforms, closefig)
            
    #Plot inividual clusters spiketrains (aligned segments), PSTH and save figures---------------        
    rasterplots_psth(spike_times, ep_len, n_seg, seg_id, pos_clustlist, clust_id, chan_grp, STIM, stim_time, stim_duration, water_time, water_duration, name, savedir_spikesPSTH, closefig)   
   
    #Plot all spikes and spikes times per cluster, and save spikes data--------
    plot_and_save_spikes(spike_times, ep_len, n_seg, seg_id, pos_clustlist, clust_id, chan_grp, STIM, stim_time, stim_duration, water_time, water_duration, name, savedir_allspikes, closefig)
         
    #Compute, plot and save auto-correlograms----------------------------------
    bin_size=0.01
    window_size=1
    symmetrize=True
    auto_corr(spike_index, sampling_rate, seg_id,  pos_clustlist, clust_id, chan_grp, window_size, bin_size, symmetrize, savedir_autocorr, closefig)
   
    #Compute, plot and save cross-correlograms---------------------------------
    bin_size=0.001
    window_size=0.1
    cross_corr(spike_index, sampling_rate, seg_id,  pos_clustlist, clust_id, chan_grp, window_size, bin_size, symmetrize, savedir_crosscorr, closefig)