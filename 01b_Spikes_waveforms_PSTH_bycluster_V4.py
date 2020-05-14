
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:44:09 2019
@author: Ludo and Fede (modified from Sam Garcia) + Anna

This script allows to extract spike times for each cluster from tridesclous catalogue
in excel sheet
+ figure plot 
+PSTH
This script should run in a tridesclous environement where TDC is installed 
Take a look at the DATA INFO section, fill the info, run and let the magic happen. 
"""
#------------------------------------------------------------------------------
#-----------------------------DATA INFO----------------------------------------
#----------------------------FILL BELOW----------------------------------------

#The path of the TDC catalogue file - must be STRING format
path =r'D:/F.LARENO.FACCINI/Preliminary Results/Ephy/6336 (Atlas - Male)/h5/1300/P13/rbf/No stim/tdc_6336_1300_P13_7mW_NoStim_SingleUnit'

#Name of the experiment, protocol... anything to discriminate the date. Will be used
#for datasheets/figure labeling - must be STRING format  
name = '6336-1300-P13_NoStim'

#Where to save datasheets and figures. If None, nothing will be saved  - must be STRING format
savedir = r'D:/F.LARENO.FACCINI/Preliminary Results/Ephy/Spike Sorting Results'

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

#The opacity for the waveform rms. 1. = solid, 0. = transparent 
wf_alpha =0.2


#------------------------------------------------------------------------------
#-----------------------------THE SCRIPT---------------------------------------
#---------------------------DO NOT MODIFY--------------------------------------
import tridesclous as tdc 
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd 


#Load the catalogue
dataio = tdc.DataIO(path)

#Number of segments
n_seg=dataio.nb_segment

#Compute mean episode lenght (in seconds)
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
    
    #The probe geometry and specs 
    probe_geometry = cc.geometry
    probe_channels = cc.nb_channel
    
    #Positive clusters list
    clust_list = np.unique(waveforms_label)
    mask=(clust_list >= 0)
    pos_clustlist=clust_list[mask]
    n_clust=len(pos_clustlist)    


    #Figure for waveforms------------------------------------------------------
          
    for cluster, idx in zip(pos_clustlist, range(n_clust)):
        fig1 = plt.figure(figsize=(10,12))
        plt.title('{} Average Waveform Cluster {} (ch_group = {})'.format(name,cluster,chan_grp))
        plt.xlabel('Probe location (micrometers)')
        plt.ylabel('Probe location (micrometers)')
        print(cluster)
        for loc, prob_loc in zip(range(len(probe_geometry)), probe_geometry): 
            x_offset, y_offset = prob_loc[0], prob_loc[1]
            #base_x = np.arange(0,len(waveforms[1,:,loc]),1)  
            base_x = np.linspace(-15,15,num=len(waveforms[idx+1,:,loc])) #Basic x-array for plot, centered
            clust_color = 'C{}'.format(idx)

            wave = waveforms[idx+1,:,loc]+y_offset
            plt.plot(base_x+2*x_offset,wave,color=clust_color)
            plt.fill_between(base_x+2*x_offset,wave-wf_rms[idx+1],wave+wf_rms[idx+1], color=clust_color,alpha=wf_alpha)

    
        if savedir !=None :
            fig1.savefig('{}/{}_Waveforms_changrp{}_Cluster{}.pdf'.format(savedir,name,chan_grp,cluster))         
            with pd.ExcelWriter('{}/{}_waveforms_changrp{}.xlsx'.format(savedir,name,chan_grp)) as writer:
                #File infos 
                waveform_info = pd.DataFrame(cc.clusters)
                waveform_info.to_excel(writer, sheet_name='info')
                for cluster, idx in zip(pos_clustlist, range(n_clust)):
                    clust_WF = pd.DataFrame(waveforms[idx,:,:])      
                    clust_WF.to_excel(writer,sheet_name='cluster {}'.format(cluster))           
        else : 
            print ('No savedir specified : nothing will be saved')
        
        if closefig==True:
            plt.close()



    #Spike Times extraction per cluster, aligned segments----------------------------------------        
        
    for cluster, idx in zip(pos_clustlist,range(n_clust)):
        fig, ax =plt.subplots(2, 1, sharex=True)
        fig.suptitle('CLUSTER{}'.format(idx), fontsize=16)
        ax[0].set_title('Raster plot')
        ax[0].set_ylabel('Segment') 
        ax[1].set_title('PSTH')
        ax[1].set_ylabel('Firing rate') 
        ax[1].set_xlabel('Time (s)')       
        ticks = np.arange(0,ep_len,1)
        ax[1].set_xticks(ticks)
        
        ax[0].axvspan(water_time, water_time+water_duration,color='skyblue',alpha=0.6)
        ax[0].axvspan(stim_time,stim_time+stim_duration,color='lightcoral',alpha=0.4)  
        
        SPIKES_clust = [] #To store spikes for each cluster, one array per segment
        #cluster_list = [] #To store the cluster for file indexing 
        seg_list=np.arange(n_seg) #To store the segments for file indexing 
        
        clust_color = 'C{}'.format(idx)
        #cluster_list.append(str(cluster))
    
        for seg_num in range(n_seg):  
            #len_seg = dataio.get_segment_length(seg_num)  
            #time_vector = np.arange(0,len_seg,1)*sampling_period
            temp_ = [] #To store spikes from each cluster            
            for i,j in np.ndenumerate(seg_id):
                if j == seg_num:
                    if clust_id[i]==cluster:
                          temp_.append(spike_times[i])
                
            SPIKES_clust.append(np.asarray(np.ravel(temp_)))
        
            ax[0].eventplot(np.ravel(temp_), lineoffsets=seg_num, linelengths=0.5, linewidth=0.5, color=clust_color)
       
        #PSTH
        nbins=int(ep_len*6)
        plt.style.use('seaborn')
        #ax[1].hist(SPIKES_clust, stacked=True, alpha=0.9)
        ax[1].hist(np.hstack(SPIKES_clust), nbins, alpha=0.9)

        #SAVE THE SPIKE DATA (or not) ---------------------------------------------    
        if savedir != None:
            sorted_spikes = pd.DataFrame(SPIKES_clust,index=seg_list)
            sorted_spikes.to_excel('{}/{}_SPIKETIMES_aligned_CLUSTER{}_changrp{}.xlsx'.format(savedir,name,cluster,chan_grp),index_label='Segment')
            plt.savefig('{}/{}_Spike_times_aligned_CLUSTER{}_changrp{}.pdf'.format(savedir,name,cluster,chan_grp))
        else : 
            print ('No savedir specified : nothing will be saved')
                        
        if closefig==True:
            plt.close()
     
        
