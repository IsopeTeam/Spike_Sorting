# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:22:32 2020

@author: Master5.INCI-NSN
"""

#import tridesclous as tdc 
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd 
import warnings
#import elephant.spike_train_generation as Spike
#from quantities import Hz, s, ms
#import elephant.statistics as stat
#import elephant.conversion as conv
#from neo.core import SpikeTrain
#import elephant.kernels as kernels

warnings.simplefilter("ignore", UserWarning)



# Figure for waveforms------------------------------------------------------

def plotwaveforms(clusters_catalogue, waveforms_label, waveforms, wf_rms, wf_alpha, chan_grp, probe_geometry, name, savedir, closefig):
    
    for cluster, idx in zip(np.unique(waveforms_label),range(len(np.unique(waveforms_label)))):
        if cluster < 0: #No waveforms
            continue
        
        fig1 = plt.figure(figsize=(12,4))
        plt.title('{} Average Waveform Cluster {} (ch_group = {})'.format(name,cluster,chan_grp))
        plt.xlabel('Probe location (micrometers)')
        plt.ylabel('Probe location (micrometers)')
        
        for loc, prob_loc in zip(range(len(probe_geometry)), probe_geometry): 
            x_offset, y_offset = prob_loc[0], prob_loc[1]
            #base_x = np.arange(0,len(waveforms[1,:,loc]),1)  
            base_x = np.linspace(-15,15,num=len(waveforms[idx,:,loc])) #Basic x-array for plot, centered
            clust_color = 'C{}'.format(idx)
            
            wave = waveforms[idx,:,loc]+y_offset
            plt.plot(base_x+2*x_offset,wave,color=clust_color)
            plt.fill_between(base_x+2*x_offset,wave-wf_rms[idx],wave+wf_rms[idx], color=clust_color,alpha=wf_alpha)
            
            
        if savedir !=None :
            fig1.savefig('{}/{}_Waveforms_changrp{}_Cluster{}.jpg'.format(savedir,name,chan_grp,cluster))         
            with pd.ExcelWriter('{}/{}_waveforms_changrp{}.xlsx'.format(savedir,name,chan_grp)) as writer:
                #File infos 
                waveform_info = pd.DataFrame(clusters_catalogue)
                waveform_info.to_excel(writer, sheet_name='info')
                for cluster, idx in zip(np.unique(waveforms_label),range(len(np.unique(waveforms_label)))):
                    if cluster < 0: #No waveforms
                        continue
                    clust_WF = pd.DataFrame(waveforms[idx,:,:])      
                    clust_WF.to_excel(writer,sheet_name='cluster {}'.format(cluster))           
        else : 
            print ('No savedir specified : nothing will be saved')
        
        if closefig==True:
            plt.close()
    
    return 0



        
#Spike Times extraction per cluster, aligned segments----------------------------------------        

def rasterplots_psth(spike_times, ep_len, n_seg, seg_id, pos_clustlist, clust_id, chan_grp, STIM, stim_time, stim_duration, water_time, water_duration, name, savedir, closefig):
        
    n_clust=len(pos_clustlist) 
    for cluster, idx in zip(pos_clustlist,range(n_clust)):
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        #  ax1 = fig.add_subplot(311)
        # ax2 = fig.add_subplot(312, sharex=ax1)
        # ax3 = fig.add_subplot(313, sharex=ax2)
        fig.suptitle('Cluster{}'.format(cluster), fontsize=22)
        ax1.set_title('Raster plot', fontsize=16)
        ax2.set_title('PSTH', fontsize=16)
        ax2.set_xlabel('Time (s)')
        ax1.set_ylabel('Segment')        
        ticks = np.arange(0,ep_len,1)
        ax2.set_xticks(ticks)
        
        ax1.axvspan(water_time, water_time+water_duration,color='skyblue',alpha=0.4, label='water release')
        ax2.axvspan(water_time, water_time+water_duration,color='skyblue',alpha=0.4, label='water release')
        if STIM==True:
            ax1.axvspan(stim_time,stim_time+stim_duration,color='lightcoral',alpha=0.3, label='stimulation')
            ax2.axvspan(stim_time,stim_time+stim_duration,color='lightcoral',alpha=0.3, label='stimulation')
        ax1.legend()    
        
        SPIKES_clust = [] #To store spikes for each cluster, one array per segment
        #cluster_list = [] #To store the cluster for file indexing 
        seg_list=np.arange(n_seg) #To store the segments for file indexing 
        
        clust_color = 'C{}'.format(idx)
        #cluster_list.append(str(cluster))
    
        for seg_num in range(n_seg):  
            #len_seg = dataio.get_segment_length(seg_num)  
            # time_vector = np.arange(0,len_seg,1)*sampling_period
            
            temp_ = [] #To store spikes from each cluster
            
            for i,j in np.ndenumerate(seg_id):
                if j == seg_num:
                    if clust_id[i]==cluster:
                          temp_.append(spike_times[i])
                
            SPIKES_clust.append(np.asarray(np.ravel(temp_)))
        
            ax1.eventplot(np.ravel(temp_), lineoffsets=seg_num, linelengths=0.5, linewidth=1, color=clust_color)
       
        
        # PSTH  
        bins = 100
        new_spikelist = np.sort(np.asarray(pd.DataFrame(SPIKES_clust[:])).ravel())
        new_spikelist = new_spikelist[~np.isnan(new_spikelist)]
        ax2.hist(new_spikelist, bins=bins, density=True)
        #ax2.axvspan(water_time, water_time+water_duration,color='lightcoral',alpha=0.4)
        #ax2.axvspan(stim_time,stim_time+stim_duration,color='skyblue',alpha=0.6)
        ax2.set_ylabel('Firing rate')
        
        
        #Mean firing rate
        f_rate=len(new_spikelist)/(ep_len*n_seg)
        print('Cluster{}'.format(cluster))
        print('mean firing rate ={}'.format(f_rate))
        
        # # Firing Frequency
        # sigma = 0.1 *s
        
        # spike_list = np.asarray(pd.DataFrame(SPIKES_clust[:]))
        
        # for indx,i in enumerate(spike_list):
        #     SpikeT = SpikeTrain(i*s,t_start=0.0*s,t_stop=(ep_len+0.5)*s)   # NEO object             
        #     kernel = kernels.GaussianKernel(sigma)
        #     inst_firing_rate = stat.instantaneous_rate(SpikeT, sigma, kernel)
        #     mean_firing_rate = stat.mean_firing_rate(i*s,t_start=0.0*s,t_stop=(ep_len+0.5)*s)
        #     time_x = np.arange(0,ep_len, ep_len/len(inst_firing_rate))
            
        #     if len(i)>0:
        #         print('Mean firing rate of Cluster {}, segment {}:'.format(cluster,indx), mean_firing_rate)
        #     ax3.plot(time_x,inst_firing_rate, label='segment {}'.format(indx))
        #     # ax3.hlines(mean_firing_rate, xmin=SpikeT.t_start, xmax=SpikeT.t_stop, linestyle='--', label='mean segment {}'.format(indx))
        #     ax3.axvspan(water_time, water_time+water_duration,color='lightcoral',alpha=0.4)
        #     ax3.axvspan(stim_time,stim_time+stim_duration,color='skyblue',alpha=0.6)
        #     ax3.set_ylabel('Firing Rate (Hz)')
        #     ax3.legend(loc='upper right', fontsize='x-small')
        
    #SAVE THE SPIKE DATA (or not) ---------------------------------------------    
        if savedir != None:
            sorted_spikes = pd.DataFrame(SPIKES_clust,index=seg_list)
            sorted_spikes.to_excel('{}/{}_SPIKETIMES_PSTH_CLUSTER{}_changrp{}.xlsx'.format(savedir,name,cluster,chan_grp),index_label='Segment')
            plt.savefig('{}/{}_Spikes_PSTH_CLUSTER{}_changrp{}.jpg'.format(savedir,name,cluster,chan_grp))
        else : 
            print ('No savedir specified : nothing will be saved')
            
        if closefig==True:
            plt.close()
            
    return 0
     


#Spike Times extraction, all clusters----------------------------------------
    
def plot_and_save_spikes(spike_times, ep_len, n_seg, seg_id, pos_clustlist, clust_id, chan_grp, STIM, stim_time, stim_duration, water_time, water_duration, name, savedir, closefig) :
    
    for ind in range(len(spike_times)):
        spike_times[ind]=spike_times[ind]+(ep_len*seg_id[ind])
                    
    fig1, ax =plt.subplots(2,1,figsize=(12,8))
    ax[0].set_title('{} All spike times (ch_group = {})'.format(name,chan_grp))
    ax[0].eventplot(spike_times, linewidth=0.1)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Cluster ID')
    ticks = np.arange(0,(ep_len*n_seg),9)
    ax[0].set_xticks(ticks)
    ax[1].set_xticks(ticks)
    
    time_vector = np.arange(0,(ep_len*n_seg),1)

    water_vector = np.arange(water_time,time_vector[-1],float(ep_len))
    for water in water_vector:
        ax[0].axvspan(water,water+water_duration,color='lightcoral',alpha=0.4, label='water release')
        ax[1].axvspan(water,water+water_duration,color='lightcoral',alpha=0.4, label='water release')
        
    if STIM==True:
        stim_vector = np.arange(stim_time,time_vector[-1],float(ep_len))
        for stim in stim_vector:
            ax[0].axvspan(stim,stim+stim_duration,color='skyblue',alpha=0.6, label='stimulation')
            ax[1].axvspan(stim,stim+stim_duration,color='skyblue',alpha=0.6, label='stimulation')
    #ax[0].legend()

    SPIKES = [] #To store all the spikes, one array per cluster
    
    n_clust=len(pos_clustlist) 
    for cluster, idx in zip(pos_clustlist,range(n_clust)):
        
        clust_color = 'C{}'.format(idx)
    
        temp_ = [] #To store spikes from each cluster
    
        for i,j in np.ndenumerate(clust_id):
            if j == cluster:
                temp_.append(spike_times[i])
                
        SPIKES.append(np.asarray(np.ravel(temp_)))
        
        ax[1].eventplot(np.ravel(temp_), lineoffsets=cluster, linelengths=0.5, linewidth=0.5, color=clust_color)
       
    #SAVE THE SPIKE DATA (or not) ---------------------------------------------    
    if savedir != None:
        sorted_spikes = pd.DataFrame(SPIKES,index=pos_clustlist)
        sorted_spikes.to_excel('{}/{}_Spike_times_changrp{}.xlsx'.format(savedir,name,chan_grp),index_label='Cluster')
        fig1.savefig('{}/{}_Spike_times_changrp{}.pdf'.format(savedir,name,chan_grp))
     
    if closefig==True:
        plt.close()
        
    return 0

