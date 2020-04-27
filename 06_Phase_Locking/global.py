#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 18:23:31 2019

@author: lilian
"""
import numpy as np
import generalized_improved as script
import matplotlib.pyplot as plt

channel_groups = [ [0,2,4,6] , [1,3,5,7]]#, [9,11,13,15], [8,10,12,14] ]
channel_groups_spike_sorting = [i for i in range(len(channel_groups))]
nb_trial_no_stim = 5
t_stop = 9.0
sampling_rate = 20000
f_start = 1.
f_stop = 50.
freq_bin = 0.5
scalo_threshold = 70

#definition of events

event_name = ["sound1","delay1","sound2","delay2", "water", "rest"]
time_interval = [0.5,1.,0.5,0.5,0.205]  #the list of the times associated (in seconds), without the last
if len(time_interval) ==1:
    time_list = [t_stop]
else :
    end = t_stop-sum(time_interval)
    time_interval = time_interval + [end] #the list of the times associated (in seconds), with the last, with respect to the duration
    time_list = script.addition_time_list (time_interval) #cumulative sum of event duration

#localisation of the signal
file_loc_sig = "/Users/lilian/Documents/python_data/DATA_LFP/2019-06-13T16-51-30Concatenate_1500um_P13.rbf"

#alpha risk for each test performed
alpha = 0.05


#phase-locking analysis entries
file_loc_spike = "/Users/lilian/Documents/python_data/DATA_spike/6336-1500-P13_Spike_times_changrp_"
kap_threshold_computation = True
boot_kap_threshold = 0.5 #not important if kap_threshold_computation is True
min_spike_per_event = 10

#phase-amplitude coupling analysis entries
nb_bin_MI_histogram = 18
nb_permutation_replication = 4


def average_channel_group(file_loc_sig,channel_groups):
    sig = np.fromfile(file_loc_sig,dtype='float64').reshape(-1,16)
    T = []
    for chgrp in range(len(channel_groups)):
        print("averaging channel group", chgrp)
        T.append([])
        for sample in range(np.shape(sig)[0]):
            av = 0
            for i in range(len(channel_groups[chgrp])):
                av+=sig[sample,channel_groups[chgrp][i]]
            T[chgrp].append(av/len(channel_groups[chgrp]))
    return T

avr_sig = average_channel_group(file_loc_sig, channel_groups)

def LFP_cutting_by_stim(nb_trial_no_stim, avr_sig, channel_groups, sampling_rate, t_stop):
    split_in = int((np.shape(avr_sig)[1]/sampling_rate)/t_stop)
    array_no_stim = []
    array_stim = []
    for i in range(len(channel_groups)):
        array_no_stim.append([])
        array_stim.append([])
        X = script.trial_cutting_LFP(avr_sig[i], split_in, sampling_rate, t_stop)
        array_no_stim[i] = X[0:nb_trial_no_stim] #i suppose the nostim are the first
        array_stim[i] = X[nb_trial_no_stim:]
    return array_no_stim , array_stim

sig_no_stim, sig_stim = LFP_cutting_by_stim(nb_trial_no_stim, avr_sig, channel_groups, sampling_rate, t_stop)

def lecture_chgrp(channel_groups, file_loc_spike):
    spike_sorting_per_chgrp = []
    for i in range(len(channel_groups)):
        spike_sorting_per_chgrp.append(script.lecture(file_loc_spike,i))
    return spike_sorting_per_chgrp

def train_cutting_by_stim(nb_trial_no_stim, spike_sorting_per_chgrp, channel_groups, t_stop):
    spike_sorting_no_stim = []
    spike_sorting_stim = []
    for chgrp in range(len(spike_sorting_per_chgrp)):
        neuron_spike_stim = []
        neuron_spike_no_stim = []
        for neuron in range(len(spike_sorting_per_chgrp[chgrp])):
            neuron_spike_stim.append([])
            neuron_spike_no_stim.append([])    
            neuron_spike_no_stim[neuron] = script.trial_cutting_train(spike_sorting_per_chgrp[chgrp], t_stop)[neuron][0:nb_trial_no_stim]
            neuron_spike_stim[neuron] = script.trial_cutting_train(spike_sorting_per_chgrp[chgrp], t_stop)[neuron][nb_trial_no_stim:]
        spike_sorting_no_stim.append(neuron_spike_no_stim)  
        spike_sorting_stim.append(neuron_spike_stim)
    return spike_sorting_no_stim , spike_sorting_stim

Neurons = lecture_chgrp(channel_groups, file_loc_spike)
Neurons_no_stim , Neurons_stim = train_cutting_by_stim(nb_trial_no_stim, Neurons, channel_groups, t_stop)

def new_concatenation(sig_no_stim, sig_stim):
    concat_no_stim = []
    concat_stim = []
    for chgrp in range(len(sig_no_stim)):#same length for sig_stim
        concat_no_stim.append(np.asarray(sig_no_stim[chgrp]).flatten().tolist())
        concat_stim.append(np.asarray(sig_stim[chgrp]).flatten().tolist())
    return concat_no_stim , concat_stim

concat_no_stim, concat_stim = new_concatenation(sig_no_stim, sig_stim)

def overall_kap_analysis(channel_groups, concat_no_stim, concat_stim, spike_sorting_no_stim, spike_sorting_stim, t_stop, f_start, f_stop, freq_bin, scalo_threshold = 70, min_spike_per_event = 10, kap_threshold_computation = True, alpha = 0.05, boot_kap_threshold = 0.5):
    disp = [[],[]]
    kappa_array = [[],[]]
    freq_selection = [[],[]]
    cluster_freq = [[],[]]
    for chgrp in range(len(channel_groups)):
        print()
        print("PHASE LOCKING ANALYSIS FOR THE CHANNEL GROUP {}".format(chgrp))
        
        print()
        print("NO STIMULATION")
        Buffer_no_stim = script.main_kappa_analysis(concat_no_stim[chgrp], spike_sorting_no_stim[chgrp], event_name, time_list, sampling_rate = sampling_rate, t_start = 0., 
                                                            t_stop = t_stop, f_start = f_start,
                                                            f_stop = f_stop, frequency_bin = freq_bin, 
                                                            threshold = scalo_threshold, min_spike_per_event = min_spike_per_event, kap_threshold_computation = kap_threshold_computation, alpha = alpha, boot_kap_threshold = boot_kap_threshold)
        print()
        print("STIMULATION")
        Buffer_stim = script.main_kappa_analysis(concat_stim[chgrp], spike_sorting_stim[chgrp], event_name, time_list, sampling_rate = sampling_rate, t_start = 0., 
                                                            t_stop = t_stop, f_start = f_start,
                                                            f_stop = f_stop, frequency_bin = freq_bin, 
                                                            threshold = scalo_threshold, min_spike_per_event = min_spike_per_event, kap_threshold_computation = kap_threshold_computation, alpha = alpha, boot_kap_threshold = boot_kap_threshold)
        disp[0].append(Buffer_no_stim[0])
        disp[1].append(Buffer_stim[0])
        kappa_array[0].append(Buffer_no_stim[1])
        kappa_array[1].append(Buffer_stim[1])
        freq_selection[0].append(Buffer_no_stim[2])
        freq_selection[1].append(Buffer_stim[2])
        cluster_freq[0].append(Buffer_no_stim[3])
        cluster_freq[1].append(Buffer_stim[3])

    return disp , kappa_array, freq_selection, cluster_freq

def overall_MI_analysis(channel_groups, concat_no_stim, concat_stim, t_stop, f_start, f_stop, freq_bin, scalo_threshold = 70, nb_bin = 20, permut_rep = 200, alpha = 0.05):
    phase = [[],[]]
    ampl = [[],[]]
    MI_array = [[],[]]
    sig_array = [[],[]]
    cluster_freq = [[],[]]
    for chgrp in range(len(channel_groups)):
        print()
        print("PHASE-AMPLITUDE COUPLING ANALYSIS FOR THE CHANNEL GROUP {}".format(chgrp))
        
        print()
        print("NO STIMULATION")
        Buffer_no_stim = script.main_MI_analysis(concat_no_stim[chgrp], event_name, time_list, sampling_rate = sampling_rate, t_start = 0., 
                                                            t_stop = t_stop, f_start = f_start,
                                                            f_stop = f_stop, frequency_bin = freq_bin, 
                                                            threshold = scalo_threshold, nb_bin = nb_bin, permut_rep = permut_rep, alpha = alpha)
        print()
        print("STIMULATION")
        Buffer_stim = script.main_MI_analysis(concat_stim[chgrp], event_name, time_list, sampling_rate = sampling_rate, t_start = 0., 
                                                            t_stop = t_stop, f_start = f_start,
                                                            f_stop = f_stop, frequency_bin = freq_bin, 
                                                            threshold = scalo_threshold, nb_bin = nb_bin, permut_rep = permut_rep, alpha = alpha)
        phase[0].append(Buffer_no_stim[0])
        phase[1].append(Buffer_stim[0])
        ampl[0].append(Buffer_no_stim[1])
        ampl[1].append(Buffer_stim[1])
        MI_array[0].append(Buffer_no_stim[2])
        MI_array[1].append(Buffer_stim[2])
        sig_array[0].append(Buffer_no_stim[3])
        sig_array[1].append(Buffer_stim[3])
        cluster_freq[0].append(Buffer_no_stim[4])
        cluster_freq[1].append(Buffer_stim[4])
    
    return phase, ampl, MI_array, sig_array, cluster_freq



disp , kap, freq, cluster_freq_kap = overall_kap_analysis(channel_groups, concat_no_stim, concat_stim, Neurons_no_stim, Neurons_stim, t_stop, f_start, f_stop, freq_bin, scalo_threshold = scalo_threshold, min_spike_per_event = min_spike_per_event, kap_threshold_computation = kap_threshold_computation, alpha = alpha, boot_kap_threshold = boot_kap_threshold)            

phase, amplitude, MI, sig_array, cluster_freq_MI = overall_MI_analysis(channel_groups, concat_no_stim, concat_stim, t_stop, f_start, f_stop, freq_bin, scalo_threshold = scalo_threshold, nb_bin = nb_bin_MI_histogram, permut_rep = nb_permutation_replication, alpha = alpha)

script.histogram(phase[1][0][3][4],amplitude[1][0][3][4])



    
    






    

    
    

                
       


        
    
    

