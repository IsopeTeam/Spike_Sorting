#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 01:13:58 2019

@author: lilian
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import generalized_improved as script
from scipy.stats import norm




x = np.linspace(0.0,10.00000001,200000)



################## TEST PHASE-LOCKING ############################################
sig_kappa = sp.sin(4*sp.pi*x) + sp.sin(8*sp.pi*x) + sp.sin(6*sp.pi*x) + sp.sin(20*sp.pi*x)

phase_locking2 = np.linspace(0.0,10.0,21).tolist()
phase_locking4 = np.linspace(0.0,10.0,41).tolist()
no_phase_locking = np.linspace(0.0,10.0,67).tolist()
pseudo_neurons = [[phase_locking2], [phase_locking4], [no_phase_locking]]

time_list = [10.001]
event_name = ["final test"]


sampling_rate = 20000
t_start = 0.
t_stop = 10.0
f_start = 1.
f_stop = 30.
frequency_bin = 0.5
threshold = 50
min_spike_per_event = 1
kap_threshold_computation = True
alpha = 0.05
boot_kap_threshold = 0.5


disp , kap, freq, cluster_freq_kap = script.main_kappa_analysis(sig_kappa, pseudo_neurons, event_name, time_list, sampling_rate, t_start, t_stop, f_start,
         f_stop, frequency_bin, threshold, min_spike_per_event, kap_threshold_computation, alpha,
         boot_kap_threshold)

########################### TEST PHASE-AMPLITUDE COUPLING #########################
print("MI TEST ################")

nb_bin = 20
permut_rep = 4

sig_low = sp.sin(4*sp.pi*x)
sig_high_locked = 0.5*(sig_low+1)*sp.sin(24*sp.pi*x)
sig_high_nolocking = sp.sin(48*sp.pi*x)

sig_MI = sig_low + sig_high_locked +sig_high_nolocking

phase, amplitude, MI_array, sig_array, cluster_freq_MI = script.main_MI_analysis(sig_MI, event_name, time_list, sampling_rate = sampling_rate, t_start = 0., 
                                                            t_stop = t_stop, f_start = f_start,
                                                            f_stop = f_stop, frequency_bin = frequency_bin, 
                                                            threshold = threshold, nb_bin = nb_bin, permut_rep = permut_rep, alpha = alpha)

########################## TEST NOISE ############################################


var_sig = 0.001
noised_sig_kappa = sig_kappa + norm.rvs(0,var_sig,len(x))

var_neuron = 0.01
def noise_neuron(var_neuron, neuron):
    noised_neuron = []
    noise = norm.rvs(0,var_neuron,len(neuron))
    for i in range(len(neuron)):
        noised_neuron.append(min(neuron[i]+noise[i], t_stop))
    return noised_neuron
noised_pseudo_neurons = [[noise_neuron(var_neuron,phase_locking2)], [noise_neuron(var_neuron,phase_locking4)], [noise_neuron(var_neuron,no_phase_locking)]]

noise_disp , noise_kap, noise_freq, noise_cluster_freq_kap = script.main_kappa_analysis(noised_sig_kappa, noised_pseudo_neurons, event_name, time_list, sampling_rate, t_start, t_stop, f_start,
         f_stop, frequency_bin, threshold, min_spike_per_event, kap_threshold_computation, alpha,
         boot_kap_threshold) 



noised_sig_low = sig_low + norm.rvs(0,var_sig,len(x))
noised_sig_high_locked = sig_high_locked + norm.rvs(0,var_sig,len(x))
noised_sig_high_nolocking = sig_high_nolocking + norm.rvs(0,var_sig,len(x))



