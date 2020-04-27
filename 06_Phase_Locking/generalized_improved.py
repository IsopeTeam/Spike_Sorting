#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:01:27 2019

@author: lilian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:50:33 2019

@author: lilian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:28:54 2019

@author: lilian
"""

#from compute_timefreq import compute_timefreq
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import fftpack
import pandas as pd
import random
import warnings
import math 
import scipy as sp
from scipy.stats import f, norm, vonmises, uniform
from scipy.special import i1, i0 


#---------------------------------------!!!!!!!!!! FUNCTIONS !!!!!!!!!!-------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#---------------------------------------------------DO NOT--------------------------------------------------------
#---------------------------------------------------MODIFY--------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#--------------------------------------------GO BELOW FOR DATA INPUT----------------------------------------------
def generate_wavelet_fourier(len_wavelet, f_start, f_stop, delta_freq, 
            sampling_rate, f0, normalisation):
    """
    Compute the wavelet coefficients at all scales and makes its Fourier transform.
    When different signal scalograms are computed with the exact same coefficients, 
        this function can be executed only once and its result passed directly to compute_morlet_scalogram
        
    Output:
        wf : Fourier transform of the wavelet coefficients (after weighting), Fourier frequencies are the first 
    """
    # compute final map scales
    scales = f0/np.arange(f_start,f_stop,delta_freq)*sampling_rate
    # compute wavelet coeffs at all scales
    xi=np.arange(-len_wavelet/2.,len_wavelet/2.)
    xsd = xi[:,np.newaxis] / scales
    wavelet_coefs=np.exp(complex(1j)*2.*np.pi*f0*xsd)*np.exp(-np.power(xsd,2)/2.)

    weighting_function = lambda x: x**(-(1.0+normalisation))
    wavelet_coefs = wavelet_coefs*weighting_function(scales[np.newaxis,:])

    # Transform the wavelet into the Fourier domain
    #~ wf=fft(wavelet_coefs.conj(),axis=0) <- FALSE
    wf=fftpack.fft(wavelet_coefs,axis=0)
    wf=wf.conj() # at this point there was a mistake in the original script
    
    return wf


def convolve_scalogram(sig, wf):
    """
    Convolve with fft the signal (in time domain) with the wavelet
    already computed in freq domain.
    
    Parameters
    ----------
    sig: numpy.ndarray (1D, float)
        The signal
    wf: numpy.array (2D, complex)
        The wavelet coefficient in fourrier domain.
    """
    n = wf.shape[0]
    assert sig.shape[0]<=n, 'the sig.size is longer than wf.shape[0] {} {}'.format(sig.shape[0], wf.shape[0])
    sigf=fftpack.fft(sig,n)
    wt_tmp=fftpack.ifft(sigf[:,np.newaxis]*wf,axis=0)
    wt = fftpack.fftshift(wt_tmp,axes=[0])
    return wt

def compute_timefreq(sig, sampling_rate, f_start, f_stop, delta_freq=1., nb_freq=None,
                f0=2.5,  normalisation = 0.,  min_sampling_rate=None, wf=None,
                t_start=0., zero_pad=True, joblib_memory=None):
    """
    
    """
    #~ print 'compute_timefreq'
    sampling_rate = float(sampling_rate)
    
    if nb_freq is not None:
        delta_freq = (f_stop-f_start)/nb_freq
    
    if min_sampling_rate is None:
        min_sampling_rate =  min(4.* f_stop, sampling_rate)
        
    
    #decimate
    ratio = int(sampling_rate/min_sampling_rate)
    #~ print 'ratio', ratio
    if ratio>1:
        # sig2 = tools.decimate(sig, ratio)
        sig2 = scipy.signal.decimate(sig, ratio,n=4, zero_phase=True) #ORDER OF 4 FOR SMALL FREQ INTERVALLS !!!!!
    else:
        sig2 = sig
        ratio=1
    
    tfr_sampling_rate = sampling_rate/ratio
    #~ print 'tfr_sampling_rate', tfr_sampling_rate
    
    n_init = sig2.size
    if zero_pad:
        n = int(2 ** np.ceil(np.log(n_init)/np.log(2))) # extension to next power of  2
    else:
        n = n_init
    #~ print 'n_init', n_init, 'n', n
    if wf is None:
        if joblib_memory is None:
            func = generate_wavelet_fourier
        else:
            func = joblib_memory.cache(generate_wavelet_fourier)
        wf = func(n, f_start, f_stop, delta_freq, 
                            tfr_sampling_rate, f0, normalisation)
    
    assert wf.shape[0] == n
    
    wt = convolve_scalogram(sig2, wf)
    wt=wt[:n_init,:]
    
    freqs = np.arange(f_start,f_stop,delta_freq)
    times = np.arange(n_init)/tfr_sampling_rate + t_start
    return wt, times, freqs, tfr_sampling_rate


#-------------------------RIDGE EXTRACTION in 2D-------------------------------
#------------------------------------------------------------------------------
def ridge_map(ampl_map, threshold=70.):
    max_power = np.max(ampl_map) #Max power observed in frequency spectrum
    freq_power_threshold = float(threshold) #The threshold range for power detection of the ridge 
    cut_off_power = max_power/100.0*freq_power_threshold #Computes power above trheshold
    
    boolean_map = ampl_map >= cut_off_power #For plot 
    
    value_map = np.copy(ampl_map)
    
    for i,j in np.ndenumerate(ampl_map):
        if j <= cut_off_power:
            value_map[i] = 0.0 #For computation, all freq < trhesh = 0.0
            
    return boolean_map, value_map


#-------------------------SIGNAL FILTERING FUNCTION----------------------------
#------------------------------------------------------------------------------
def filter_signal(signal, order=8, sample_rate=20000,freq_low=400,freq_high=2000, axis=0):
    
    import scipy.signal
    
    Wn = [freq_low/(sample_rate/2),freq_high/(sample_rate/2)]
    
    sos_coeff = scipy.signal.iirfilter(order,Wn,btype='band',ftype='butter',output='sos')
    
    filtered_signal = scipy.signal.sosfiltfilt(sos_coeff, signal, axis=axis)
    
    return filtered_signal
    

########################CLUSTER RIDGE EXTRACTION################################################

#NOTE : you can change the percentage of overlap, here it is 90% (see collage)

#definition of correspondance
def collage(A,B,percent = 0.90):
    assert(len(A)==len(B))
    n = len(A)
    A_extract = [i for i in range(n) if A[i]!=0]
    B_extract = [i for i in range(n) if B[i]!=0]
    number_paste = 0
    for i in range(len(A_extract)):
        for j in range(len(B_extract)):
            if i==j:
                number_paste = number_paste + 1
    return number_paste/len(A_extract)>=percent
    

#empty frequency band suppresing
def filtering_empty(ridge_map_comp):
    interest_freq = []
    for i in range (ridge_map_comp.shape[1]):
        if len([x for x in ridge_map_comp.transpose()[i] if x!=0])!=0:
            interest_freq.append([ridge_map_comp.transpose()[i],i])
    return(interest_freq)
    
#identification of clusters
def cluster_freq_building(interest_freq):
    cluster_freq = []
    array = []
    for j in range(len(interest_freq)-1):
        if interest_freq[j][1]==interest_freq[j+1][1]-1:
            if collage(interest_freq[j][0],interest_freq[j+1][0]):
                array.append(interest_freq[j][1])
                if j == len(interest_freq)-2:
                    array.append(interest_freq[j+1][1])
                    cluster_freq.append(array)
            else :
                cluster_freq.append(array + [interest_freq[j][1]])
                array = []
                if j == len(interest_freq)-2:
                    cluster_freq.append([interest_freq[j+1][1]])
        else :
            cluster_freq.append(array + [interest_freq[j][1]])
            array = []
            if j == len(interest_freq)-2:
                cluster_freq.append([interest_freq[j+1][1]])
    return cluster_freq

#conversion in interval of frequency
def conversion_interval(cluster_freq):
    for i in range(len(cluster_freq)):
        if len(cluster_freq[i])>2:
            cluster_freq[i] = [min(cluster_freq[i]),max(cluster_freq[i])]
    return cluster_freq        
        
def conversion_frequency (cluster_freq, delta_freq, f_start):
    A = []
    for i in range(len(cluster_freq)):
        A.append([])
        for j in range(len(cluster_freq[i])):
            A[i].append(cluster_freq[i][j]*delta_freq + f_start)
    return A
    
#construction of a 2-d array with time and phase for each sampling   
def arrangement_time_array(instant_phase, sampling_rate):
    A = []
    n = len(instant_phase[0])
    for i in range (len(instant_phase)):
        #instant_phase[i] = instant_phase[i]
        A.append([])
        for j in range(n):
            A[i].append([j/sampling_rate, instant_phase[i][j]])
    return A

#the function to do computation in extraction_instant_phase. the delta_freq must be declared in the main
def phase_band_computation(sig, sampling_rate, global_f_stop, band_f_start, band_f_stop):
    
    sampling_rate = float(sampling_rate)
    
    min_sampling_rate =  min(4.* global_f_stop, sampling_rate)
    
    #decimate
    ratio = int(sampling_rate/min_sampling_rate)
    #~ print 'ratio', ratio
    if ratio>1:
        # sig2 = tools.decimate(sig, ratio)
        sig2 = scipy.signal.decimate(sig, ratio,n=4, zero_phase=True) #ORDER OF 4 FOR SMALL FREQ INTERVALLS !!!!!
    else:
        sig2 = sig
        ratio=1
    
    band_sig = filter_signal(sig2, order = 8, sample_rate = min_sampling_rate, freq_low = band_f_start, freq_high = band_f_stop)
    band_analytic_signal = scipy.signal.hilbert(band_sig)
    band_instantaneous_phase = np.angle(band_analytic_signal)
    
    return band_instantaneous_phase.tolist()

#the function to do computation in extraction_ampl_envelope.
def ampl_band_computation(sig, sampling_rate, global_f_stop, band_f_start, band_f_stop):
    
    sampling_rate = float(sampling_rate)
    
    min_sampling_rate =  min(4.* global_f_stop, sampling_rate)
    
    #decimate
    ratio = int(sampling_rate/min_sampling_rate)
    #~ print 'ratio', ratio
    if ratio>1:
        # sig2 = tools.decimate(sig, ratio)
        sig2 = scipy.signal.decimate(sig, ratio,n=4, zero_phase=True) #ORDER OF 4 FOR SMALL FREQ INTERVALLS !!!!!
    else:
        sig2 = sig
        ratio=1
    
    band_sig = filter_signal(sig2, order = 8, sample_rate = min_sampling_rate, freq_low = band_f_start, freq_high = band_f_stop)
    band_analytic_signal = scipy.signal.hilbert(band_sig)
    band_ampl = np.abs(band_analytic_signal)
    
    return band_ampl.tolist()
    
#extraction of phase for each cluster of frequency
def extraction_instant_phase (cluster_freq, phase_map, delta_freq, sig, f_start, f_stop, sampling_rate):
    instant_phase = []
    for i in range(len(cluster_freq)):
        assert(len(cluster_freq[i])<3) #if not do conversion interval
        if len(cluster_freq[i])==1:
            instant_phase.append(phase_map.transpose()[cluster_freq[i][0]].tolist())
        else:
            freq_band = [cluster_freq[i][0]*delta_freq+f_start,cluster_freq[i][1]*delta_freq+f_start]
            instant_phase.append(phase_band_computation(sig, sampling_rate, global_f_stop = f_stop, 
                            band_f_start = freq_band[0], band_f_stop = freq_band[1]))
    return instant_phase

#extraction of amplitude for each cluster of frequency
def extraction_ampl_envelope (cluster_freq, ampl_map, delta_freq, sig, f_start, f_stop, sampling_rate):
    ampl = []
    for i in range(len(cluster_freq)):
        assert(len(cluster_freq[i])<3) #if not do conversion interval
        if len(cluster_freq[i])==1:
            ampl.append(ampl_map.transpose()[cluster_freq[i][0]].tolist())
        else:
            freq_band = [cluster_freq[i][0]*delta_freq+f_start,cluster_freq[i][1]*delta_freq+f_start]
            ampl.append(ampl_band_computation(sig, sampling_rate, global_f_stop = f_stop, 
                            band_f_start = freq_band[0], band_f_stop = freq_band[1]))
    return ampl

def trial_cutting_LFP(sig, split_in, sampling_rate, t_stop):
    
    array_return = []
    sampling_period = 1./sampling_rate
    limit_time = t_stop * split_in
    if len(sig)*sampling_period < limit_time: #We look if the signal is too long for the split_in and t_stop
        #Delimits the length and begining of each episodes 
        step = int(len(sig)/split_in)
        starts = np.arange(0,len(sig),step)
    else: 
        
        limit_sample = int(limit_time*sampling_rate)
        sig = sig[0:limit_sample]
        
        #Delimits the length and begining of each episodes 
        step = int(len(sig)/split_in)
        starts = np.arange(0,len(sig),step)
        
    for i in range(len(starts)):
        
        segment_start = int(starts[i])
        segment_stop = int(segment_start+step)
        
        array_return.append(sig[segment_start:segment_stop])
    
    return array_return



###############################READING NEURON FILES##############################################

#lecture des fichiers
def lecture(file_loc_train,ch_grp):   
    file_loc="{}{}.xlsx".format(file_loc_train,ch_grp)
    reader=pd.read_excel(file_loc)
    reader = np.array(reader)
    reader = list(reader)
    
    for i in range(len(reader)):
        reader[i] = list(reader[i])
        reader[i] = [x for x in reader[i] if np.isnan(x) == False]
        reader[i] = reader[i][1:]#elimination of the first value who is an indent for excel array
            
    return(reader)
    
#découpage des spikes trains collés
def trial_cutting_train (Neurons, t_stop):
    D = []
    for neuron in range(len(Neurons)):
        neuron_per_trial = []
        num_episodes = math.ceil(float(Neurons[neuron][-1])/t_stop)#on a peut-être un gros problème
        for i in range(num_episodes):
            neuron_per_trial.append([])
        ep = 0
        for j in range(len(Neurons[neuron])):
            if (Neurons[neuron][j]>=float(t_stop)*ep and Neurons[neuron][j]<float(t_stop)*(ep+1)): 
                neuron_per_trial[ep].append(Neurons[neuron][j]-float(t_stop)*ep)
            else:
                while not (Neurons[neuron][j]>=float(t_stop)*ep and Neurons[neuron][j]<float(t_stop)*(ep+1)):
                    ep = ep+1
                neuron_per_trial[ep].append(Neurons[neuron][j]-float(t_stop)*ep)
        D.append(neuron_per_trial)
    return D

#selection of the 10% most interessants train spike
def select(Trains, percent = 10):
    len_train = []
    for i in range(len(Trains)):
        len_train.append([i,len(Trains[i])])
    len_train.sort(key = lambda x : x[1], reverse = True)
    nb_selected = int(len(Trains)*percent/100)
    index_selected = []
    i = 0
    while i<=nb_selected:
        index_selected.append(len_train[i][0])
        i+=1
    list_selected = []
    for i in index_selected:
        list_selected.append(Trains[i])
    return list_selected

###########################ANALYSIS OF TRAIN SPIKE WITH LFP########################

#phase_histogramm, LFP_phase_time is a 2-d array with time and phase for each sampling. i'm assuming that LFP_phase_time is modulo 2*pi
def phase_disposition (neuron, LFP_phase_time):
    neuron_copy = np.copy(neuron) #to remove spikes that have already been placed
    neuron_copy = neuron_copy.tolist()
    assert(len(np.shape(neuron))==1)
    assert(np.shape(LFP_phase_time)[1]==2)
    phase_disposed_spike = []
    time_phase = []
    spike_accepted = []

    #n = int(len(LFP_phase_time)/10)
    for i in range(np.shape(LFP_phase_time)[0]-1):
        #if i%n==0: print("{}th computation over {}".format(i,len(LFP_phase_time)))
        if LFP_phase_time[i][1]<LFP_phase_time[i+1][1] or LFP_phase_time[i][1]-LFP_phase_time[i+1][1]<sp.pi:#i suppose is [time,phase]
            time_phase.append([i,LFP_phase_time[i][0]]) 

        else:  
            time_phase.append([i,LFP_phase_time[i][0]])
            for j in range(len(neuron_copy)):
                if neuron_copy[j] == "placed":
                    continue
                placed = False
                
                for x in range(len(time_phase)-1):
                    if time_phase[x][1]<neuron_copy[j]<time_phase[x+1][1]:
                        phase_disposed_spike.append( (LFP_phase_time[ time_phase[x][0] ][1]+LFP_phase_time[time_phase[x+1][0]][1])/2 ) #mean of time window phase before and after the spike
                        placed = True
                        break
                    elif neuron_copy[j] == time_phase[x][1]:
                        phase_disposed_spike.append(LFP_phase_time[ time_phase[x][0] ][1])
                        placed = True
                        break 
                    
                if not placed:
                    if neuron_copy[j] == time_phase[len(time_phase)-1][1]:
                        phase_disposed_spike.append(LFP_phase_time[ time_phase[x][0] ] [1])
                        placed = True
                    elif neuron_copy[j] < time_phase[0][1]:
                        phase_disposed_spike.append( (LFP_phase_time[time_phase[0][0]][1]/2+LFP_phase_time[time_phase[0][0]-1][1]/2+sp.pi)%(sp.pi)) #type of mean for radian numbers
                        placed = True
                if placed:
                    spike_accepted.append(neuron_copy[j])
                    neuron_copy[j] = "placed"
            time_phase = []
    if np.shape(time_phase)[0]==0: #implies that the time_phase was computed in the last for loop, ie the last LFP_time_phase is alone in a new phase cycle
        if len(spike_accepted)!=len(neuron_copy):
            if len(neuron_copy)-len(spike_accepted) == 1:
                phase_disposed_spike.append((LFP_phase_time[-1][1]/2+LFP_phase_time[-2][1]/2+sp.pi)%(sp.pi))
                spike_accepted.append(neuron_copy[-1])
                neuron_copy[-1] = "placed"
            else: 
                print("i couldn't place all spikes, check if there is any problem in your input")
    
    else:#analyse of the last time_phase
        
        time_phase.append([np.shape(LFP_phase_time)[0]-1,LFP_phase_time[np.shape(LFP_phase_time)[0]-1][0]]) #
        for j in range(len(neuron_copy)):
            if neuron_copy[j] == "placed":
                continue
            placed = False
            for x in range(len(time_phase)-1):
                if time_phase[x][1]<neuron_copy[j]<time_phase[x+1][1]:
                    phase_disposed_spike.append( (LFP_phase_time[ time_phase[x][0] ][1]+LFP_phase_time[time_phase[x+1][0]][1])/2 ) #mean of time window phase before and after the spike
                    placed = True
                    break
                elif neuron_copy[j] == time_phase[x][1]:
                    phase_disposed_spike.append(LFP_phase_time[ time_phase[x][0] ][1])
                    placed = True
                    break 
                
            if not placed:
                if neuron_copy[j] == time_phase[len(time_phase)-1][1]:
                    phase_disposed_spike.append(LFP_phase_time[ time_phase[x][0] ] [1])
                    placed = True
                elif neuron_copy[j] > time_phase[-1][1]:
                        phase_disposed_spike.append(LFP_phase_time[time_phase[0][0]][1]) #approximation for the last spike if he is out of time
                        placed = True 
                elif neuron_copy[j] < time_phase[0][1]:
                        phase_disposed_spike.append( (LFP_phase_time[time_phase[0][0]][1]/2+LFP_phase_time[time_phase[0][0]-1][1]/2+sp.pi)%(sp.pi)) #type of mean for radian numbers
                        placed = True
                        
            if placed:
                spike_accepted.append(neuron_copy[j])
                neuron_copy[j] = "placed"
    
        if len(spike_accepted)!=len(neuron_copy):
            print("i couldn't place all spikes, check if there is any problem in your input")
    phase_disposed_spike.sort()
    #print("we placed {} spikes over {}".format(len(spike_accepted),len(neuron)))
    return phase_disposed_spike

#time of event appening since we record
def addition_time_list (time_interval):
    growing_time_interval = []
    for i in range(len(time_interval)):
        if i == 0 : 
            growing_time_interval.append(time_interval[i])
        else :    
            growing_time_interval.append(time_interval[i]+growing_time_interval[-1])
    return growing_time_interval

#result : for each event a list of LFP_signal
def event_cutting_LFP(LFP_signal, time_list):
    list_LFP_cutted = []
    for event in range(len(time_list)):
        list_LFP_cutted.append([])
    event = 0
    
    if type(LFP_signal[0])==list: #for arranged time array
        for i in range(len(LFP_signal)):
            if LFP_signal[i][0] < time_list[event]:
                list_LFP_cutted[event].append(LFP_signal[i])
            else :
                event+=1
                list_LFP_cutted[event].append(LFP_signal[i])
    
    else:
        for i in range(len(LFP_signal)):
            if LFP_signal[i] < time_list[event]:
                list_LFP_cutted[event].append(LFP_signal[i])
            else :
                event+=1
                list_LFP_cutted[event].append(LFP_signal[i])
                
    return list_LFP_cutted

#result : for each event a list of spike time
def event_cutting_train(train, time_list):
    list_train_cutted = []
    for event in range(len(time_list)):
        list_train_cutted.append([])
    event = 0
    for i in range(len(train)):
        if train[i]<time_list[event]:
            list_train_cutted[event].append(train[i])
        else:
            while not train[i]<time_list[event]:
                event+=1
            list_train_cutted[event].append(train[i])
    return list_train_cutted 
  
#same function for array    
def LFP_array_cutting(array, time_list):
    A = []
    for freq in range(len(array)):
        A.append([])
        for trial in range(len(array[freq])):
            A[freq].append([])
            A[freq][trial] = event_cutting_LFP(array[freq][trial], time_list)
    return A
   
def train_array_cutting(array, time_list):
    A = []
    for neuron in range(len(array)):
        A.append([])
        for trial in range(len(array[neuron])):
            A[neuron].append([])
            A[neuron][trial] = event_cutting_train(array[neuron][trial],time_list)
    return A

def transpose_to_event_per_trial(array, nb_event):
    A = []
    for neuron in range(len(array)):
        A.append([])
        for event in range(nb_event):
            A[neuron].append([])
            for trial in range(len(array[neuron])):
                A[neuron][event].append(array[neuron][trial][event])
    return A

#phase_disposition for a group of LFP_phase_time and neuron
def event_phase_disposition(Neurons, LFP):
    disp = []
    for neuron in range(len(Neurons)):
        disp.append([])
        for event in range(len(Neurons[neuron])):
            disp[neuron].append([])
            for freq in range(len(LFP)):
                disp[neuron][event].append([])
                for trial in range(len(Neurons[neuron][event])):
                    if len(Neurons[neuron][event][trial])==0:
                        calcul = []
                    else:
                        calcul = phase_disposition(Neurons[neuron][event][trial], LFP[freq][event][trial])
                    disp[neuron][event][freq] += calcul
                disp[neuron][event][freq].sort()
                if not disp[neuron][event][freq]:
                    disp[neuron][event][freq] = "no spikes in this event"
                        #if calcul[1]:
                         #   error.append([i,event,j])    
    return disp

################################VON MISES : PLOT AND ESTIMATION########################################

#inversion of a function to get the maximum likelihood estimator of kappa
def A1inverted_approximation(R):
    assert(R>=0 or np.isnan(R))
    if 0<=R<0.53:
        return 2*R + R**3 + 5*R**5/6
    if 0.53<=R<=0.85:
        return -0.4 + 1.39*R + 0.43/(1-R)
    else:
        return 1/(R**3-4*R**2+3*R)

#construction of the R coefficient
def R_coef(phase_disposed_spike_time):
    C = sp.cos(phase_disposed_spike_time)
    S = sp.sin(phase_disposed_spike_time)
    
    C = sum(C)
    S = sum(S)
    
    R = sp.sqrt(C**2+S**2)/len(phase_disposed_spike_time)
    return(R)
    
#maximum likelihood estimation of kappa
def estimation(R, n, bias = False):
    k = A1inverted_approximation(R)
    if bias:
        if k<2:
            k = max([0,k-2/(n*k)])
        else:
            k = (n-1)**3*k/(n**3-n)
    return k

#main function
def kappa(phase_disposed_spike_time, bias = False):
    R = R_coef(phase_disposed_spike_time)
    n = len(phase_disposed_spike_time)
    if (n<16 & (not bias)):
        warnings.warn("your data size is small, you should use the bias correction")  

    k = estimation(R, n, bias)
    return k

#building confidence interval for kappa estimation
def confidence_interval (phase_disposed_spike_time, normal_approximation = False, bootstrap = True, rep = 1000, alpha = 0.05, bias = False):
    for i in range (len(phase_disposed_spike_time)):
        if type(phase_disposed_spike_time[i])==str:
            return "the data is not relevant enough"
    R = R_coef(phase_disposed_spike_time)
    n = len(phase_disposed_spike_time)
    k = kappa(phase_disposed_spike_time, bias)
    
    if (normal_approximation):
        warnings.warn("be sure your data size is very large before using normal approximation (>200)")   
        var_k = 1/(n*(1-R**2-R/k))
        qnorm_alpha = norm.ppf(1-alpha/2)
    
    if (bootstrap): #construction of the bootstrap estimators and the upper and lower bound of the associated percentile confidence interval
        low = 0
        high = 0
        boot_estimators = []
        for i in range(rep):
            R_boot = 0
            k_boot = 0
            boot_sample = []
            for step in range(n):
                boot_sample.append(random.choice(phase_disposed_spike_time))
            R_boot = R_coef(boot_sample)
            #penser à faire le biais
            k_boot = estimation(R_boot, n, bias)
            boot_estimators.append(k_boot)
        list.sort(boot_estimators)
        low = boot_estimators[int(alpha/2*rep)+1]
        high = boot_estimators[int((1-alpha/2)*rep)] #rounded number, the way i do it is not a big deal if rep is large
    if (bootstrap & normal_approximation):
        return [[max([k-qnorm_alpha*sp.sqrt(var_k/n),0]),k+qnorm_alpha*sp.sqrt(var_k/n)],[low,high]]
    
    if (bootstrap):
        return [low,high]
    
    if (normal_approximation):
        return [k-qnorm_alpha*sp.sqrt(var_k/n),k+qnorm_alpha*sp.sqrt(var_k/n)]
    
    if (not (normal_approximation | bootstrap)):
        warnings.warn("i can't build a confidence interval if you don't choose a way to build it. If you don't know what to choose, choose bootstrap = True")
        return k


#particular function for test_kappa_dif
def func_1(x):
    return math.asin(sp.sqrt(3/8)*x)
def func_2(x):
    return math.asinh((x-1.089)/0.258)

#two-sample test for equality of kappa
def test_kappa_dif(phase_disposition1, phase_disposition2, alpha = 0.05):
    for i in range (len(phase_disposition1)):
        if type(phase_disposition1[i])==str:
            return "the data is not relevant enough"
    for i in range (len(phase_disposition2)):
        if type(phase_disposition2[i])==str:
            return "the data is not relevant enough"
    R = R_coef(phase_disposition1+phase_disposition2)
    R1 = R_coef(phase_disposition1)
    R2 = R_coef(phase_disposition2)
    n1 = len(phase_disposition1)
    n2 = len(phase_disposition2)
    print("H0 : kappa1 == kappa2, H1 : kappa1 != kappa2")
    if R<0.45:
        STAT = 2*(func_1(2*R1)-func_1(2*R2))/sp.sqrt(3*(1/(n1-4)+1/(n2-4)))
        print(STAT)
        if (STAT<-norm.ppf(1-alpha/2)) | (STAT>norm.ppf(1-alpha/2)):
            return "on rejette H0 au seuil alpha = {}".format(alpha)
        else:
            return "on ne rejette pas H0 au seuil alpha = {}, on peut donc affirmer que les deux échantillons ont le même phase-locking".format(alpha)
    
    if 0.45<=R<=0.70:
        STAT = (func_2(R1)-func_2(R2))/(0.893*sp.sqrt(1/(n1-3)+1/(n2-3)))
        if (STAT<-norm.ppf(1-alpha/2)) | (STAT>norm.ppf(1-alpha/2)):
            return "on rejette H0 au seuil alpha = {}".format(alpha)
        else:
            return "on ne rejette pas H0 au seuil alpha = {}, on peut donc affirmer que les deux échantillons ont le même phase-locking".format(alpha)
    
    if R>0.7:
        STAT = ((n1-R1)*(n2-1)/((n2-R2)*(n1-1)))
        if (STAT<f.ppf(q = alpha/2, dfn = n1-1, dfd = n2-2)) | (STAT>f.ppf(q = 1-alpha/2, dfn = n1-1, dfd = n2-2)):
            return "on rejette H0 au seuil alpha = {}".format(alpha)
        else:
            return "on ne rejette pas H0 au seuil alpha = {}, on peut donc affirmer que les deux échantillons ont le même phase-locking".format(alpha)

def mean_angles(phase_disposition):
    C = sp.cos(phase_disposition)
    S = sp.sin(phase_disposition)
    C = sum(C)
    S = sum(S)
    if C>=0:
        return sp.arctan(S/C)
    else:
        return sp.arctan(S/C)+sp.pi
    
#function to fit histogramm of spikes and vonmises fitting
def plot_fitting(phase_disposition, confidence_interval_show = True, alpha = 0.05, nb_bin = 20):
    fig, ax = plt.subplots()
    ax.hist(phase_disposition, bins = nb_bin, density = True)
    coef_fitting = [kappa(phase_disposition),mean_angles(phase_disposition)]
    x = np.linspace(-sp.pi,sp.pi,500)
    ax.plot(x, vonmises.pdf(x, coef_fitting[0], coef_fitting[1]))
    ax.set_title("plot_example")
    ax.set_xlabel("phase")
    ax.set_ylabel("histogramm")
    return fig, ax
    #if confidence_interval_show:
    #    I = confidence_interval(phase_disposition, alpha = alpha)
     #   ax.legend("{}<kappa<{}".format(I[0],I[1]))
    
#array of kappa per neuron per event per freq
def kappa_array(disposition_array, min_spike = 10, automatic_biais_corr = False):
    kappa_array = []
    for neuron in range(len(disposition_array)):
        kappa_array.append([])
        for event in range(len(disposition_array[neuron])):
            kappa_array[neuron].append([])
            for freq in range(len(disposition_array[neuron][event])):
                if type(disposition_array[neuron][event][freq]) == str:
                    kappa_array[neuron][event].append("no spikes")
                    continue
                if len(disposition_array[neuron][event][freq])<min_spike:
                    kappa_array[neuron][event].append("not enough spike to be relevant")
                    continue
                if len(disposition_array[neuron][event][freq])<16 or automatic_biais_corr:
                    kappa_array[neuron][event].append(kappa(disposition_array[neuron][event][freq], bias = True))
                else:
                    kappa_array[neuron][event].append(kappa(disposition_array[neuron][event][freq]))
    return kappa_array

def disp_per_freq(array):
    A = []
    nb_freq = len(array[0][0])
    for neuron in range(len(array)):
        A.append([])
        for freq in range(nb_freq):
            A[neuron].append([])
            for event in range(len(array[neuron])):
                A[neuron][freq].append(array[neuron][event][freq])
    return A

def CI_array(disposition, min_spike = 10, alpha = 0.05):
    A = []
    for n in range(len(disposition)):
        A.append([])
        for fr in range(len(disposition[n])):
            print("computation confidence interval neuron {}, event {}".format(n,fr))
            A[n].append([])
            for e in range(len(disposition[n][fr])):
                if type(disposition[n][fr][e]) == str:
                    A[n][fr].append("no spikes")
                    continue
                elif len(disposition[n][fr][e])<min_spike:
                    A[n][fr].append("not enough spike to be relevant")
                    continue
                else:
                    A[n][fr].append(confidence_interval(disposition[n][fr][e], alpha = 0.05))
    return A

def phase_locked_freq_selection_per_event(kappa_array_per_event, conv_cluster_freq, conversion_freq = False):
    max_kappa = []
    
    for neuron in range(len(kappa_array_per_event)):
        max_kappa.append([])
        for event in range(len(kappa_array_per_event[neuron])):
            if type(kappa_array_per_event[neuron][event][0])==str:
                max_kappa[neuron].append("not enough data")
                continue
            else:
                max_kappa[neuron].append([0,0])
                for freq in range(len(kappa_array_per_event[neuron][event])):
                    if kappa_array_per_event[neuron][event][freq]>max_kappa[neuron][event][0]:
                        max_kappa[neuron][event][0]=kappa_array_per_event[neuron][event][freq]
                        if conversion_freq:
                            max_kappa[neuron][event][1]=conv_cluster_freq[freq]
                        else:
                            max_kappa[neuron][event][1]=freq
    return max_kappa

#phase locking is biased for frequency bands an integer time greater than the real phase locking frequency, we select the first over a threshold 

#function to select the biaised frequency bands
def big_selection_per_event(kappa_array_per_event, conv_cluster_freq, conversion_freq = False, threshold = 20):
    big_kappa = []
    
    for neuron in range(len(kappa_array_per_event)):
        big_kappa.append([])
        for event in range(len(kappa_array_per_event[neuron])):
            if type(kappa_array_per_event[neuron][event][0])==str:
                big_kappa[neuron].append("not enough data")
            else:
                big_kappa[neuron].append([])
                for freq in range(len(kappa_array_per_event[neuron][event])):
                    if kappa_array_per_event[neuron][event][freq]>threshold:
                        if conversion_freq:
                            big_kappa[neuron][event].append([kappa_array_per_event[neuron][event][freq],conv_cluster_freq[freq]])
                        else:
                            big_kappa[neuron][event].append([kappa_array_per_event[neuron][event][freq],freq])
    return big_kappa

#function to select the real phase-locking frequency band
def fondamental_selection_per_event(kappa_array, conversion_freq, conv_cluster_freq, kap_threshold = 20):
    big_kappa = big_selection_per_event(kappa_array, conversion_freq = conversion_freq, threshold = kap_threshold, conv_cluster_freq = conv_cluster_freq)
    A = []
    for neuron in range(len(big_kappa)):
        A.append([])
        for event in range(len(big_kappa[neuron])):
            if type(big_kappa[neuron][event])==str:
                A[neuron].append("not enough data in this event")
            elif len(big_kappa[neuron][event])==0:
                max_kappa=[0,0]
                for freq in range(len(kappa_array[neuron][event])):
                    if kappa_array[neuron][event][freq]>max_kappa[0]:
                        max_kappa[0]=kappa_array[neuron][event][freq]
                        max_kappa[1]=conv_cluster_freq[freq] 
                A[neuron].append(max_kappa)
            else:
                A[neuron].append(big_kappa[neuron][event][0])#the first frequency whose the kappa is big really caries the phase-locking
    return A 

def border_length(disp, min_spike_per_event):
    max_length = 0
    min_length = len(disp[0][0][0])
    for neuron in range(len(disp)):
        for event in range(len(disp[neuron])):
            for freq in range(len(disp[neuron][event])):
                if len(disp[neuron][event][freq])>max_length:
                    max_length = len(disp[neuron][event][freq])
                if len(disp[neuron][event][freq])<min_length:
                    min_length = len(disp[neuron][event][freq])
    if min_length<min_spike_per_event:
        min_length = min_spike_per_event
    return [min_length, max_length]

def threshold_index(nb_unif, sample_size, alpha):
    bias = False
    if sample_size<16:
        bias = True
    sample = []
    for i in range(nb_unif):
        sample.append(list(uniform.rvs(size = sample_size, loc = -sp.pi, scale = 2*sp.pi)))
    unif_proportion = 0
    threshold = 0
    while unif_proportion < 1-alpha:
        buffer = 0
        threshold += 0.01
        for i in range(len(sample)):
            if kappa(sample[i], bias)<threshold:
                buffer += 1
        unif_proportion = float(buffer/nb_unif)
        #print(unif_proportion, threshold)
    return threshold

def A(x):
    return i1(x)/i0(x)

def survive_uniform(x,n):
    f = 1/sp.sqrt( (A(x)/x * (1-A(x)**2+A(x)/x)))
    g = sp.exp(-n * ( x*A(x) -sp.log(i0(x)) ) )
    return f*g

def corr_survive_uniform(x,n):
    X = survive_uniform((pow(n,3)+n)/pow(n-1,3)*x,n)*survive_uniform(2,n)
    Y = survive_uniform(0.5*(x+sp.sqrt(x**2+8/n)),n)*(1-survive_uniform(2,n))
    return X+Y

def threshold_theory_index(list_sample_size, alpha = 0.05):
    list_threshold = []
    for size in list_sample_size:
        z = 0.01
        if size>=16:
            while survive_uniform(z,size)>alpha:
                z += 0.005
            list_threshold.append(z-0.0025)
        else :
            while corr_survive_uniform(z,size)>alpha:
                z += 0.005
            list_threshold.append(z-0.0025)
    return list_threshold

#plotting function to see the threshold
"""
list_sample_size = np.arange(8,74,1)
fig, ax = plt.subplots()
X = threshold_theory_index(list_sample_size, 0.05)
ax.plot(list_sample_size,X)
for i in range(len(X)): X[i] = [X[i],survive_uniform(X[i],list_sample_size[i])]
"""
def function_threshold_sample_size(list_sample_size, nb_unif, alpha):
    threshold_list = []
    threshold = 0
    for i in range(len(list_sample_size)):
        print("threshold computation for the sample size ", list_sample_size[i])
        threshold = threshold_index(nb_unif, list_sample_size[i], alpha)
        threshold_list.append(threshold)
    #plt.plot(list_sample_size, threshold_list)
    return threshold_list  

def func(x,a,b,c):
    return a*np.exp(-b*x)+c

def fitting_function_threshold_sample_size(threshold_list, list_sample_size):
    coefficient = sp.optimize.curve_fit(func, list_sample_size,  threshold_list)
    fit = coefficient[0][0] * np.exp(-coefficient[0][1]*np.asarray(list_sample_size))+coefficient[0][2]
    return fit  

def frequency_selection_sample_threshold(fitted_threshold, kap, disp, conv_cluster_freq, min_spike_considered):
    selection_array = []
    for neuron in range(len(kap)):
        selection_array.append([])
        for event in range(len(kap[neuron])):
            locking = False
            if type(kap[neuron][event][0])==str:
                selection_array[neuron].append("not enough data")
                continue
            
            for freq in range(len(kap[neuron][event])):
                index = max(len(disp[neuron][event][freq])-min_spike_considered,0)
                #print(index)
                if kap[neuron][event][freq]>fitted_threshold[index]:
                    selection_array[neuron].append([kap[neuron][event][freq],conv_cluster_freq[freq]])
                    locking = True
                    break
            if not locking:
                selection_array[neuron].append("no phase locking")
    return selection_array
          
def frequency_selection_bootstrap(CI_array, boot_kap_threshold, conv_cluster_freq):
    selection_array = []
    for neuron in range(len(CI_array)):
        selection_array.append([])
        for event in range(len(CI_array[neuron])):
            locking = False
            if type(CI_array[neuron][event][0])==str:
                selection_array[neuron].append("not enough data")
                continue
            for freq in range(len(CI_array[neuron][event])):
                if CI_array[neuron][event][freq][0]> boot_kap_threshold:
                    selection_array[neuron].append([CI_array[neuron][event][freq],conv_cluster_freq[freq]])
                    locking = True
                    break
            if not locking:
                selection_array[neuron].append("no phase locking")
    return selection_array

def print_event_phase_locking(freq_per_neuron, event_name, event_studied):
    print("we look if we have phase_locking in event {}, {}".format(event_studied,event_name[event_studied]))
    for i in range(len(freq_per_neuron)):
        print("neuron {}".format(i),freq_per_neuron[i][event_studied])
        
def print_each_event(freq_select, event_name):
    for i in range(len(freq_select[0])):
        print()
        print_event_phase_locking(freq_select, event_name, i)  
        
################################MODULATION INDEX########################################

def histogram(phase, ampl, nb_bin = 20):
    hist_array = []
    for n in range(nb_bin):
        hist_array.append([])
    
    for i in range(len(phase)):
        for n in range(nb_bin):
            if n*(2*np.pi)/nb_bin<=phase[i]+np.pi and phase[i]+np.pi<=(n+1)*(2*np.pi)/nb_bin:
                hist_array[n].append(ampl[i])
    P_A_hist = []
    
    for n in range(nb_bin):
        if len(hist_array[n])!=0:
            P_A_hist.append(sum(hist_array[n])/len(hist_array[n]))
        else:
            P_A_hist.append(0)
        
    s = sum(P_A_hist)
    for n in range(nb_bin):
        P_A_hist[n] = P_A_hist[n]/s
        
    return P_A_hist

def MI_comput(hist):   
    modulation_index = 0
    nb_bin = len(hist)
    for n in range(nb_bin):
        modulation_index = modulation_index + (1/np.log(nb_bin))*hist[n]*np.log(hist[n]*nb_bin)
    
    return modulation_index

def permut_test(phase, ampl, MI, nb_bin = 20, rep = 1000, alpha = 0.05):
    MI_dist = []
    cut = 0
    permuted_ampl = []
    fake_hist = []
    
    for i in range(rep):
        cut = random.randint(0,len(ampl)-1)
        permuted_ampl = ampl[cut:]+ampl[0:cut]
        fake_hist = histogram(phase, permuted_ampl, nb_bin = nb_bin)
        MI_dist.append(MI_comput(fake_hist))
    
    significant = False
    if MI>MI_dist[int(rep*(1-alpha))]:
        significant = True
        
    return significant

def histogram_array (analysis_phase, analysis_ampl, nb_bin = 20):
    hist_array = []
    for freq_low in range(len(analysis_phase)-1):
        hist_array.append([])
        for freq_high in range(freq_low+1, len(analysis_phase)):
            hist_array[freq_low].append([])
            for event in range(len(analysis_phase[freq_low])):
                #print(freq_low, freq_high, event)
                hist_array[freq_low][freq_high-freq_low-1].append(histogram(analysis_phase[freq_low][event],analysis_ampl[freq_high][event], nb_bin = nb_bin))
    return hist_array
    
def MI_array (hist_array):
    MI_array = []
    for freq_low in range(len(hist_array)):
        MI_array.append([])
        for freq_high in range(len(hist_array[freq_low])):
            MI_array[freq_low].append([])
            for event in range(len(hist_array[freq_high][freq_low])):
                MI_array[freq_low][freq_high].append(MI_comput(hist_array[freq_high][freq_low][event]))
    return MI_array

def signification_array(analysis_phase, analysis_ampl, MI_array, nb_bin = 20, rep = 1000, alpha = 0.05):
    sig_array = []
    for freq_low in range(len(MI_array)):
        sig_array.append([])
        print("permutation test for low freq {}".format(freq_low))
        for freq_high in range(len(MI_array[freq_low])):
            print("            ", "high freq {}".format(freq_high+freq_low+1))
            sig_array[freq_low].append([])
            for event in range(len(MI_array[freq_low][freq_high])):
                sig_array[freq_low][freq_high].append(permut_test(analysis_phase[freq_low][event], analysis_ampl[freq_high+freq_low+1][event], MI_array[freq_low][freq_high][event], nb_bin = nb_bin, rep = rep, alpha = 0.05))
    return sig_array

#-------------------------------------------------------MAIN-----------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
def main_kappa_analysis(sig, Neurons, event_name, time_list, sampling_rate = 20000, t_start = 0., t_stop = 9.0, f_start = 1.,
         f_stop = 50., frequency_bin = 0.5, threshold = 70, min_spike_per_event = 10, kap_threshold_computation = True, alpha = 0.05,
         boot_kap_threshold = 0.5):
    
    #--------------------------------------------- THAT'S IT-------------- -------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------  
    
    #Morlet wavelet analysis
    complex_map, map_times, freqs, tfr_sampling_rate = compute_timefreq(sig, sampling_rate, f_start, f_stop, delta_freq=frequency_bin, nb_freq=None,
                        f0=2.5,  normalisation = 0., t_start=t_start)
    print("scalo computation complete")
    #amplitude and phase map
    ampl_map = np.abs(complex_map)
    phase_map = np.angle(complex_map) 
       
    #ridge extraction over the threshold 
    ridge_map_plot, ridge_map_comp = ridge_map(ampl_map, threshold=threshold)
    print("ridge done")
    #selection, grouping and conversion of frequency filtered signal with amplitude over the threshold    
    interest_freq = filtering_empty(ridge_map_comp)
    cluster_freq = cluster_freq_building(interest_freq)
    cluster_freq = conversion_interval(cluster_freq)
    conv_cluster_freq = conversion_frequency(cluster_freq, frequency_bin, f_start)
    print("cluster freq done", conv_cluster_freq)
    #extraction and arrangement of the phase for each frequency group
    
    instant_phase_raw = extraction_instant_phase(cluster_freq, phase_map, frequency_bin, sig, f_start, f_stop, sampling_rate)
    
    
    #cutting in trials
    
    split_in = int((len(sig)/sampling_rate)/t_stop)
    
    instant_phase_per_trial = []
    for freq in range(len(instant_phase_raw)):
        trial_cut = trial_cutting_LFP(instant_phase_raw[freq], split_in, tfr_sampling_rate, t_stop)
        instant_phase_per_trial.append(arrangement_time_array(trial_cut,tfr_sampling_rate))
    print("phase computation complete")
    
    #--------------------------------------------- KAPPA Analysis -----------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    #cutting the signal_per_trial and the train_per_trial by each event
    instant_phase_per_trial_per_event = LFP_array_cutting(instant_phase_per_trial, time_list)
    Neurons_per_trial_per_event = train_array_cutting(Neurons, time_list)
    
    print("cutting by event")
    #arrangement of phase and neurons per event per trial instead of per trial per event
    
    instant_phase_per_event_per_trial = transpose_to_event_per_trial(instant_phase_per_trial_per_event, len(event_name))
    Neurons_per_event_per_trial = transpose_to_event_per_trial(Neurons_per_trial_per_event, len(event_name))
   
    print("LFP and Neurons OK")

    disp = event_phase_disposition(Neurons_per_event_per_trial, instant_phase_per_event_per_trial)
    
    print("phase disposition done")
    #kappa array, bias correction is automatic when nb_spikes<16 
    kap = kappa_array(disp, min_spike = min_spike_per_event)
    
    if kap_threshold_computation:
        
        len_interval = border_length(disp, min_spike_per_event)
        list_sample_size = list(np.arange(len_interval[0],len_interval[1]+1,1))
        #threshold_list = function_threshold_sample_size(list_sample_size, 200, alpha)
        #fitted_threshold = fitting_function_threshold_sample_size(threshold_list, list_sample_size)
        fitted_threshold = threshold_theory_index(list_sample_size, alpha = alpha)
        freq_select = frequency_selection_sample_threshold(fitted_threshold, kap, disp, conv_cluster_freq, len_interval[0])
        
    else :
        
        CI = CI_array(disp, min_spike = min_spike_per_event, alpha = 2*alpha) #2*alpha cause we compare only the lower bound that is defined by alpha/2
        freq_select = frequency_selection_bootstrap(CI, boot_kap_threshold, conv_cluster_freq)

    #print_each_event(freq_select,event_name)
    
    return disp, kap, freq_select, conv_cluster_freq

def main_MI_analysis (sig, event_name, time_list, sampling_rate = 20000, t_start = 0., t_stop = 9.0, f_start = 1.,
         f_stop = 50., frequency_bin = 0.5, threshold = 70, nb_bin = 20, permut_rep = 1000, alpha = 0.05):
    #--------------------------------------------- THAT'S IT-------------- -------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------  
    
    #Morlet wavelet analysis
    complex_map, map_times, freqs, tfr_sampling_rate = compute_timefreq(sig, sampling_rate, f_start, f_stop, delta_freq=frequency_bin, nb_freq=None,
                        f0=2.5,  normalisation = 0., t_start=t_start)
    print("scalo computation complete")
    #amplitude and phase map
    ampl_map = np.abs(complex_map)
    phase_map = np.angle(complex_map) 
       
    #ridge extraction over the threshold 
    ridge_map_plot, ridge_map_comp = ridge_map(ampl_map, threshold=threshold)
    
    
    delta_freq = frequency_bin
    extent = (map_times[0], map_times[-1], freqs[0]-delta_freq/2., freqs[-1]+delta_freq/2.)
    plt.imshow(ampl_map.transpose(), interpolation='nearest', origin ='lower', aspect = 'auto', extent = extent, cmap='hot')
    
    
    print("ridge done")
    #selection, grouping and conversion of frequency filtered signal with amplitude over the threshold    
    interest_freq = filtering_empty(ridge_map_comp)
    cluster_freq = cluster_freq_building(interest_freq)
    cluster_freq = conversion_interval(cluster_freq)
    conv_cluster_freq = conversion_frequency(cluster_freq, frequency_bin, f_start)
    print("cluster freq done", conv_cluster_freq)
    #extraction and arrangement of the phase for each frequency group
    
    instant_phase_raw = extraction_instant_phase(cluster_freq, phase_map, frequency_bin, sig, f_start, f_stop, sampling_rate)
    ampl_envelope_raw = extraction_ampl_envelope(cluster_freq, ampl_map, frequency_bin, sig, f_start, f_stop, sampling_rate)
    
    #cutting in trials
    
    split_in = int((len(sig)/sampling_rate)/t_stop)
    
    instant_phase_per_trial = []
    ampl_per_trial = []
    for freq in range(len(instant_phase_raw)):
        phase_trial_cut = trial_cutting_LFP(instant_phase_raw[freq], split_in, tfr_sampling_rate, t_stop)
        ampl_trial_cut = trial_cutting_LFP(ampl_envelope_raw[freq], split_in, tfr_sampling_rate, t_stop)
        instant_phase_per_trial.append(arrangement_time_array(phase_trial_cut,tfr_sampling_rate))
        ampl_per_trial.append(arrangement_time_array(ampl_trial_cut,tfr_sampling_rate))
    print("phase computation complete")

    
    #cutting the signal_per_trial and the train_per_trial by each event
    instant_phase_per_trial_per_event = LFP_array_cutting(instant_phase_per_trial, time_list)
    ampl_per_trial_per_event = LFP_array_cutting(ampl_per_trial, time_list)
    print("cutting by event")
    
    #removal of the time arrangement. We could avoid this, we just need to readapt LFP_array_cutting so that it compute time with sampling rate
    for freq in range(len(instant_phase_per_trial_per_event)):
        for trial in range(len(instant_phase_per_trial_per_event[freq])):
            for event in range(len(instant_phase_per_trial_per_event[freq][trial])):
                for i in range(len(instant_phase_per_trial_per_event[freq][trial][event])):
                    instant_phase_per_trial_per_event[freq][trial][event][i] = instant_phase_per_trial_per_event[freq][trial][event][i][1]
                    ampl_per_trial_per_event[freq][trial][event][i] = ampl_per_trial_per_event[freq][trial][event][i][1]
    
    #concatenation between trial
    analysis_phase = []
    analysis_ampl = []
    for freq in range(len(cluster_freq)):
        analysis_phase.append([])
        analysis_ampl.append([])
        for event in range(len(time_list)):
            analysis_phase[freq].append([])
            analysis_ampl[freq].append([])
            for trial in range(split_in):
                analysis_phase[freq][event] += instant_phase_per_trial_per_event[freq][trial][event]
                analysis_ampl[freq][event] += ampl_per_trial_per_event[freq][trial][event]
    
    #histogramm construction
    hist = histogram_array(analysis_phase, analysis_ampl, nb_bin = nb_bin)
    
    #MI computation
    MI = MI_array(hist)
    print("MI computation complete")
    
    #permutation test
    sig_array = signification_array(analysis_phase, analysis_ampl, MI, nb_bin = nb_bin, rep = permut_rep, alpha = alpha)
    print("signification computed")
    
    return analysis_phase, analysis_ampl, MI_array, sig_array, conv_cluster_freq




