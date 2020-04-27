# Spike_Sorting
Repository to collect and organise the pipeline of spike sorting analyses.

## 1. Tridesclous (by Sam Garcia)
   - upload all the recordings of the same condition as individual segments
   - Use **common_ref_removal** to de-noise the recordings.
     This will compute the median signal among all the sites uploaded and remove it from each trace.
   - You can use two types of configuration for the probe (requires different .prb files):
        **Sparse**: all sites are listed in one single channel groups (preferred method)
        **Dense**: it respects the proper subdivision in channel groups. Each channel group is analysed separately     
   - A nice way of cleaning the clusters and of detecting good spikes from the trash is to use **LDA**.
     (select clusters --> right click --> feature projection with selection --> LDA)
     This analysis will almost alway nicely cluster and separate all the clusters, so it's not used to confirm clusters.
     It is used to detect spikes from the trash. You select one of the clusters on which you just did LDA and select also 
     the trash. From here, if the cluster is well isolated from the trash, it is possible to detect spikes that should 
     belong to taht cluster but are still in the trash.
        
        
## 2. Extract spike times and waveforms (Script done)
   - It extracts the spike times and waveforms of each cluster and saves them in 2 excel files.
      
## 3. Auto- and Cross-correlogram (Script in progress)
   - Computes the auto or cross-correlogram of the chosen clusters
      
## 4. Create database (Script to be done/ Doable by hand)
   - For each animal, one excel file.
     Each sheet of the file will correspond to a condition. In each sheet there will be:
        - Spike times
        - Waveform
        - Auto- and cross-correlogram 
        - Raster plots of the spike times
        
## 5. Analyse the database (Script to be done)
   - Changes in mean firing rate depending on the behavioural task or on the light stimulation
   - Covariance of the firing rate between clusters
      
## 6. Phase-Locking analysis (Script done, by Lilian)
   - Computes the phase-locking of each cluster's firing rate with the LFP
      
      
