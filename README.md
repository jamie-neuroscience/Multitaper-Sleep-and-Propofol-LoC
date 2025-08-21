# Multitaper-Sleep-and-Propofol-LoC
This is the code and processed data for the paper "Features of loss of consciousness: a multitaper spectral analysis of frontal cortical EEG during sleep and propofol" in review.
This contains the code for the processing of the propofol data, sleep data, and the calculation of metrics, as well as the baseline-normalised spectrograms.

------------------------ FILE EXPLANATIONS ------------------------ 

-------- CODE --------

**propofolfiles.py** - the script for baseline-normalising the propofol data and computing the group-level spectrograms

**sleeppipeline.py** - the script for processing the sleep data (bandpass filter, epoching, linear detrend, multitaper spectra), baseline-normalisation, and computing the group-level spectrograms

**metricanalysis.py** - the script for the computation of all metrics used (total and mean power, spectral slope, peak frequency, spectral parameters


-------- DATA --------

**Sleep_Spectrograms_db** - the baseline-normalised spectrograms for all participants in the sleep group, in decibels

**Sleep_Spectrograms_linear** - the baseline-normalised spectrograms for all participants in the sleep group, in raw power (µV)

**Sleep_Spectrograms_zscore** - the baseline-normalised spectrograms for all participants in the sleep group, as a z-score

**Sleep_Spectrograms_group** - the group level spectrograms (median) in decibels, raw power, and z-scored

**Propofol_Spectrograms_db** - the baseline-normalised spectrograms for all participants in the propofol group, in decibels

**Propofol_Spectrograms_linear** - the baseline-normalised spectrograms for all participants in the propofol group, in raw power (µV)

**Propofol_Spectrograms_zscore** - the baseline-normalised spectrograms for all participants in the propofol group, as a z-score

**Propofol_Spectrograms_group** - the group level spectrograms (median) for the propofol group in decibels, raw power, and z-scored


