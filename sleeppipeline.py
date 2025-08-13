import matplotlib.pyplot as plt
import numpy as np
import nitime.algorithms as tsa
from scipy.signal import detrend
import os
# ignore FutureWarning errors which clutter the outputs
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nitime.algorithms as alg
import nitime.utils as utils
import nitime.algorithms.spectral as tsa
import nitime as nit
from concurrent import futures
from itertools import islice
from scipy import signal, io, fftpack, stats
import scipy as sp
import numpy as np
import warnings
from scipy.signal import detrend
import numpy as np
import matplotlib.pyplot as plt


#Sanity Checks
def sanity_check_signal(signal, label="Signal"):
    """Print basic stats and plot signal snippet."""
    print(f"\n[{label}] mean: {np.mean(signal):.4f}, std: {np.std(signal):.4f}, min: {np.min(signal):.4f}, max: {np.max(signal):.4f}")
    if np.isnan(signal).any():
        print(f"[WARNING] NaNs detected in {label}")
    """plt.figure(figsize=(12, 2))
    plt.plot(signal[:5000])  # plot first 5k samples as a snippet
    plt.title(f"{label} snippet")
    plt.show()"""

def sanity_check_spectrogram(spectrogram, freqs, label="Spectrogram"):
    """Check shape, power range, and frequency bins."""
    total_power = np.sum(spectrogram)
    print(f"\n[{label}] shape: {spectrogram.shape}, total power: {total_power:.4f}")
    if np.isnan(spectrogram).any():
        print(f"[WARNING] NaNs detected in {label}")
    print(f"[{label}] Freq range: {freqs[0]:.2f} Hz - {freqs[-1]:.2f} Hz")
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, np.mean(spectrogram, axis=1))
    plt.title(f"Mean spectrum: {label}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean Power")
    plt.show()

def check_epoch_power(spectrogram_matrix, stage_label="Stage"):
    """Check total power per epoch for drops."""
    total_power_per_epoch = np.sum(spectrogram_matrix, axis=0)
    print(f"\n[{stage_label}] Total power per epoch: mean={np.mean(total_power_per_epoch):.8f}, std={np.std(total_power_per_epoch):.8f}")
    plt.figure(figsize=(10, 4))
    plt.plot(total_power_per_epoch, marker='o')
    plt.title(f"Total power per epoch: {stage_label}")
    plt.xlabel("Epoch")
    plt.ylabel("Total Power")
    plt.show()
    return total_power_per_epoch

import mne
from mne.datasets.sleep_physionet.age import fetch_data

from sleepanalysis import butterworth_filter, multitaper, bin_frequencies

subjects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
test_subjects = [41]
good_subjects = [0, 5, 6, 12, 16, 18, 21, 23, 33, 40]
#18 and 21

"""0 is good, 1,2,3,4 aren't great and 4 has interference at 50Hz, 
5 is relatively good, 6 is pretty good, 7 has the interference again, 
x8 is pretty good and interesting with band in middle range, but they wake back up
9 is bad, 10 has interference at 50Hz
11 has the pattern with the alpha but inteference at 50Hz and two epochs
12 is pretty good
x13 doesn't have the distinct pattern but does see a decrease in overall power it seems - they wake back up
x14 is very good, decrease alpha pattern and increase in lower frequencies - wakes back up
x15 is not bad again, includes the alpha pattern it seems
16 good but interesting, overall increase in power  in alpha band it seems --> not great?
x17 isn't bad either
18 has huge power before and large decrease after, pretty good?
19 not bad but sems some epochs are very contaminated?
20 seems ok, it has large decrease in the higher power ranges, and seems to have some other decreases in the alpha range
21 has a lot of action at the start and then there is a general increase in the lower power frequencies
22 has a lot of interference it seems, but overall pattern represented
23 good
x24 very nice data but they wake back up
25 seems to have a lot of interference at the start
26 is terrible 
27 terrible
x28 awake the whole time
x29 really good data but wakes back up!
30 a lot of interference
31 super big peaks (meant to be there)
32 pretty ok
33 better than ok
x34 wakes back up
x35 wakes back up
no recording for 36
x37 wakes back up
x38 wakes back up
39 doesn't exist
40 pretty good
41 pretty good with pattern

"""

all_zscored_spectrograms = []
all_db_spectrograms = []
all_linear_spectrograms = []

for subject in good_subjects:
    [subject_files] = fetch_data(subjects=[subject], recording=[1])

    subject_raw = mne.io.read_raw_edf(
        subject_files[0],
        stim_channel="Event marker",
        infer_types=True,
        preload=True,
        verbose="error",  # ignore issues with stored filter settings
    )
    subject_annot = mne.read_annotations(subject_files[1])
    sfreq = subject_raw.info['sfreq']
    print("Setting annotations to raw")
    subject_raw.set_annotations(subject_annot, emit_warning=False)

    annotation_desc_2_event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3": 4,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
    }



    """#Checking the full spectrogram
    full_spectrogram_raw = subject_raw.copy()
    full_spectrogram_annot = subject_annot.copy()



    full_spectrogram_annot.crop(full_spectrogram_annot[1]["onset"] - 30 * 60, full_spectrogram_annot[-2]["onset"] + 30 * 60)
    full_spectrogram_raw.set_annotations(full_spectrogram_annot, emit_warning=False)

    full_spectrogram_events, _ = mne.events_from_annotations(
        full_spectrogram_raw, event_id=annotation_desc_2_event_id, chunk_duration=2.0
    )

    event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3/4": 4,
        "Sleep stage R": 5,
    }



    fig = mne.viz.plot_events(
        full_spectrogram_events,
        event_id=event_id,
        sfreq=full_spectrogram_raw.info['sfreq'],
        first_samp=full_spectrogram_raw.first_samp,
    )
    plt.show()


    full_spectrogram_array = full_spectrogram_raw.pick_channels(['Fpz-Cz'])
    # Get the data as a numpy array (shape: [channels x samples])
    full_spectrogram_array, times = full_spectrogram_array.get_data(return_times=True)
    # fpz_cz_data shape will be (1, n_samples), so squeeze to 1D
    full_spectrogram_array = full_spectrogram_array.squeeze()


    # Apply the bandpass filter

    # parameters
    lowcut = 0.5
    highcut = 49.5

    filtered_full_subject = butterworth_filter(full_spectrogram_array, sfreq, lowcut, highcut)

    # Example: you have one channel, e.g., 'Fpz-Cz'
    channel_names = ['Fpz-Cz']
    sfreq = subject_raw.info['sfreq']  # reuse same sfreq
    channel_types = ['eeg']  # or whatever it is
    # Create new info with just this channel
    new_info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

    # Your filtered signal must be (n_channels, n_samples)
    filtered_full_subject = filtered_full_subject[np.newaxis, :]

    filtered_full_subject = mne.io.RawArray(filtered_full_subject, new_info)




    # Create 2-second epochs of the whole thing
    tmax = 2 - (1 / sfreq)  # tmax in included

    full_spectrogram_epochs = mne.Epochs(
        raw=filtered_full_subject,
        events=full_spectrogram_events,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
    )


    print("CHECK HERE")
    print(full_spectrogram_epochs)
    print("Shape:", full_spectrogram_epochs.get_data().shape)

    full_spectrogram_epochs = full_spectrogram_epochs.get_data()
    print(f"BASELINE SUBJECT EPOCHS SHAPE {full_spectrogram_epochs.shape}")
    full_spectrogram_n_epochs = full_spectrogram_epochs.shape[0]

    window_size_sec = 2
    step_size_sec = 2  # no overlap

    winsize = int(window_size_sec * sfreq)
    winstep = int(step_size_sec * sfreq)

    full_spectrogram_list = []

    # Compute the multitaper spectra
    for i in range(full_spectrogram_n_epochs):
        signal = full_spectrogram_epochs[i, 0, :]
        full_mtsg, full_t, full_freqs = multitaper(signal, movingwin=[winsize, winstep], NW=3, hz=sfreq)

        full_mean_power = np.mean(full_mtsg, axis=1)
        full_spectrogram_list.append(full_mean_power)

        # Example inside your multitaper loop:
        sanity_check_spectrogram(baseline_mtsg, baseline_freqs, label=f"Baseline epoch {i + 1}")
        

    full_spectrogram_matrix = np.column_stack(full_spectrogram_list)

    print(f"Spectrogram matrix shape: {full_spectrogram_matrix.shape}")
    print(full_spectrogram_n_epochs)

    full_binned_spectrogram = bin_frequencies(full_spectrogram_matrix, full_freqs, n_bins=100, fmin=0.0,
                                                  fmax=50)
    print(full_binned_spectrogram.shape)

    full_times = np.arange(full_binned_spectrogram.shape[1]) * 2  # seconds
    full_freqs_binned = np.linspace(0, 50, 100)



    plt.figure(figsize=(12, 6))
    plt.imshow(
        full_binned_spectrogram,
        aspect='auto',
        origin='lower',
        extent=[full_times[0], full_times[-1], full_freqs_binned[0], full_freqs_binned[-1]],
        cmap='jet'  # <-- classic blue-to-red colormap
    )
    plt.colorbar(label='Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Multitaper Spectrogram (full spectrogram, Jet Colormap) for subject {subject}')
    plt.tight_layout()
    plt.show()

    check_epoch_power(full_spectrogram_matrix, stage_label="Full spectrograms")"""








    # BASELINE ANALYSIS
    baseline_subject_annot = subject_annot.copy()
    baseline_subject_raw = subject_raw.copy()

    for annot in baseline_subject_annot:
        if annot['description'] == 'Sleep stage 1':
            stage1_onset = annot['onset']
            break  # stop at the first occurrence

    baseline_subject_annot.crop(tmin=0, tmax=1000)
    print("setting annotations to baseline")
    baseline_subject_raw.set_annotations(baseline_subject_annot, emit_warning=False)



    baseline_subject_events, _ = mne.events_from_annotations(
        baseline_subject_raw, event_id=annotation_desc_2_event_id, chunk_duration=2.0
    )


    event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3/4": 4,
        "Sleep stage R": 5,
    }

    baseline_existing_event_ids = np.unique(baseline_subject_events[:, 2])
    baseline_filtered_event_id = {k: v for k, v in event_id.items() if v in baseline_existing_event_ids}

    fig = mne.viz.plot_events(
        baseline_subject_events,
        event_id=baseline_filtered_event_id,
        sfreq=baseline_subject_raw.info['sfreq'],
        first_samp=baseline_subject_raw.first_samp,
    )
    plt.show()

    # Get only the channel you want
    baseline_subject_raw_array = baseline_subject_raw.pick_channels(['Fpz-Cz'])
    # Get the data as a numpy array (shape: [channels x samples])
    baseline_subject_raw_array, times = baseline_subject_raw_array.get_data(return_times=True)

    # fpz_cz_data shape will be (1, n_samples), so squeeze to 1D
    baseline_subject_raw_array = baseline_subject_raw_array.squeeze()

    # Apply the bandpass filter

    # parameters
    lowcut = 0.5
    highcut = 49.5

    baseline_filtered_subject = butterworth_filter(baseline_subject_raw_array, sfreq, lowcut, highcut)

    channel_names = ['Fpz-Cz']
    sfreq = subject_raw.info['sfreq']
    channel_types = ['eeg']
    new_info = mne.create_info(ch_names=channel_names,sfreq=sfreq, ch_types=channel_types)

    baseline_filtered_subject = baseline_filtered_subject[np.newaxis, : ]
    baseline_filtered_subject = mne.io.RawArray(baseline_filtered_subject, new_info)



    #Sanity Check
    """sanity_check_signal(baseline_subject_raw_array, label="Raw EEG before filter")
    sanity_check_signal(baseline_filtered_subject, label="Raw EEG after filter")"""


    # Create 2-second epochs of the whole thing
    tmax = 2 - (1 / sfreq)  # tmax in included

    baseline_subject_epochs = mne.Epochs(
        raw=baseline_filtered_subject,
        events=baseline_subject_events,
        event_id=baseline_filtered_event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
    )
    del baseline_filtered_subject


    print("CHECK HERE")
    print(baseline_subject_epochs)
    print("Shape:", baseline_subject_epochs.get_data().shape)


    baseline_subject_epochs = baseline_subject_epochs.get_data()
    print(f"BASELINE SUBJECT EPOCHS SHAPE {baseline_subject_epochs.shape}")
    baseline_n_epochs = baseline_subject_epochs.shape[0]

    window_size_sec = 2
    step_size_sec = 2  # no overlap

    winsize = int(window_size_sec * sfreq)
    winstep = int(step_size_sec * sfreq)

    baseline_spectrogram_list = []

    # Compute the multitaper spectra
    for i in range(baseline_n_epochs):
        signal = baseline_subject_epochs[i, 0, :]

        #Linear detrend
        signal = detrend(signal, type='linear')

        baseline_mtsg, baseline_t, baseline_freqs = multitaper(signal, movingwin=[winsize, winstep], NW=3, hz=sfreq)

        baseline_mean_power = np.mean(baseline_mtsg, axis=1)
        baseline_spectrogram_list.append(baseline_mean_power)

        """# Example inside your multitaper loop:
        sanity_check_spectrogram(baseline_mtsg, baseline_freqs, label=f"Baseline epoch {i + 1}")
"""
    baseline_spectrogram_matrix = np.column_stack(baseline_spectrogram_list)

    print(f"Spectrogram matrix shape: {baseline_spectrogram_matrix.shape}")
    print(baseline_n_epochs)

    baseline_binned_spectrogram = bin_frequencies(baseline_spectrogram_matrix, baseline_freqs, n_bins=100, fmin=0.0, fmax=50)
    print(baseline_binned_spectrogram.shape)

    baseline_times = np.arange(baseline_binned_spectrogram.shape[1]) * 2  # seconds
    baseline_freqs_binned = np.linspace(0, 50, 100)

    # Log-transform
    baseline_binned_spectrogram_db = 10 * np.log10(baseline_binned_spectrogram + 1e-15)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        baseline_binned_spectrogram_db,
        aspect='auto',
        origin='lower',
        extent=[baseline_times[0], baseline_times[-1], baseline_freqs_binned[0], baseline_freqs_binned[-1]],
        cmap='jet'  # <-- classic blue-to-red colormap
    )
    plt.colorbar(label='dB Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Multitaper Baseline Spectrogram (dB Power, Jet Colormap) for subject {subject}')
    plt.tight_layout()
    plt.show()



#LOC analysis

    # Step 2: Define the window (4 min before and after)
    LoC_start_time = stage1_onset - 4 * 60  # 4 minutes before
    LoC_end_time = stage1_onset + 4 * 60  # 4 minutes after

    subject_annot.crop(tmin=LoC_start_time, tmax=LoC_end_time)
    subject_raw.set_annotations(subject_annot, emit_warning=False)

    subject_events, _ = mne.events_from_annotations(
        subject_raw, event_id=annotation_desc_2_event_id, chunk_duration=2.0
    )

    event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3/4": 4,
        "Sleep stage R": 5,
    }


    existing_event_ids = np.unique(subject_events[:, 2])
    filtered_event_id = {k: v for k, v in event_id.items() if v in existing_event_ids}

    # Get only the channel you want
    subject_raw_array = subject_raw.pick_channels(['Fpz-Cz'])
    # Get the data as a numpy array (shape: [channels x samples])
    subject_raw_array, times = subject_raw_array.get_data(return_times=True)

    # fpz_cz_data shape will be (1, n_samples), so squeeze to 1D
    subject_raw_array = subject_raw_array.squeeze()

    # plot events
    fig = mne.viz.plot_events(
        subject_events,
        event_id=filtered_event_id,
        sfreq=subject_raw.info["sfreq"],
        first_samp=subject_events[0, 0],
    )

    # Apply the bandpass filter

    # parameters
    lowcut = 0.5
    highcut = 49.5

    filtered_subject = butterworth_filter(subject_raw_array, sfreq, lowcut, highcut)

    channel_names = ['Fpz-Cz']
    sfreq = subject_raw.info['sfreq']
    channel_types = ['eeg']
    new_info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

    filtered_subject = filtered_subject[np.newaxis, :]
    filtered_subject = mne.io.RawArray(filtered_subject, new_info)


    # Create 2-second epochs of the whole thing
    tmax = 2 - (1 / sfreq)   # tmax in included

    subject_epochs = mne.Epochs(
        raw=filtered_subject,
        events=subject_events,
        event_id=filtered_event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
    )
    del subject_raw

    subject_epochs = subject_epochs.get_data()
    print(f"SUBJECT EPOCHS SHAPE {subject_epochs.shape}")
    n_epochs = subject_epochs.shape[0]

    window_size_sec = 2
    step_size_sec = 2  # no overlap

    winsize = int(window_size_sec * sfreq)
    winstep = int(step_size_sec * sfreq)

    spectrogram_list = []

    # Compute the multitaper spectra
    for i in range(n_epochs):
        signal = subject_epochs[i, 0, :]

        #Linear detrend
        signal = detrend(signal, type='linear')

        mtsg, t, freqs = multitaper(signal, movingwin=[winsize, winstep], NW=3, hz=sfreq)

        mean_power = np.mean(mtsg, axis=1)
        spectrogram_list.append(mean_power)

    spectrogram_matrix = np.column_stack(spectrogram_list)

    print(f"Spectrogram matrix shape: {spectrogram_matrix.shape}")
    print(n_epochs)

    """#Sanity Check
    check_epoch_power(baseline_spectrogram_matrix, stage_label="Baseline")
    check_epoch_power(spectrogram_matrix, stage_label="LoC")"""

    binned_spectrogram = bin_frequencies(spectrogram_matrix, freqs, n_bins=100, fmin=0.0,
                                                  fmax=50)
    print(binned_spectrogram.shape)

    times = np.arange(binned_spectrogram.shape[1]) * 2  # seconds
    freqs_binned = np.linspace(0, 50, 100)

    # dB-transform safely
    binned_spectrogram_db = 10 * np.log10(binned_spectrogram + 1e-15)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        binned_spectrogram_db,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], freqs_binned[0], freqs_binned[-1]],
        cmap='jet'  # <-- classic blue-to-red colormap
    )
    plt.colorbar(label='dB Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Multitaper Spectrogram Around the LoC (dB Power, Jet Colormap) for subject {subject}')
    plt.tight_layout()
    plt.show()


    """ #Sanity Check
    baseline_mean_spectrum = np.mean(baseline_spectrogram_matrix, axis=1)
    loc_mean_spectrum = np.mean(spectrogram_matrix, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(baseline_freqs, baseline_mean_spectrum, label="Baseline")
    plt.plot(freqs, loc_mean_spectrum, label="LoC")
    plt.title(f"Mean Spectrum Comparison for subject {subject}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.show()"""

    baseline_total = np.sum(baseline_spectrogram_matrix)
    loc_total = np.sum(spectrogram_matrix)
    power_ratio = loc_total / baseline_total

    print(f"Baseline total power: {baseline_total:.8f}")
    print(f"LoC total power: {loc_total:.8f}")
    print(f"Power ratio (LoC/Baseline): {power_ratio:.3f}")

    if power_ratio < 0.2:
        print("[WARNING] Large drop in total power! Check filtering or noise.")


    #Z-SCORE NORMALISATION
    baseline = baseline_binned_spectrogram_db
    print (baseline.shape)
    aligned = binned_spectrogram_db
    print (aligned.shape)

    #Now we do baseline metrics
    baseline_mean = baseline.mean(axis=1, keepdims=True)
    baseline_std = baseline.std(axis=1, keepdims=True)

    # Now we do the normalisation
    zscored = (aligned - baseline_mean) / baseline_std

    # We add the zscored spectrograms to the list
    all_zscored_spectrograms.append(zscored)


    #LINEAR NORMALISAATION

    baseline_linear = baseline_binned_spectrogram
    aligned_linear = binned_spectrogram

    baseline_mean_linear = np.mean(baseline_linear, axis=1, keepdims=True)
    linear_normalised = aligned_linear - baseline_mean_linear
    all_linear_spectrograms.append(linear_normalised)

    check_epoch_power(linear_normalised, stage_label="Patient linear normalised")


    # DECIBEL NORMALISATION
    aligned = binned_spectrogram_db  # Your period of interest
    baseline = baseline_binned_spectrogram_db  # Your baseline period

    # Compute the baseline mean for each frequency
    baseline_mean_db = np.mean(baseline, axis=1, keepdims=True)

    # Compute baseline-normalised power in decibels
    db_normalised = aligned - baseline_mean_db
    check_epoch_power(db_normalised, stage_label="Patient db normalised")

    # Store for group-level median later
    all_db_spectrograms.append(db_normalised)

    # ---- PLOT: Linear baseline-normalised spectrogram ----
    plt.figure(figsize=(12, 5))
    plt.imshow(
        linear_normalised,
        aspect='auto',
        origin='lower',
        extent=[0, linear_normalised.shape[1], 0, linear_normalised.shape[0]],
        cmap='jet'
    )
    plt.colorbar(label='Linear Power (Baseline-Normalised)')
    plt.xlabel('Time (Epochs)')
    plt.ylabel('Frequency Bin')
    plt.title('Linear Baseline-Normalised Spectrogram')
    plt.show()

    # ---- PLOT: dB baseline-normalised spectrogram ----
    plt.figure(figsize=(12, 5))
    plt.imshow(
        db_normalised,
        aspect='auto',
        origin='lower',
        extent=[0, db_normalised.shape[1], 0, db_normalised.shape[0]],
        cmap='jet',
        vmin=-10, vmax=10  # Optional: adjust these for your data range
    )
    plt.colorbar(label='Power Change (dB, Baseline-Normalised)')
    plt.xlabel('Time (Epochs)')
    plt.ylabel('Frequency Bin')
    plt.title('dB Baseline-Normalised Spectrogram')
    plt.show()


    #SAVING THE FILES

    #Save normalised Z-score spectrogram
    zscore_output_dir = "Sleep_Spectrograms_zscore"
    os.makedirs(zscore_output_dir, exist_ok=True)
    zscore_filename = os.path.join(zscore_output_dir, f"Sleep_{subject}_zscored_spectrogram.npy")
    np.save(zscore_filename, zscored)

    # Save normalised dB spectrogram
    db_output_dir = "Sleep_Spectrograms_db"
    os.makedirs(db_output_dir, exist_ok=True)
    db_filename = os.path.join(db_output_dir, f"Sleep_{subject}_db_spectrogram.npy")
    np.save(db_filename, db_normalised)

    linear_output_dir = "Sleep_Spectrograms_linear"
    os.makedirs(linear_output_dir, exist_ok=True)
    linear_filename = os.path.join(linear_output_dir, f"Sleep_{subject}_linear_spectrogram.npy")
    np.save(linear_filename, linear_normalised)





#Creating group level metrics
print("Z scored spectrograms")
for z in all_zscored_spectrograms:

    print(f"the z-scored shape is {z.shape}")


all_sleep_zscored_spectrograms = np.stack(all_zscored_spectrograms, axis=0)
sleep_group_median_spectrogram_zscore = np.median(all_sleep_zscored_spectrograms, axis=0)
q25 = np.percentile(all_sleep_zscored_spectrograms, 25, axis=0)
q75 = np.percentile(all_sleep_zscored_spectrograms, 75, axis=0)

#Save normalised Z-score spectrogram
output_dir = "Sleep_Spectrograms_group"
os.makedirs(output_dir, exist_ok=True)
group_zscore_filename = os.path.join(output_dir, f"Sleep_group_level_zscored_spectrogram.npy")
np.save(group_zscore_filename, sleep_group_median_spectrogram_zscore)

"""plt.figure(figsize=(12, 6))
im = plt.imshow(all_sleep_zscored_spectrograms[0], aspect='auto', origin='lower',
                extent=[-240, 240, 0, 50],  # X: time in seconds (if 2s/epoch), Y: 0–50 Hz
                cmap='jet', vmin=-3, vmax=3, interpolation='bilinear')

plt.axvline(0, color='k', linestyle='--', label='LoC')
plt.colorbar(im, label='Z-score Power')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Sleep individual Spectrogram')
plt.legend()
plt.tight_layout()
plt.show()"""


print(np.isnan(sleep_group_median_spectrogram_zscore).sum())
print(np.min(sleep_group_median_spectrogram_zscore), np.max(sleep_group_median_spectrogram_zscore))

plt.figure(figsize=(12, 6))
im = plt.imshow(sleep_group_median_spectrogram_zscore, aspect='auto', origin='lower',
                extent=[-240, 240, 0, 50],  # X: time in seconds (if 2s/epoch), Y: 0–50 Hz
                cmap='RdBu_r', vmin=-3, vmax=3, interpolation='bilinear')

plt.axvline(0, color='k', linestyle='--', label='LoC')
plt.colorbar(im, label='Z-score Power')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Sleep Group-Level Spectrogram (Median across Participants)')
plt.legend()
plt.tight_layout()
plt.show()

all_db_spectrograms = np.stack(all_db_spectrograms, axis=0)
sleep_group_median_spectrogram_db = np.median(all_db_spectrograms, axis=0)
q25 = np.percentile(all_db_spectrograms, 25, axis=0)
q75 = np.percentile(all_db_spectrograms, 75, axis=0)




# Save normalised dB spectrogram
db_filename = os.path.join(output_dir, f"Sleep_group_level_db_spectrogram.npy")
np.save(db_filename, sleep_group_median_spectrogram_db)

plt.figure(figsize=(12, 6))
im = plt.imshow(sleep_group_median_spectrogram_db, aspect='auto', origin='lower',
                extent=[-240, 240, 0, 50],
                cmap='jet', vmin=-10, vmax=10, interpolation='bilinear')  # adjust dB scale

plt.axvline(0, color='k', linestyle='--', label='LoC')
plt.colorbar(im, label='Power Change (dB)')
plt.xticks(np.arange(-240, 241, 60))  # from -240 to 240 in steps of 30 seconds
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Sleep Group-Level Spectrogram (Baseline Normalised in dB, Median)')
plt.legend()
plt.tight_layout()
plt.savefig('sleep_group_spectrogram.png', dpi=300)  # save as PNG with 300 dpi resolution
plt.show()

print(np.min(aligned), np.min(baseline_mean))

"""# keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    subject_annot.crop(subject_annot[1]["onset"] - 30 * 60, subject_annot[-2]["onset"] + 30 * 60)
    subject_raw.set_annotations(subject_annot, emit_warning=False)

    events_train, _ = mne.events_from_annotations(
        subject_raw, event_id=annotation_desc_2_event_id, chunk_duration=30.0
    )

    # Get only the channel you want
    subject_raw_array = subject_raw.pick_channels(['Fpz-Cz'])
    # Get the data as a numpy array (shape: [channels x samples])
    subject_raw_array, times = subject_raw_array.get_data(return_times=True)

    # fpz_cz_data shape will be (1, n_samples), so squeeze to 1D
    subject_raw_array = subject_raw_array.squeeze()


    # Apply the bandpass filter

    #parameters
    lowcut = 0.5
    highcut = 49.5

    filtered_subject = butterworth_filter(subject_raw_array, sfreq, lowcut, highcut)
    print("BANDPASS")
    print(filtered_subject.shape)



    # Create 30-second epochs of the whole thing
    tmax = 2.0 - 1.0 / subject_raw.info["sfreq"]  # tmax in included

    subject_epochs = mne.Epochs(
        raw=subject_raw,
        events=events_train,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
    )
    del subject_raw

    subject_epochs = subject_epochs.get_data()
    print(f"SUBJECT EPOCHS SHAPE {subject_epochs.shape}")
    n_epochs = subject_epochs.shape[0]


    window_size_sec = 30
    step_size_sec = 30  # no overlap

    winsize = int(window_size_sec * sfreq)
    winstep = int(step_size_sec * sfreq)

    spectrogram_list = []

    # Compute the multitaper spectra
    for i in range(n_epochs):
        signal = subject_epochs[i, 0, : ]
        mtsg, t, freqs = multitaper(signal, movingwin=[winsize, winstep], NW=3, hz=sfreq)

        mean_power = np.mean(mtsg, axis=1)
        spectrogram_list.append(mean_power)


    spectrogram_matrix = np.column_stack(spectrogram_list)

    print(f"Spectrogram matrix shape: {spectrogram_matrix.shape}")
    print(n_epochs)

    binned_spectrogram = bin_frequencies(spectrogram_matrix, freqs, n_bins=100, fmin=0.0, fmax=50)
    print(binned_spectrogram.shape)

    times = np.arange(binned_spectrogram.shape[1]) * 30  # seconds
    freqs_binned = np.linspace(0, 50, 100)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        binned_spectrogram,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], freqs_binned[0], freqs_binned[-1]],
        cmap='viridis'
    )
    plt.colorbar(label='Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Multitaper Spectrogram')
    plt.tight_layout()
    plt.show()


    print("Spectrogram shape:", binned_spectrogram.shape)
    print("Power min:", np.min(binned_spectrogram))
    print("Power max:", np.max(binned_spectrogram))
    print("Power mean:", np.mean(binned_spectrogram))

    # Log-transform safely
    binned_spectrogram_log = np.log10(binned_spectrogram + 1e-15)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        binned_spectrogram_log,
        aspect='auto',
        origin='lower',
        extent=[times[0], times[-1], freqs_binned[0], freqs_binned[-1]],
        cmap='jet'  # <-- classic blue-to-red colormap
    )
    plt.colorbar(label='Log10 Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Multitaper Spectrogram (Log10 Power, Jet Colormap)')
    plt.tight_layout()
    plt.show()"""

#Save the linear spectrograms
all_linear_spectrograms = np.stack(all_linear_spectrograms, axis=0)
sleep_group_median_spectrogram_linear = np.median(all_linear_spectrograms, axis=0)
q25 = np.percentile(all_linear_spectrograms, 25, axis=0)
q75 = np.percentile(all_linear_spectrograms, 75, axis=0)
linear_filename = os.path.join(output_dir, f"Sleep_group_level_linear_spectrogram.npy")
np.save(linear_filename, sleep_group_median_spectrogram_linear)

check_epoch_power(sleep_group_median_spectrogram_linear, stage_label="Group Median Linear")
check_epoch_power(sleep_group_median_spectrogram_db, stage_label="Group Median db")













#this is the bandpass filter to obtain only the frequencies we want
def butterworth_filter(data, Fs, lowcut, highcut, order=5):
    """Creates a butterworth bandpass filter to get specific signals.
    Arguments:
    -----------
    data : np.ndarray
        (in form samples x channels/trials)
    Fs : float
        Sampling frequency.
    lowcut : float
        Low frequency cutoff, set to 0 for lowpass
    highcut : float
        High frequency cutoff, set to np.infty for highpass
    order : int (optional, defaults to 5)
        Filter order.
    """

    nyq = 0.5 * Fs
    low = lowcut / nyq
    high = highcut / nyq
    if low == 0:
        b, a = signal.butter(order, high, btype='low')
    elif high == np.inf:
        b, a = signal.butter(order, low, btype='high')
    else:
        b, a = signal.butter(order, [low, high], btype='band')

    # Apply butterworth filter
    data_filt = signal.filtfilt(b, a, data)

    return data_filt

def multitaper(data, movingwin, NW=3, adaptive=True, low_bias=True, jackknife=False, hz=250):
    """Multitaper spectrogram as performed by nitime. General rules of
    thumb:
    - NW is the normalized half-bandwidth of the data tapers. In MATLAB
      this is generally set to 3
    - adaptive is set to true, which means an adaptive weighting routine
      is used to combine the PSD estimates of different tapers.
    - low_bias is set to true which means Rather than use 2NW tapers,
      only use the tapers that have better than 90% spectral concentration
      within the bandwidth (still using a maximum of 2NW tapers)

    Arguments:
    -----------
    data : np.ndarray
        in form [samples] (1D)
    movingwin : [winsize, winstep]
        i.e length of moving window and step size. This is in units of
        samples (so window/Fs is units time)
    NW : (optional)
        defaults to 3. maximum number of tapers is 2NW

    Note: jackknife is not currently returned
    """
    assert len(data.shape) == 1, "Data must be 1-D."
    assert len(movingwin) == 2, "Windowing must include size, step."

    # collect params
    winsize, winstep = movingwin
    windowed_data = window(data, winsize, winstep)

    mtsg = []
    for wd in windowed_data:
        f, psd_mt, jk = tsa.multi_taper_psd(
            np.asarray(wd), Fs=hz, NW=NW,
            adaptive=adaptive, jackknife=jackknife)
        mtsg.append(psd_mt)

    mtsg = np.vstack(mtsg).T
    t = np.arange(mtsg.shape[1]) * winstep / hz
    return mtsg, t, f


def window(seq, winsize, winstep):
    """
    Returns a sliding window of width winsize and step of winstep
    from the data. Returns a list.
    """
    assert winsize >= winstep, "Window step must me at most window size."
    gen = islice(window_1(seq, n=winsize), None, None, winstep)
    for result in gen:
        yield list(result)

def window_1(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
       Window step is 1.
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result




# Example usage:
# mtsg, t, freqs = multitaper(single_epoch_data, movingwin=[3000, 3000], hz=100)
# ... do this for all epochs, stack results into mtsg_all ...
# binned_spectrogram = bin_frequencies(mtsg_all, freqs, n_bins=100, fmin=0, fmax=50)

print(np.mean(baseline_mtsg))
print(np.mean(binned_spectrogram))
print(np.mean(binned_spectrogram_db))