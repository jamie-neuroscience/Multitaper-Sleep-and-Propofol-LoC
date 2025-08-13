import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os


all_zscored_spectrograms = []
all_db_spectrograms = []


# Load in the data
case_id = 0
ana_file_path = r'GABA_Data/Volunteer/' # Cases 2 to 15


# Provide a list of all the propofol cases
volunteer_cases = ['02', '03', '04', '05', '07', '08', '09', '10', '13', '15']
test_volunteer_cases = ['02']

#Define the frequency Bands
freq_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Low Beta': (13, 20),
    'High Beta': (20, 30),
    'Gamma': (30, 49.5)
}

#Go through the anaesthesia cases --> everything is within this loop for the anaesthesia
for case_id in volunteer_cases:

    # Set up the path files
    spectrogram_path = f"{case_id}_Sdb.csv"
    freq_path = f"{case_id}_f.csv"
    time_path = f"{case_id}_t.csv"
    consciousness_path = f"{case_id}_l.csv"

    spectrogram_file = (ana_file_path + spectrogram_path)
    freq_file = (ana_file_path + freq_path)
    time_file = (ana_file_path + time_path)
    consciousness_file = (ana_file_path + consciousness_path)

    # Load the files
    case_spectrogram = np.loadtxt(spectrogram_file, delimiter=',')  # Shape of 100 rows for the 100 bins, and then 5000+ columns for each epoch
    number_of_epochs = case_spectrogram.shape[1]
    case_frequencies = np.loadtxt(freq_file, delimiter=',')  # Shape of 100 rows for each frequency bin
    case_times = np.loadtxt(time_file, delimiter=",")  # Shape of 5000+ rows for the time at each epoch
    case_consciousness = np.loadtxt(consciousness_file, delimiter=',')  # Shape of 5000+ rows for consciousness at each epoch

    # Find the transition in consciousness

    # We add one here because it returns the first value in the pair of the transition, whereas we want the second value
    LoC_index = np.where((case_consciousness[:-1] == 1) & (case_consciousness[1:] == 0))[0] + 1
    LoC_time = case_times[(LoC_index)]
    last_conscious_index = LoC_index[0] - 1


    print(f"The {case_id} patient goes unconscious at the index: " + str(LoC_index))
    print(f"The {case_id} patient goes unconscious at the time: " + str(LoC_time))

    if len(LoC_index) == 0:
        raise ValueError("No loss of consciousness found")

    # Take 120 epochs (4 minutes) either side of the LoC
    LoC_index = int(LoC_index[0])
    before_LoC = LoC_index - 120
    after_LoC = LoC_index + 120

    # Take 120 epochs from around the LoC
    if LoC_index >= 120 and LoC_index + 120 < number_of_epochs:
        epoch_index_around_LoC = [before_LoC, after_LoC]
    else:
        print(f"warning: not enough epochs around loss of consciousness")

    print(f"The epochs around the LoC for {case_id} are: {epoch_index_around_LoC}")

    aligned = case_spectrogram[:, before_LoC:after_LoC]
    baseline = case_spectrogram[:, 0:last_conscious_index]

    #Now we calculate the baseline metrics
    baseline_mean = baseline.mean(axis=1, keepdims=True)
    baseline_std = baseline.std(axis=1, keepdims=True)

    # Now we do the normalisation
    zscored = (aligned - baseline_mean)/baseline_std

    # We add the zscored spectrograms to the list
    all_zscored_spectrograms.append(zscored)


    # Let's try it with decibels instead
    aligned = case_spectrogram[:, before_LoC:after_LoC]  # Your period of interest
    baseline = case_spectrogram[:, 0:last_conscious_index]  # Your baseline period

    # Compute the baseline mean for each frequency
    baseline_mean_db = np.mean(baseline, axis=1, keepdims=True)

    # Compute baseline-normalised power in decibels
    db_normalised = aligned - baseline_mean_db

    # Store for group-level median later
    all_db_spectrograms.append(db_normalised)

    print("  ")

    output_dir = 'Propofol_Spectrograms_db'
    os.makedirs(output_dir, exist_ok=True)

    """# Save Z-scored spectrogram
    zscored_filename = os.path.join(output_dir, f"Propofol_{case_id}_zscored_spectrogram.npy")
    np.save(zscored_filename, zscored)

    # Save dB-normalised spectrogram
    db_filename = os.path.join(output_dir, f"Propofol_{case_id}_db_spectrogram.npy")
    np.save(db_filename, db_normalised)"""






#Editing the Anaesthesia Data with z scored
all_ana_zscored_spectrograms = np.stack (all_zscored_spectrograms, axis=0)
ana_group_median_spectrogram = np.median(all_ana_zscored_spectrograms, axis=0)  # Resulting shape: (freqs, epochs)
q25 = np.percentile(all_ana_zscored_spectrograms, 25, axis=0)
q75 = np.percentile(all_ana_zscored_spectrograms, 75, axis=0)


"""# Save group-level Z-scored median
group_zscored_filename = os.path.join(output_dir, "propofol_group_median_zscored_spectrogram.npy")
np.save(group_zscored_filename, ana_group_median_spectrogram)"""


plt.figure(figsize=(12, 6))
im = plt.imshow(ana_group_median_spectrogram, aspect='auto', origin='lower',
                extent=[-240, 240, 0, 50],  # X: time in seconds (if 2s/epoch), Y: 0–50 Hz
                cmap='jet', vmin=-3, vmax=3, interpolation='bilinear')

plt.axvline(0, color='k', linestyle='--', label='LoC')
plt.colorbar(im, label='Z-score Power')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Propofol Group-Level Spectrogram (Median across Participants)')
plt.legend()
plt.tight_layout()
plt.show()



# Stack and compute median across participants with db
all_ana_db_spectrograms = np.stack(all_db_spectrograms, axis=0)
ana_group_median_spectrogram = np.median(all_ana_db_spectrograms, axis=0)
q25 = np.percentile(all_ana_db_spectrograms, 25, axis=0)
q75 = np.percentile(all_ana_db_spectrograms, 75, axis=0)

"""# Save group-level dB median
group_db_filename = os.path.join(output_dir, "propofol_group_median_db_spectrogram.npy")
np.save(group_db_filename, ana_group_median_spectrogram)"""

#Save time and frequencies

np.save(os.path.join(output_dir, "frequencies.npy"), case_frequencies)
times = np.arange(-120, 120) * 2  # gives [-240, -238, ..., +236, +238]
np.save(os.path.join(output_dir, "correct_times.npy"), times)
print(times)


plt.figure(figsize=(12, 6))
im = plt.imshow(ana_group_median_spectrogram, aspect='auto', origin='lower',
                extent=[times[0], times[-1], 0, 50],
                cmap='jet', vmin=-10, vmax=10, interpolation='bilinear')  # adjust dB scale

plt.axvline(0, color='k', linestyle='--', label='LoC')
plt.colorbar(im, label='Power Change (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.xticks(np.arange(-240, 241, 30))  # from -240 to 240 in steps of 30 seconds
plt.title('Propofol Group-Level Spectrogram (Baseline Normalised in dB, Median)')
plt.legend()
plt.tight_layout()
plt.savefig('propofol_group_spectrogram.png', dpi=300)  # save as PNG with 300 dpi resolution
plt.show()

print(np.min(aligned), np.min(baseline_mean))



