import mne 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
from pathlib import Path
path = "/work/sanis/data/EPHYS/data/"
file_names = list(Path(path).rglob("*.wav"))
def gen_figures(file_name, overlap=3, winlen=10, freq_min=15, 
                freq_max=30, wsize=2500, th_perc=99.5):
    sfreq, data = wavfile.read(file_name)
    time = np.arange(len(data))/sfreq
    dt = wsize/sfreq/2

    tf = np.abs(mne.time_frequency.stft(data, wsize=wsize).squeeze())
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=sfreq)

    stft_df = pd.DataFrame(data=tf[::-1], 
                        index=pd.Series(np.round(freqs[::-1]/1000, 1), name="Frequencies (kHz)"),
                        columns=np.arange(0, dt*tf.shape[1], dt))

    plt_dat = stft_df.loc[(stft_df.index > freq_min) & (stft_df.index < freq_max),:]
    v_max= np.percentile(plt_dat, th_perc)

    nb_samples= None
    for start_time in np.arange(0,stft_df.columns[-1],winlen):
        end_time = start_time+winlen+overlap
        plt_dat = stft_df.loc[(stft_df.index > freq_min) & (stft_df.index < freq_max), 
                                    (stft_df.columns > start_time) & (stft_df.columns < end_time)]
        if nb_samples is None:
            nb_samples=plt_dat.shape[1] #x-axis(columns-index)
        elif nb_samples != plt_dat.shape[1]:
            tmp_dat=np.zeros([plt_dat.shape[0],nb_samples]) #y-axis(row-index)
            tmp_dat[:,:plt_dat.shape[1]] = plt_dat.values
            plt_dat = tmp_dat
        plt.imsave(fname=f"usv_image_E_CUE/{Path(file_name).with_suffix('').name}_{start_time}_{end_time}.png", 
                   arr=plt_dat, cmap="gray_r", vmax= v_max)
        print(plt_dat.shape)

for file_name in file_names:
    gen_figures(file_name)
print('Done')
