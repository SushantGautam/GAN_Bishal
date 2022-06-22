import librosa
import numpy as np
import os
from ourLTFATStft import LTFATStft
import ltfatpy
import imageio
import math
from ADC_Sampling import ADC_Sampling
fft_hop_size = 64
fft_window_length = 256
clipBelow = -10
anStftWrapper = LTFATStft()
sampling_fre = 8000
imagesize = 64
trunk_step = 26
data_root = "./timit_8k_8bit/TIMIT/TEST/"
bit_num = 16
count = 0
phase = "test"
status = "noise"
if status == "noise":
    if_noise = True
else:
    if_noise = False
n_mels= 64
target_SNR = 50
outroot = "./timit_mel_{}_{}k_{}_{}bit_random_white".format(status,sampling_fre//1000,phase,bit_num)
mel_matrix = librosa.filters.mel(sr=sampling_fre,n_fft=fft_window_length,n_mels=n_mels)
if not os.path.exists(outroot):
    os.mkdir(outroot)
for root,dirs,files in os.walk(data_root):
    for name in files:
        if os.path.splitext(name)[1]==".wav":
            audio,sr = librosa.core.load(os.path.join(root,name),sr=sampling_fre,mono=False)
            count +=1
            audio = audio-np.mean(audio)
            if if_noise == True:
                temp_SNR = np.random.randint(10,target_SNR)
                audio = ADC_Sampling(audio,temp_SNR)
                name = name + '_{}db_{}'.format(temp_SNR, 'white')
            audio = audio/np.max(np.abs(audio.flatten()))
            audio = audio.astype(np.float64)
            real_DGT = anStftWrapper.oneSidedStft(signal=audio,windowLength=fft_window_length,hopSize=fft_hop_size)
            mag = np.abs(real_DGT)
            mag = np.dot(mel_matrix,mag)
            mag = mag/np.max(mag.flatten())
            mag = np.log(np.clip(mag,a_min=np.exp(clipBelow),a_max=None))
            mag = mag/(-1*clipBelow)+1
            if phase == "train":
                for i in range((mag.shape[1]-imagesize)//trunk_step):
                    slice_mag = mag[:,i*trunk_step:(i+1)*trunk_step+imagesize]
                    slice_mag_ = np.round(slice_mag[0:n_mels,:]*(2**(bit_num)-1))
                    root_ = root[16:].replace("/","_")
                    filename = os.path.join(outroot,root_+"_"+name+str(i)+".png")
                    if bit_num == 16:
                        imageio.imwrite(filename,slice_mag_.astype(np.uint16))
                    elif bit_num == 8:
                        imageio.imwrite(filename,slice_mag_.astype(np.uint8))
                    else:
                        raise NotImplementedError
            elif phase == "test":
                for i in range(math.ceil(mag.shape[1]/imagesize)):
                    if (i+1)*imagesize<= mag.shape[1]:
                        slice_mag = mag[:,i*imagesize:(i+1)*imagesize]
                    else:
                        slice_mag = mag[:,i*imagesize:]
                        slice_mag = np.pad(slice_mag,(0,imagesize-slice_mag.shape[1]),'constant')
                    slice_mag_ = np.round(slice_mag[0:n_mels,:]*(2**(bit_num)-1))
                    root_ = root[16:].replace("/","_")
                    filename = os.path.join(outroot,root_+"_"+name+str(i)+".png")
                    if bit_num == 16:
                        imageio.imwrite(filename,slice_mag_.astype(np.uint16))
                    elif bit_num == 8:
                        imageio.imwrite(filename,slice_mag_.astype(np.uint8))
                    else:
                        raise NotImplementedError
