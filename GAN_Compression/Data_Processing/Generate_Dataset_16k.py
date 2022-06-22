import librosa
import numpy as np
import os
from ourLTFATStft import LTFATStft
import ltfatpy
import imageio
import math
fft_hop_size = 128
fft_window_length = 512
clipBelow = -10
anStftWrapper = LTFATStft()
sampling_fre = 16000
imagesize = 128
trunk_step = 16
data_root = "./data/TEST/"
bit_num = 16
count = 0
phase = "test"
n_mels=128
outroot = "./timit_mel_{}k_{}_{}bit".format(sampling_fre//1000,phase,bit_num)
mel_matrix = librosa.filters.mel(sr=sampling_fre,n_fft=fft_window_length,n_mels=n_mels)
if not os.path.exists(outroot):
    os.mkdir(outroot)
for root,dirs,files in os.walk(data_root):
    for name in files:
        if os.path.splitext(name)[1]==".wav":
            audio,sr = librosa.core.load(os.path.join(root,name),sr=sampling_fre,mono=False)
            count +=1
            audio = audio-np.mean(audio)
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
                    root_ = root[14:].replace("/","_")
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
                    root_ = root[14:].replace("/","_")
                    filename = os.path.join(outroot,root_+"_"+name+str(i)+".png")
                    if bit_num == 16:
                        imageio.imwrite(filename,slice_mag_.astype(np.uint16))
                    elif bit_num == 8:
                        imageio.imwrite(filename,slice_mag_.astype(np.uint8))
                    else:
                        raise NotImplementedError