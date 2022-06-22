import librosa
import numpy as np
import os
from ourLTFATStft import LTFATStft
import ltfatpy
import imageio
import math
from ADC_Sampling import ADC_Sampling
import numpy as np
import copy
fft_hop_size = 64
fft_window_length = 256
clipBelow = -10
anStftWrapper = LTFATStft()
sampling_fre = 8000
imagesize = 64
trunk_step = 26
data_root = "./timit_8k_8bit/TIMIT/TEST"
bit_num = 16
count = 0
phase = "test"
status = "noise"
if status == "noise":
    if_noise = True
else:
    if_noise = False
n_mels= 64
target_SNR = 20
if_audio = True
if_ADC = False 
outroot = "./timit_mel_{}_{}k_{}_{}bit_various_noise_{}db_ADC_{}".format(status,sampling_fre//1000,phase,bit_num,target_SNR,if_ADC)
test_noise_filenames={0:"babble",1:"destroyerengine",2:"f16",3:"factory",4:"leopard", 5:"m109", 6:"machinegun",7:"pink", 8:"volvo",9:"white"}
train_noise_filenames={0:"babble",1:"destroyerengine",2:"f16",3:"factory",4:"leopard", 5:"m109", 6:"machinegun"}
noise_file_paths = "./Noise92/{}_8000.wav"
mel_matrix = librosa.filters.mel(sr=sampling_fre,n_fft=fft_window_length,n_mels=n_mels)
outroot_A = os.path.join(outroot,"A")
outroot_B = os.path.join(outroot,"B")
audio_output = os.path.join(outroot,"audio")
if not os.path.exists(outroot):
    os.mkdir(outroot)
if not os.path.exists(outroot_A):
    os.mkdir(outroot_A)
if not os.path.exists(outroot_B):
    os.mkdir(outroot_B)
if not os.path.exists(audio_output):
    os.mkdir(audio_output)
for root,dirs,files in os.walk(data_root):
    for name in files:
        if os.path.splitext(name)[1]==".wav":
            audio,sr = librosa.core.load(os.path.join(root,name),sr=sampling_fre,mono=False)
            orgin_audio = copy.deepcopy(audio)
            count +=1
            audio = audio-np.mean(audio)
            if phase == "train":
                noise_filenames = train_noise_filenames
            elif phase == "test":
                noise_filenames = test_noise_filenames
            if if_noise == True:
                noise_type = np.random.randint(0,len(noise_filenames))
                noise_filepath = noise_file_paths.format(noise_filenames[noise_type])
                SNR = np.random.randint(0,target_SNR)
                noise,sr = librosa.core.load(noise_filepath,sr=sampling_fre,mono=False)
                audio_rms = np.sqrt(np.mean(audio**2))
                noise_rms = np.sqrt(np.mean(noise**2))
                audio_gan = 10**(SNR/20)*noise_rms/audio_rms
                random_slice= np.random.randint(0,len(noise)-len(audio))
                noise_slice = noise[random_slice:random_slice+len(audio)]
                audio = audio_gan*audio + noise_slice
                root_ = root[16:].replace("/", "_")
                if if_ADC:
                    audio = ADC_Sampling(audio,100)
                if if_audio:
                    librosa.output.write_wav(os.path.join(audio_output,root_+name),y=audio,sr=sr,norm=True)
            audio = audio/np.max(np.abs(audio.flatten()))
            orgin_audio = orgin_audio/np.max(np.abs(orgin_audio.flatten()))
            audio = audio.astype(np.float64)
            orgin_audio = orgin_audio.astype(np.float64)
            real_DGT = anStftWrapper.oneSidedStft(signal=audio,windowLength=fft_window_length,hopSize=fft_hop_size)
            orgin_DGT = anStftWrapper.oneSidedStft(signal=orgin_audio,windowLength=fft_window_length,hopSize=fft_hop_size)
            mag = np.abs(real_DGT)
            mag = np.dot(mel_matrix,mag)
            mag = mag/np.max(mag.flatten())
            mag = np.log(np.clip(mag,a_min=np.exp(clipBelow),a_max=None))
            mag = mag/(-1*clipBelow)+1
            orgin_mag = np.dot(mel_matrix,np.abs(orgin_DGT))
            orgin_mag = orgin_mag/np.max(orgin_mag.flatten())
            orgin_mag = np.log(np.clip(orgin_mag,a_min=np.exp(clipBelow),a_max=None))/(-1*clipBelow)+1
            assert len(mag) == len(orgin_mag)
            if phase == "train":
                for i in range((mag.shape[1]-imagesize)//trunk_step):
                    slice_mag = mag[:,i*trunk_step:(i+1)*trunk_step+imagesize]
                    slice_mag_ = np.round(slice_mag[0:n_mels,:]*(2**(bit_num)-1))
                    orgin_slice_mag = orgin_mag[:,i*trunk_step:(i+1)*trunk_step+imagesize]
                    orgin_slice_mag_ = np.round(orgin_slice_mag[0:n_mels,:]*(2**(bit_num)-1))
                    root_ = root[16:].replace("/","_")
                    filename_A = os.path.join(outroot_A,root_+"_"+name+str(i)+noise_filenames[noise_type]+str(SNR)+"dB.png")
                    filename_B = os.path.join(outroot_B,root_+"_"+name+str(i)+noise_filenames[noise_type]+str(SNR)+"dB.png")
                    if bit_num == 16:
                        imageio.imwrite(filename_A,slice_mag_.astype(np.uint16))
                        imageio.imwrite(filename_B,orgin_slice_mag_.astype(np.uint16))
                    elif bit_num == 8:
                        imageio.imwrite(filename_A,slice_mag_.astype(np.uint8))
                        imageio.imwrite(filename_B,orgin_slice_mag_.astype(np.uint8))
                    else:
                        raise NotImplementedError
            elif phase == "test":
                for i in range(math.ceil(mag.shape[1]/imagesize)):
                    if (i+1)*imagesize<= mag.shape[1]:
                        slice_mag = mag[:,i*imagesize:(i+1)*imagesize]
                        orgin_slice_mag =orgin_mag[:,i*imagesize:(i+1)*imagesize]
                    else:
                        slice_mag = mag[:,i*imagesize:]
                        slice_mag = np.pad(slice_mag,(0,imagesize-slice_mag.shape[1]),'constant')
                        orgin_slice_mag = orgin_mag[:,i*imagesize:]
                        orgin_slice_mag = np.pad(orgin_slice_mag,(0,imagesize-orgin_slice_mag.shape[1]),'constant')
                    slice_mag_ = np.round(slice_mag[0:n_mels,:]*(2**(bit_num)-1))
                    orgin_slice_mag_ = np.round(orgin_slice_mag[0:n_mels,:]*(2**(bit_num)-1))
                    root_ = root[16:].replace("/","_")
                    filename_A = os.path.join(outroot_A,root_+"_"+name+str(i)+noise_filenames[noise_type]+str(SNR)+"dB.png")
                    filename_B = os.path.join(outroot_B,root_+"_"+name+str(i)+noise_filenames[noise_type]+str(SNR)+"dB.png")
                    if bit_num == 16:
                        imageio.imwrite(filename_A,slice_mag_.astype(np.uint16))
                        imageio.imwrite(filename_B,orgin_slice_mag_.astype(np.uint16))
                    elif bit_num == 8:
                        imageio.imwrite(filename_A,slice_mag_.astype(np.uint8))
                        imageio.imwrite(filename_B, orgin_slice_mag_.astype(np.uint8))
                    else:
                        raise NotImplementedError
