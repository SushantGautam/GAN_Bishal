# GAN_Compression
The code of [**"Unified Signal Compression Using Generative Adversarial Networks"**](https://arxiv.org/abs/1912.03734), 45th International Conference on Acoustics, Speech, and Signal Processing, ICASSP 2020. The code is for audio compression part, but can be used for image compression by small modifications. 

### Abstract
We propose a unified compression framework that uses generative adversarial networks (GAN) to compress image and speech signals. The compressed signal is represented by a latent vector fed into a generator network which is trained to produce high quality signals that minimize a target objective function. To efficiently quantize the compressed signal, non-uniformly quantized optimal latent vectors are identified by iterative back-propagation with ADMM optimization performed for each iteration. Our experiments show that the proposed algorithm outperforms prior signal compression methods for both image and speech compression quantified in various metrics including bit rate, PSNR, and neural network based signal classification accuracy 

### Citation
--------

If you find this code useful, please consider citing:

```text
@inproceedings{liu2020unified,
  title={Unified Signal Compression Using Generative Adversarial Networks},
  author={Liu, Bowen and Cao, Ang and Kim, Hun-Seok},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3177--3181},
  year={2020},
  organization={IEEE}
}
```
### Code 
To run audio compression, we need run two stage scripts.

The first stage is to get the compressed vector and decompressed spectrums by python code.

The second stage is to transform the spectrums into audios by Matlab code.

## First Stage (python code)
### Installation 
####1. Install Pytorch and Torchvision [website](https://pytorch.org/)
####2. Install requirement package libfftw3-dev, liblapack-dev, ltfatpy, librosa
For Linux, you can use the command:

sudo apt install libfftw3-dev

sudo apt install liblapack-dev

pip install ltfatpy, librosa 

### Prepare for the Dataset
The first thing is to make dataset for training and evaluating. In our experiments, we mainly use TIMIT dataset and Librispeech dataset. You could also use dataset you want, especially for evaluation data.

The basic data preparing pipline is: we calculate the mel-sepctrums of each audio and get a 2D tensor with shape: n_mel_bins x time_frames. Then we separate the whole 2-D tensor into small tensors whose shapes could be fed into GAN networks. We have two phases including "train" and "test".

For "train" phase, we separate the whole mel-spectrums with overlap. For example, for a 128*200 mel-spectrum, if we set overlap_step is 22, the input to GAN is 128x128, then this mel-spectrum would be seperated into 5 sub mel-spectrums (there would be 22 frames shift for two small mel-sepctrums).we pad 0 for the last mel-spectrums

For "test" phase, these sub-melspectrums would not have overlaps.

You need to generate 4 folders, named "train_A", "train_B", "test_A", "test_B".

"train_A" is the input for training.

"train_B" is the target output for training.

"test_A" is the input for test.

"test_B" is the target for test.

For clean audio compression task, "train_A" is the same with "train_B", "test_A" is the same as "test_B".
#### 1. Download
Download dataset into ./Data_Processing folder. The dataset is expected to have ".wav" or ".flac" audio file.

#### 2. Change configuration
You may want to edit Generate_Dataset_16k.py or Generate_Dataset_8k.py if you want to compress audios with 16k or 8k Hz sampling ratio.

In these files, you need to edit the "data_root" to your audio file root; edit the "outroot" to your target output root.

Change the phase to "train" or "test" to get preprocessed dataset and rename them to  "train_A","train_B","test_A","test_B".

#### 3. Make Dataset folder
Under the folder of project, make folder:

mkdir /dataset/customer_dataset_name

Then put "train_A","train_B","test_A","test_B" under this folder.

### Edit the configure file

Go to options folder and edit the configure file. 

In the "base_options.py" file, you may want to edit this file for the name, NN structure, data path and other settings.
You may want to edit them according to the explanation in the options files. 

Please note that to control the bitrate, you may want to edit the "C_channel" argument.
The target bitrate calculation equation is: bitrate = (finesize/(2^n_downsample_global))^2\*C_channel\*bit.
For example, in 16kHz audio compression, the finesize is 128, n_downsample_global is 4, then the bitrate is 8\*8\*C_channel*bit.
The typical bit is 4.


We provide two typical options for 16kHz audio with 2kbps and 8kHz audio with 2kbps for easy beginning. 
 --name "16k_2kbps_clean" --loadSize 140 --fineSize 128 --dataroot ./dataset/timit_16k_2kbps --n_downsample_global 4 --n_blocks_global 6 --C_channel 8 --n_cluster 16 --OneDConv True --OneDConv_size 63 --ngf 64 --max_ngf 512 --sampling_ratio 16000 --n_fft 512 --n_mels 128 --Conv_type C
 
 --name "16k_1kbps_clean" --loadSize 140 --fineSize 128 --dataroot ./dataset/single_audio_test --n_downsample_global 4 --n_blocks_global 6 --C_channel 4 --n_cluster 16 --OneDConv True --OneDConv_size 63 --ngf 64 --max_ngf 512 --sampling_ratio 16000 --n_fft 512 --n_mels 128 --Conv_type C 
 
 In the "train_Q_options", you may want to adjust the niter and niter_decay to change changing epochs.
 
 ### Training 
 After all the settings done, you could start to train by typing:
 python train_Q.py
 
Our training contains three phases: 
First, training as a normal GAN with floating point latent vector.
Second, training with a soft-quantizer: the inputs to the decoder are soft-quantized.
Third, training with a hard-quantizer: the inputs to the decoder are hard-quantized.  



 ### Testing
 There are several ways for testing.
 
 "python test.py" will output decompressed results of mel-spectrums, and you could evaluate the difference between original and decompressed mel-spectrums. However, the mel-spectrums could be used to transfer back to audios. We need to calculate the spectrums.
 
 "python Compression_ADMM.py" will output decompressed results of spectrums with ADMM. You may want to download the output folders into Matlab folder for further evaluation.
 
 "python audio_test.py" will output decompressed results of spectrums without ADMM. You may want to download the output folders into Matlab folder for further evaluation.
 
 You may want to change the "output_path" in these files to specific output folder.
 
## Second Stage (Matlab code)

If you would like to listen to the the decompressed audios, you may need to run the matlab code.
We get spectrums by python code and would utilize a maltab time-frequency toolbox to transfer them back to audios.

### Installation
We need to install LTFAT (The Large Time-Frequency Analysis Toolbox) [website](http://ltfat.github.io/doc/demos/demo_frsynabs.html) and Phase ReTrieval for time-frequency representations toolbox [website](https://github.com/ltfat/phaseret)

You could install them according to the instructions in website. Or you could directly include our provided folder.(**It could only work on Windows**)

### Inference

Please remember to add these  packages into path 

Put the OUTPUTFOLDER (generated by python code) in the Matlab path and 

####1. Inference Single audio
In the "Generating_Audio.m" file, change the variable settings, including:

output_root_path: the output path to store the audio file

output_file_name: the name of the output audio file

image_path: the name of the OUTPUTFOLDER

sampling_ratio: 8000 or 16000

Then you could run the "Generating_Audio.m" file.
####2. Inference TIMIT dataset
"TIMIT_Generation.m" is for 16kHz TIMIT audio reconstruction
"TIMIT_Generation_8k.m" is for 8kHz TIMIT audio reconstruction.
In the file, change the "output_root_path" to the location where you store the output spectrums.
change the "timit_root_path" to TIMIT audio file path, (we only use the original audio to calculate the power of audios).

####3. Audio Recovery Function
You may want to write your own scripts for audio recovery.
You could utilize the "Recover_Audio.m" function, whose input: "a" refers to fft_hop_size, "M" refers to fft_window_length, tfr=M/a
