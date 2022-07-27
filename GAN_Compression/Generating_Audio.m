clear all
clc
addpath("C:\Program Files\MATLAB\R2018a\toolbox\phaseret", "C:\Program Files\MATLAB\R2018a\toolbox\ltfat")
ltfatstart();
phaseretstart;
disp("setting up is done.");
output_root_path = '.\test_audio';
output_file_name = "test_out.wav";
image_path = '..\test_out_denoise\';
sampling_ratio = 8000;
if ~exist(output_root_path,'dir')
    mkdir(output_root_path);
end
seachstring = strcat("*syn.png");
candidate_list_unsort = dir(fullfile(image_path,seachstring));
[~,index] = natsortfiles({candidate_list_unsort.name});
candidate_list = candidate_list_unsort(index);
Audio = cell(length(candidate_list),1);
for k = 1:length(candidate_list)
    signal= double(imread(fullfile(candidate_list(k).folder,candidate_list(k).name)))/65535;
    if (sampling_ratio == 8000)
        audio = Recover_Audio(signal,64,256,4);
    elseif (sampling_ratio == 16000)
        audio = Recover_Audio(signal,128,512,4);
    else
        print("NO such sampling ratio");
    end
    Audio{k} = audio;
end
Audio_all = [];
for index =1:length(Audio)
    Audio_temp = Audio{index};
    Audio{index} = Audio_temp;
    Audio_all=[Audio_all;Audio_temp];
end

Audio_all_m = Audio_all;
if (sampling_ratio ==8000)
    audiowrite(fullfile(output_root_path,output_file_name),Audio_all_m,8000);
elseif (sampling_ratio == 16000)
    audiowrite(fullfile(output_root_path,output_file_name),Audio_all_m,16000);
else
    print("No such sampling ratio");
end

