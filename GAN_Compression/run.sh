#!/usr/bin/env bash
MODE=$1
FILE=$2
OUTFILE=$3
if [ $MODE == "16k_2kbps_clean" ]
then
  python Com_Single_Audio_Clean.py --name "16k_2kbps_clean" --loadSize 140 --fineSize 128 --dataroot ./dataset/single_audio_test --n_downsample_global 4 --n_blocks_global 6 --C_channel 8 --n_cluster 16 --OneDConv True --OneDConv_size 63 --ngf 64 --max_ngf 512 --sampling_ratio 16000 --n_fft 512 --n_mels 128 --Conv_type C --input_file $FILE --output_path $OUTFILE
elif [ $MODE == "16k_1kbps_clean" ]
then
  python Com_Single_Audio_Clean.py --name "16k_1kbps_clean" --loadSize 140 --fineSize 128 --dataroot ./dataset/single_audio_test --n_downsample_global 4 --n_blocks_global 6 --C_channel 4 --n_cluster 16 --OneDConv True --OneDConv_size 63 --ngf 64 --max_ngf 512 --sampling_ratio 16000 --n_fft 512 --n_mels 128 --Conv_type C --input_file $FILE --output_path $OUTFILE
elif [ $MODE == "8k_2kbps_clean" ]
then
  python Com_Single_Audio_Clean.py --name "8k_2kbps_clean" --loadSize 90 --fineSize 64 --dataroot ./dataset/single_audio_test --n_downsample_global 3 --n_blocks_global 3 --C_channel 4 --n_cluster 16  --ngf 64 --max_ngf 256 --sampling_ratio 8000 --n_fft 256 --n_mels 64 --Conv_type C --input_file $FILE --output_path $OUTFILE
elif [ $MODE == "8k_1kbps_clean" ]
then
  python Com_Single_Audio_Clean.py --name "8k_1kbps_clean" --loadSize 90 --fineSize 64 --dataroot ./dataset/single_audio_test --n_downsample_global 3 --n_blocks_global 3 --C_channel 2 --n_cluster 16 - --ngf 64 --max_ngf 256 --sampling_ratio 8000 --n_fft 256 --n_mels 64 --Conv_type C --input_file $FILE --output_path $OUTFILE
elif [ $MODE == "8k_0p5kbps_clean" ]
then
  python Com_Single_Audio_Clean.py --name "8k_0p5kbps_clean" --loadSize 90 --fineSize 64 --dataroot ./dataset/single_audio_test --n_downsample_global 3 --n_blocks_global 3 --C_channel 1 --n_cluster 16 --ngf 64 --max_ngf 256 --sampling_ratio 8000 --n_fft 256 --n_mels 64 --Conv_type C --input_file $FILE --output_path $OUTFILE
elif [ $MODE == "8k_2kbps_pink" ]
then
  python Com_Single_Audio_Noise.py --name "8k_2kbps_pink" --loadSize 90 --fineSize 64 --dataroot ./dataset/single_audio_test --n_downsample_global 3 --n_blocks_global 3 --C_channel 4 --n_cluster 16 --ngf 64 --max_ngf 256  --sampling_ratio 8000 --n_fft 256 --n_mels 64 --Conv_type C --input_file $FILE --output_path $OUTFILE
else
  echo "No such mode"
fi