# python 3.6 required

sudo apt install libfftw3-dev
sudo apt install liblapack-dev
# pip install ltfatpy, librosa 
sudo apt install python3-ltfatpy
sudo apt install --no-install-recommends libdpkg-perl cmake gcc g++ make libfftw3-dev liblapack-dev cython3 python3 python3-dev python3-setuptools python3-pip python3-wheel python3-tk python3-matplotlib python3-scipy
pip3 install --no-deps ltfatpy


 cp -R ./dataset/timit_8k_mel/train_A ./dataset/timit_8k_mel/train_B
 cp -R ./dataset/timit_8k_mel/test_A ./dataset/timit_8k_mel/test_B