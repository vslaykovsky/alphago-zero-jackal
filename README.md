
## dependencies

### install nvidia + cuda + cudnn
// linux disable secure boot, disable fast boot

sudo service gdm stop
echo 'blacklist nouveau' > /etc/modprobe.d/blacklist-nvidia-nouveau.conf
echo 'options nouveau modeset=0' >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf

sudo apt-get install gcc make
sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb

// install repository

sudo apt-get install cuda
sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb

### libraries

```
cd ~/Downloads
mkdir libtorch-cxx11-abi-shared-with-deps-1.8.1+cu111
unzip libtorch-cxx11-abi-shared-with-deps-1.8.1+cu111.zip
mv libtorch libtorch-cxx11-abi-shared-with-deps-1.8.1+cu111 

sudo apt-get install libopencv-dev libprotobuf-dev nlohmann-json3-dev libgtest-dev libboost-dev protobuf-compiler
```

### build tensorboard_logger
cd third_party/tb_logger && make 


### remote install

apt-get install cmake vim wget unzip
wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip
unzip libtorch*.zip
apt-get install libopencv-dev libprotobuf-dev nlohmann-json3-dev libgtest-dev libboost-dev protobuf-compiler


# article

- intro: jackal game 
- existing algos: alphago, alphago-zero, mu-zero
- encoding: cnn vs lstm, arrows encoding, integer encoding, state/action encoding. Loss function
- performance: multithreading c++ vs python, bottlenecks of state copy/encoding, queue based model execution,
- training: replay buffer, model degrades, state function only?
- determinization: closed tiles



# TODO
- test state value of synthetic game states