# Evaluation Pt 2: Precision, Recall Curves

Gaining a better understanding of how well our network is performing

Code to go along with blog: [Evaluation Pt 2: Precision, Recall Curves](http://www.curiousinspiration.com/posts/writing-the-code-for-precision-and-recall)

# Build

`mkdir build`

`cd build`

`cmake -D BLAS_INCLUDE_DIR=/usr/local/opt/openblas/include \
       -D BLAS_LIB_DIR=/usr/local/opt/openblas/lib \
       -D GLOG_INCLUDE_DIR=~/Code/3rdParty/glog-0.3.5/glog-install/include/ \
       -D GLOG_LIB_DIR=~/Code/3rdParty/glog-0.3.5/glog-install/lib/ \
       -D GTEST_INCLUDE_DIR=~/Code/3rdParty/googletest-release-1.8.0/install/include/ \
       -D GTEST_LIB_DIR=~/Code/3rdParty/googletest-release-1.8.0/install/lib/ ..`

`make`

`./tests`

`GLOG_logtostderr=1 ./feedforward_neural_net`


# Dependencies

## Create 3rdParty dir

`mkdir -p ~/Code/3rdParty`

## Install GLOG

`wget https://github.com/google/glog/archive/v0.3.5.tar.gz -O glog-v0.3.5.tar.gz`

`tar xzvf glog-v0.3.5.tar.gz`

`cd glog-0.3.5/`

`mkdir glog-install`

`./configure --prefix=~/Code/3rdParty/glog-0.3.5/glog-install`

`make -j4 && make install`

## Install GTest

`wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz -O gtest-1.8.0.tar.gz`

`tar xzvf gtest-1.8.0.tar.gz`

`cd googletest-release-1.8.0/`

`mkdir build`

`mkdir install`

`cd build`

`cmake -D CMAKE_INSTALL_PREFIX=~/Code/3rdParty/googletest-release-1.8.0/install/ ..`

`make -j4`

`make install`

## Install OpenMP

`brew install libomp`

