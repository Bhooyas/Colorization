#!/bin/bash

mkdir data
cd data
curl -o https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip
unzip coil-100.zip
cd ..
