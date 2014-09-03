#!/bin/sh
nvcc -arch sm_13 -o main NN.cu -I /usr/local/cuda/include -I ../nvcc_test/inc/ -L /usr/local/cuda/lib -lcudart -lcublas -lcusparse
