#!/bin/bash -e
clang sniff.cc -ldl -shared -fPIC -o ../rknn-demo/sniff.so
