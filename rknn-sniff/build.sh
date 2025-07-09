#!/bin/bash
LD_LIBRARY_PATH="/root/.pyenv/versions/3.11.4/lib" LD_PRELOAD="/data/Dev/rk3588/rknn-sniff/preload_python.so" clang -Xclang  -ast-dump include.cc -I../rknn-header