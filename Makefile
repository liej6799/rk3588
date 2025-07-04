# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++11 -I../rknn-header/ -Iinclude/
LDFLAGS := -L../rknn-header/ -lrknnrt

# Source files
SRC_DIR := src
INCLUDE_DIR := include
CNPY_SRC := $(INCLUDE_DIR)/cnpy/cnpy.cpp

# Targets
BIN1 := rknn_create_mem_demo
BIN2 := rknn_api_test_custom_cpu_op

BIN1_SRC := $(SRC_DIR)/rknn_create_mem_demo.cpp
BIN2_SRC := $(SRC_DIR)/rknn_api_test_custom_cpu_op.cpp $(CNPY_SRC)

# Install paths
INSTALL_DIR := install/rknn_api_demo_Linux
MODEL_DIR := model
MODEL_INSTALL_DIR := $(INSTALL_DIR)/model
LIB_INSTALL_DIR := $(INSTALL_DIR)/lib

# Model and image files
MODEL_FILES := $(wildcard $(MODEL_DIR)/*)
IMAGE_FILES := $(wildcard $(MODEL_DIR)/*.jpg)

.PHONY: all clean install

all: $(BIN1) $(BIN2)

$(BIN1): $(BIN1_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN2): $(BIN2_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

install: all
	mkdir -p $(INSTALL_DIR)
	mkdir -p $(MODEL_INSTALL_DIR)
	mkdir -p $(LIB_INSTALL_DIR)
	cp $(BIN1) $(INSTALL_DIR)/
	cp $(BIN2) $(INSTALL_DIR)/
	cp -r $(MODEL_DIR) $(INSTALL_DIR)/
	cp $(IMAGE_FILES) $(MODEL_INSTALL_DIR) 2>/dev/null || true
	cp ../rknn-header/librknnrt.so $(LIB_INSTALL_DIR)/
	@if [ -n "$(RGA_LIB)" ]; then cp $(RGA_LIB) $(LIB_INSTALL_DIR)/; fi

clean:
	rm -f $(BIN1) $(BIN2)
	rm -rf $(INSTALL_DIR) 