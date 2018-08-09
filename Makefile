CXX = nvcc
CXXFLAGS = --expt-extended-lambda -O3 -std=c++11 -Xcompiler -fopenmp

UMPIRE_LIB_1 = $(UMPIRE_PATH)/lib/libumpire.a
UMPIRE_LIB_2 = $(UMPIRE_PATH)/lib/libumpire_resource.a
UMPIRE_LIB_3 = $(UMPIRE_PATH)/lib/libumpire_strategy.a
UMPIRE_LIB_4 = $(UMPIRE_PATH)/lib/libumpire_op.a
UMPIRE_LIB_5 = $(UMPIRE_PATH)/lib/libumpire_util.a
UMPIRE_LIB_6 = $(UMPIRE_PATH)/lib/libumpire_tpl_judy.a

CUDA_LIB = /usr/local/cuda-9.2/lib64/libcudart_static.a
SYS_LIB = /usr/lib/x86_64-linux-gnu/librt.so

LIB = $(UMPIRE_LIB_1) $(UMPIRE_LIB_2) $(UMPIRE_LIB_3) $(UMPIRE_LIB_1) $(UMPIRE_LIB_2) $(UMPIRE_LIB_3) $(UMPIRE_LIB_4) $(CUDA_LIB) -ldl $(SYS_LIB) $(UMPIRE_LIB_5) $(UMPIRE_LIB_6) 

raja-stream: main.cu RAJAStream.cu
	$(CXX) $(CXXFLAGS) -DUSE_RAJA -I$(RAJA_PATH)/include -I$(UMPIRE_PATH)/include $^ -o $@ -L$(RAJA_PATH)/lib -lRAJA -L$(UMPIRE_PATH)/lib $(LIB) 

raja-device: main.cu RAJAStream_device.cu
	$(CXX) $(CXXFLAGS) -DUSE_RAJA -I$(RAJA_PATH)/include -I$(UMPIRE_PATH)/include $^ -o $@ -L$(RAJA_PATH)/lib -lRAJA -L$(UMPIRE_PATH)/lib $(LIB) 


.PHONY: clean
clean:
	rm -f raja-stream

