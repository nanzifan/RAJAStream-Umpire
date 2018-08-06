
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>

#define VERSION_STRING "3.3"

#include "Stream.h"

#if defined(CUDA)
#include "CUDAStream.h"
#elif defined(HIP)
#include "HIPStream.h"
#elif defined(HC)
#include "HCStream.h"
#elif defined(OCL)
#include "OCLStream.h"
#elif defined(USE_RAJA)
#include "RAJAStream.hpp"
#elif defined(KOKKOS)
#include "KokkosStream.hpp"
#elif defined(ACC)
#include "ACCStream.h"
#elif defined(SYCL)
#include "SYCLStream.h"
#elif defined(OMP)
#include "OMPStream.h"
#endif

// Default size of 2^13
unsigned int ARRAY_SIZE = 16;
unsigned int num_times = 100;
unsigned int deviceIndex = 0;
bool use_float = false;
bool triad_only = false;
bool output_as_csv = false;
std::string csv_separator = ",";



int main(int argc, char *argv[])
{

  RAJAStream *r;
  r = new RAJAStream(ARRAY_SIZE, deviceIndex);
  r->init_arrays(1.0,2.0,0);
  r->copy();
}

