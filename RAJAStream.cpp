
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "RAJAStream.hpp"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

// #ifndef ARRAY_SIZE

RAJAStream::RAJAStream(const unsigned int ARRAY_SIZE)
    : array_size(ARRAY_SIZE)
{

  std::cout << "memory allocation\n"; 
  a = (double*)malloc(sizeof(double) * ARRAY_SIZE);
  b = (double*)malloc(sizeof(double) * ARRAY_SIZE);
  c = (double*)malloc(sizeof(double) * ARRAY_SIZE);
  cudaMalloc((void**)&d_a, sizeof(double)*ARRAY_SIZE);
  cudaMalloc((void**)&d_b, sizeof(double)*ARRAY_SIZE);
  cudaMalloc((void**)&d_c, sizeof(double)*ARRAY_SIZE);

}


RAJAStream::~RAJAStream()
{
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}


void RAJAStream::init_arrays(double initA, double initB, double initC)
{
  std::cout << "init" << std::endl;
  // // double* RAJA_RESTRICT a = d_a;
  // // double* RAJA_RESTRICT b = d_b;
  // // double* RAJA_RESTRICT c = d_c;
  // forall<policy>(RangeSegment(0, array_size), [=] RAJA_DEVICE (RAJA::Index_type index)
  // {
  //   d_a[index] = initA;
  //   d_b[index] = initB;
  //   d_c[index] = initC;
  // });

  for (int i=0; i<array_size; i++)
  {
    a[i] = initA;
    // std::cout << "a allocate\n";
    b[i] = initB;
    c[i] = initC;
  }

  std::cout << "host init finish" << std::endl;
  for (int i=0; i<array_size; i++)
  {
    std::cout << "a[i] " << a[i] << " b[i] " << b[i] << " c[i] " << c[i] << std::endl;
  }

  cudaMemcpy(d_a, a, sizeof(double)*array_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(double)*array_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, sizeof(double)*array_size, cudaMemcpyHostToDevice);

  std::cout << "end init" << std::endl;

  double* tmp1 = (double*)malloc(sizeof(double) * array_size);
  double* tmp2 = (double*)malloc(sizeof(double) * array_size);
  double* tmp3 = (double*)malloc(sizeof(double) * array_size);
  cudaMemcpy(tmp1, d_a, sizeof(double)*array_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(tmp2, d_b, sizeof(double)*array_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(tmp3, d_c, sizeof(double)*array_size, cudaMemcpyDeviceToHost);

  std::cout << "test device init " << std::endl;
  for (int i=0; i<array_size; i++)
  {
    std::cout << "a[i] " << tmp1[i] << " b[i] " << tmp2[i] << " c[i] " << tmp3[i] << std::endl;
  }
}


// template <typename double>
// __global__ void copy_kernel(const double * a, double * c)
// {
//   const int i = blockDim.x * blockIdx.x + threadIdx.x;
//   c[i] = a[i];
// }


void RAJAStream::copy()
{
  std::cout << "copy" << std::endl;
  std::cout << "array_size is " << array_size << std::endl;

  double* tmp1 = (double*)malloc(sizeof(double) * array_size);
  double* tmp2 = (double*)malloc(sizeof(double) * array_size);
  double* tmp3 = (double*)malloc(sizeof(double) * array_size);
  cudaMemcpy(tmp1, d_a, sizeof(double)*array_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(tmp2, d_b, sizeof(double)*array_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(tmp3, d_c, sizeof(double)*array_size, cudaMemcpyDeviceToHost);

  std::cout << "test device init " << std::endl;
  for (int i=0; i<array_size; i++)
  {
    std::cout << "a[i] " << tmp1[i] << " b[i] " << tmp2[i] << " c[i] " << tmp3[i] << std::endl;
  }

  double* RAJA_RESTRICT da = d_a;
  double* RAJA_RESTRICT dc = d_c;

  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, array_size), 
    [=] RAJA_DEVICE (int i) {
    printf("inside copy, i is%d\n", i);
    printf("d_a[i] is %d\n", da[i]);
    dc[i] = da[i];
  });
}
