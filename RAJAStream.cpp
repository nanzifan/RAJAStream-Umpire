
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

using RAJA::forall;
using RAJA::RangeSegment;

// #ifndef ARRAY_SIZE
// #define ARRAY_SIZE 33554432
// #endif

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

// template <class T>
RAJAStream<double>::RAJAStream(const unsigned int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE)
{
  // RangeSegment(0, ARRAY_SIZE);
  // index_set.push_back(seg);

#ifdef RAJA_TARGET_CPU
  d_a = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*ARRAY_SIZE);
  d_b = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*ARRAY_SIZE);
  d_c = (double*)aligned_alloc(ALIGNMENT, sizeof(double)*ARRAY_SIZE);
#else
  std::cout << "memory allocation\n"; 
  a = (double*)malloc(sizeof(double) * ARRAY_SIZE);
  b = (double*)malloc(sizeof(double) * ARRAY_SIZE);
  c = (double*)malloc(sizeof(double) * ARRAY_SIZE);
  cudaMalloc((void**)&d_a, sizeof(double)*ARRAY_SIZE);
  cudaMalloc((void**)&d_b, sizeof(double)*ARRAY_SIZE);
  cudaMalloc((void**)&d_c, sizeof(double)*ARRAY_SIZE);
  cudaDeviceSynchronize();
#endif
}

// template <class T>
RAJAStream<double>::~RAJAStream()
{
#ifdef RAJA_TARGET_CPU
  free(d_a);
  free(d_b);
  free(d_c);
#else
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
#endif
}

// template <class T>
void RAJAStream<double>::init_arrays(double initA, double initB, double initC)
{
  std::cout << "init" << std::endl;
  // // double* RAJA_RESTRICT a = d_a;
  // // T* RAJA_RESTRICT b = d_b;
  // // T* RAJA_RESTRICT c = d_c;
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

// template <class double>
void RAJAStream<double>::read_arrays(
        std::vector<double>& a, std::vector<double>& b, std::vector<double>& c)
{
  std::copy(d_a, d_a + array_size, a.data());
  std::copy(d_b, d_b + array_size, b.data());
  std::copy(d_c, d_c + array_size, c.data());
}

// template <typename double>
// __global__ void copy_kernel(const T * a, T * c)
// {
//   const int i = blockDim.x * blockIdx.x + threadIdx.x;
//   c[i] = a[i];
// }

template <class double>
void RAJAStream<double>::copy()
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
  // copy_kernel<<<1, 1024>>>(d_a, d_c);
  // std::cout << "kernel functino finished " << array_size << std::endl;

  // T* RAJA_RESTRICT a = d_a;
  // T* RAJA_RESTRICT c = d_c;
  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, array_size), 
    [=] RAJA_DEVICE (int i)
  {
    // std::cout << "inside copy, i is " << i << std::endl;
    // std::cout << "d_a[i] is " << d_a[i] << std::endl;
    printf("inside copy, i is%d\n", i);
    printf("d_a[i] is %d\n", d_a[i]);
    d_c[i] += d_a[i];
  });
}

template <class T>
void RAJAStream<T>::mul()
{
  // T* RAJA_RESTRICT b = d_b;
  // T* RAJA_RESTRICT c = d_c;
  const T scalar = startScalar;
  forall<RAJA::cuda_exec<256>>(RangeSegment(0, array_size), [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    d_b[index] = scalar*d_c[index];
  });
}

template <class T>
void RAJAStream<T>::add()
{
  // T* RAJA_RESTRICT a = d_a;
  // T* RAJA_RESTRICT b = d_b;
  // T* RAJA_RESTRICT c = d_c;
  forall<RAJA::cuda_exec<256>>(RangeSegment(0, array_size), [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    d_c[index] = d_a[index] + d_b[index];
  });
}

template <class T>
void RAJAStream<T>::triad()
{
  // T* RAJA_RESTRICT a = d_a;
  // T* RAJA_RESTRICT b = d_b;
  // T* RAJA_RESTRICT c = d_c;
  const T scalar = startScalar;
  forall<RAJA::cuda_exec<256>>(RangeSegment(0, array_size), [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    d_a[index] = d_b[index] + scalar*d_c[index];
  });
}

template <class T>
T RAJAStream<T>::dot()
{
  // T* RAJA_RESTRICT a = d_a;
  // T* RAJA_RESTRICT b = d_b;

  RAJA::ReduceSum<RAJA::cuda_reduce<256>, T> sum(0.0);

  forall<RAJA::cuda_exec<256>>(RangeSegment(0, array_size), [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    sum += d_a[index] * d_b[index];
  });

  return T(sum);
}


void listDevices(void)
{
  std::cout << "This is not the device you are looking for.";
}


std::string getDeviceName(const int device)
{
  return "RAJA";
}


std::string getDeviceDriver(const int device)
{
  return "RAJA";
}

template class RAJAStream<float>;
template class RAJAStream<double>;
