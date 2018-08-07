
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

template <class T>
RAJAStream<T>::RAJAStream(const unsigned int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE)
{
  // RangeSegment(0, ARRAY_SIZE);
  // index_set.push_back(seg);

#ifdef RAJA_TARGET_CPU
  d_a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*ARRAY_SIZE);
  d_b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*ARRAY_SIZE);
  d_c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*ARRAY_SIZE);
#else
  std::cout << "memory allocation\n"; 
  a = (T*)malloc(sizeof(T) * ARRAY_SIZE);
  b = (T*)malloc(sizeof(T) * ARRAY_SIZE);
  c = (T*)malloc(sizeof(T) * ARRAY_SIZE);
  cudaMalloc((void**)&d_a, sizeof(T)*ARRAY_SIZE);
  cudaMalloc((void**)&d_b, sizeof(T)*ARRAY_SIZE);
  cudaMalloc((void**)&d_c, sizeof(T)*ARRAY_SIZE);
  cudaDeviceSynchronize();
#endif
}

template <class T>
RAJAStream<T>::~RAJAStream()
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

template <class T>
void RAJAStream<T>::init_arrays(T initA, T initB, T initC)
{
  std::cout << "init" << std::endl;
  // // T* RAJA_RESTRICT a = d_a;
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

  cudaMemcpy(d_a, a, sizeof(T)*array_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(T)*array_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, sizeof(T)*array_size, cudaMemcpyHostToDevice);

  std::cout << "end init" << std::endl;

  // T* tmp1 = (T*)malloc(sizeof(T) * array_size);
  // T* tmp2 = (T*)malloc(sizeof(T) * array_size);
  // T* tmp3 = (T*)malloc(sizeof(T) * array_size);
  // cudaMemcpy(tmp1, d_a, sizeof(T)*array_size, cudaMemcpyDeviceToHost);
  // cudaMemcpy(tmp2, d_b, sizeof(T)*array_size, cudaMemcpyDeviceToHost);
  // cudaMemcpy(tmp3, d_c, sizeof(T)*array_size, cudaMemcpyDeviceToHost);

  // std::cout << "test device init " << std::endl;
  // for (int i=0; i<array_size; i++)
  // {
  //   std::cout << "a[i] " << tmp1[i] << " b[i] " << tmp2[i] << " c[i] " << tmp3[i] << std::endl;
  // }
}

template <class T>
void RAJAStream<T>::read_arrays(
        std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
    std::cout << "read" << std::endl;

  std::copy(d_a, d_a + array_size, a.data());
  std::copy(d_b, d_b + array_size, b.data());
  std::copy(d_c, d_c + array_size, c.data());
}

// template <typename T>
// __global__ void copy_kernel(const T * a, T * c)
// {
//   const int i = blockDim.x * blockIdx.x + threadIdx.x;
//   c[i] = a[i];
// }

template <class T>
void RAJAStream<T>::copy()
{
  std::cout << "copy" << std::endl;
  std::cout << "array_size is " << array_size << std::endl;

  // T* tmp1 = (T*)malloc(sizeof(T) * array_size);
  // T* tmp2 = (T*)malloc(sizeof(T) * array_size);
  // T* tmp3 = (T*)malloc(sizeof(T) * array_size);
  // cudaMemcpy(tmp1, d_a, sizeof(T)*array_size, cudaMemcpyDeviceToHost);
  // cudaMemcpy(tmp2, d_b, sizeof(T)*array_size, cudaMemcpyDeviceToHost);
  // cudaMemcpy(tmp3, d_c, sizeof(T)*array_size, cudaMemcpyDeviceToHost);

  // std::cout << "test device init " << std::endl;
  // for (int i=0; i<array_size; i++)
  // {
  //   std::cout << "a[i] " << tmp1[i] << " b[i] " << tmp2[i] << " c[i] " << tmp3[i] << std::endl;
  // }
  // copy_kernel<<<1, 1024>>>(d_a, d_c);
  // std::cout << "kernel functino finished " << array_size << std::endl;

  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT c = d_c;
  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, array_size), 
    [=] RAJA_DEVICE (int i)
  {
    // std::cout << "inside copy, i is " << i << std::endl;
    // std::cout << "d_a[i] is " << d_a[i] << std::endl;
    printf("inside copy, i is%d\n", i);
    printf("d_a[i] is %lf\n", a[i]);
    c[i] = a[i];
  });
}

template <class T>
void RAJAStream<T>::mul()
{
    std::cout << "mul" << std::endl;
  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT c = d_c;
  const T scalar = startScalar;
  forall<RAJA::cuda_exec<256>>(RangeSegment(0, array_size), [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    c[index] = scalar*a[index];
  });
}

template <class T>
void RAJAStream<T>::add()
{
  std::cout << "add" << std::endl;

  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  forall<RAJA::cuda_exec<256>>(RangeSegment(0, array_size), [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    c[index] = a[index] + b[index];
  });
}

template <class T>
void RAJAStream<T>::triad()
{
    std::cout << "triad" << std::endl;

  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  const T scalar = startScalar;
  forall<RAJA::cuda_exec<256>>(RangeSegment(0, array_size), [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    c[index] = b[index] + scalar*a[index];
  });
}

template <class T>
T RAJAStream<T>::dot()
{
    std::cout << "dot" << std::endl;

  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;

  RAJA::ReduceSum<RAJA::cuda_reduce<256>, T> sum(0.0);

  forall<RAJA::cuda_exec<256>>(RangeSegment(0, array_size), [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    sum += a[index] * b[index];
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
