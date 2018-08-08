
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
#include "umpire/ResourceManager.hpp"

using RAJA::forall;
using RAJA::RangeSegment;

// #ifndef ARRAY_SIZE
// #define ARRAY_SIZE 33554432
// #endif

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

auto& rm = umpire::ResourceManager::getInstance();
auto h_alloc = rm.getAllocator("HOST");
auto d_alloc = rm.getAllocator("DEVICE");
auto d_const_alloc = rm.getAllocator("DEVICE_CONST");

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
  a = static_cast<T*>(h_alloc.allocate(array_size * sizeof(T)));
  b = static_cast<T*>(h_alloc.allocate(array_size * sizeof(T)));
  c = static_cast<T*>(h_alloc.allocate(array_size * sizeof(T)));
  d_a = static_cast<T*>(d_const_alloc.allocate(array_size * sizeof(T)));
  d_b = static_cast<T*>(d_alloc.allocate(array_size * sizeof(T)));
  d_c = static_cast<T*>(d_alloc.allocate(array_size * sizeof(T)));
  cudaDeviceSynchronize();
#endif
}

template <class T>
RAJAStream<T>::~RAJAStream()
{
  h_alloc.deallocate(a);
  h_alloc.deallocate(b);
  h_alloc.deallocate(c);
#ifdef RAJA_TARGET_CPU
  free(d_a);
  free(d_b);
  free(d_c);
#else
  d_const_alloc.deallocate(d_a);
  d_alloc.deallocate(d_b);
  d_alloc.deallocate(d_c);
#endif
}

template <class T>
void RAJAStream<T>::init_arrays(T initA, T initB, T initC)
{
  // std::cout << "init" << std::endl;
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
    b[i] = initB;
    c[i] = initC;
  }

  rm.copy(d_a, a, sizeof(T)*array_size);
  rm.copy(d_b, b, sizeof(T)*array_size);
  rm.copy(d_c, c, sizeof(T)*array_size);

}

template <class T>
void RAJAStream<T>::read_arrays(
        std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
    // std::cout << "read" << std::endl;

  cudaMemcpy(a.data(), d_a, sizeof(T)*array_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(b.data(), d_b, sizeof(T)*array_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(c.data(), d_c, sizeof(T)*array_size, cudaMemcpyDeviceToHost);
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
  // std::cout << "copy" << std::endl;
  // std::cout << "array_size is " << array_size << std::endl;

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
    c[i] = a[i];
  });
}

template <class T>
void RAJAStream<T>::mul()
{
    // std::cout << "mul" << std::endl;
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
  // std::cout << "add" << std::endl;

  T* RAJA_RESTRICT a = d_a;
  T* RAJA_RESTRICT b = d_b;
  T* RAJA_RESTRICT c = d_c;
  RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, array_size), 
    [=] RAJA_DEVICE (RAJA::Index_type index)
  {
    c[index] = a[index] + b[index];
  });
}

template <class T>
void RAJAStream<T>::triad()
{
    // std::cout << "triad" << std::endl;

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
    // std::cout << "dot" << std::endl;

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
