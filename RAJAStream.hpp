// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include "RAJA/RAJA.hpp"

#include "Stream.h"

#define IMPLEMENTATION_STRING "RAJA"

#ifdef RAJA_TARGET_CPU
typedef RAJA::policy::indexset::ExecPolicy<
        RAJA::seq_segit,
        RAJA::omp_parallel_for_exec> policy;
typedef RAJA::omp_reduce reduce_policy;
#else
// const size_t block_size = 128;
// typedef RAJA::cuda_exec<block_size> policy;
// typedef RAJA::cuda_reduce<block_size> reduce_policy;
#endif

// template <class double>
class RAJAStream : public Stream<double>
{
  protected:
    // Size of arrays
    unsigned int array_size;

    // Contains iteration space
    // RAJA::IndexSet index_set;

    double* a;
    double* b;
    double* c;

    // Device side pointers to arrays
    double* d_a;
    double* d_b;
    double* d_c;

  public:

    RAJAStream(const unsigned int, const int);
    ~RAJAStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual double dot() override;

    virtual void init_arrays(double initA, double initB, double initC) override;
    virtual void read_arrays(
            std::vector<double>& a, std::vector<double>& b, std::vector<double>& c) override;
};

