
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <vector>
#include <string>

// Array values
#define startA (0.1)
#define startB (0.2)
#define startC (0.0)
#define startScalar (0.4)

// template <class double>
class Stream
{
  public:

    virtual ~Stream(){}

    // Kernels
    // These must be blocking calls
    virtual void copy() = 0;
    virtual void mul() = 0;
    virtual void add() = 0;
    virtual void triad() = 0;
    virtual double dot() = 0;

    // Copy memory between host and device
    virtual void init_arrays(double initA, double initB, double initC) = 0;
    virtual void read_arrays(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c) = 0;

};


// Implementation specific device functions
void listDevices(void);
std::string getDeviceName(const int);
std::string getDeviceDriver(const int);

