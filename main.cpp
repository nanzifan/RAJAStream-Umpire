
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <iostream>

#define VERSION_STRING "3.3"

#include "RAJAStream.hpp"

// Default size of 2^13
unsigned int ARRAY_SIZE = 16;

int main(int argc, char *argv[])
{

  RAJAStream *r;
  r = new RAJAStream(ARRAY_SIZE);
  r->init_arrays(1.0,2.0,0);
  r->copy();
}

