#include <iostream>

// Default size of 2^13
unsigned int ARRAY_SIZE = 16;

int main(int argc, char *argv[])
{
	unsigned int array_size = ARRAY_SIZE;


    double* a;
    double* b;
    double* c;

    // Device side pointers to arrays
    double* d_a;
    double* d_b;
    double* d_c;

    std::cout << "memory allocation\n"; 
	a = (double*)malloc(sizeof(double) * ARRAY_SIZE);
	b = (double*)malloc(sizeof(double) * ARRAY_SIZE);
	c = (double*)malloc(sizeof(double) * ARRAY_SIZE);
	cudaMalloc((void**)&d_a, sizeof(double)*ARRAY_SIZE);
	cudaMalloc((void**)&d_b, sizeof(double)*ARRAY_SIZE);
	cudaMalloc((void**)&d_c, sizeof(double)*ARRAY_SIZE);

    std::cout << "init" << std::endl;

	for (int i=0; i<array_size; i++)
	{
		a[i] = 1.0;
		b[i] = 2.0;
		c[i] = 0.0;
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

	std::cout << "copy" << std::endl;
	std::cout << "array_size is " << array_size << std::endl;

	tmp1 = (double*)malloc(sizeof(double) * array_size);
	tmp2 = (double*)malloc(sizeof(double) * array_size);
	tmp3 = (double*)malloc(sizeof(double) * array_size);
	cudaMemcpy(tmp1, d_a, sizeof(double)*array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp2, d_b, sizeof(double)*array_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp3, d_c, sizeof(double)*array_size, cudaMemcpyDeviceToHost);

	std::cout << "test device init " << std::endl;
	for (int i=0; i<array_size; i++)
	{
		std::cout << "a[i] " << tmp1[i] << " b[i] " << tmp2[i] << " c[i] " << tmp3[i] << std::endl;
	}

	// double* RAJA_RESTRICT da = d_a;
	// double* RAJA_RESTRICT dc = d_c;

	RAJA::forall<RAJA::cuda_exec<256>>(RAJA::RangeSegment(0, array_size), 
		[=] RAJA_DEVICE (int i) {
			printf("inside copy, i is%d\n", i);
			printf("d_a[i] is %d\n", d_a[i]);
			d_c[i] = d_a[i];
	});

	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}
