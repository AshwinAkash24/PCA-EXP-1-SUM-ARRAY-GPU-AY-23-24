# PCA: EXP-1  SUM ARRAY GPU
## ENTER YOUR NAME: Ashwin Akash M
## ENTER YOUR REGISTER NO: 212223230024
## DATE :11-09-2025
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
```python
%%writefile matrix_add.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h> // for abs()

// Check CUDA errors
#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}
#endif

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Initialize data
void initialData(float *ip, const int size)
{
    for(int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// Host matrix addition
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++)
        for (int ix = 0; ix < nx; ix++)
            C[iy * nx + ix] = A[iy * nx + ix] + B[iy * nx + ix];
}

// Check result
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double eps = 1.0E-8;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > eps)
        {
            match = false;
            printf("Mismatch at %d: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }

}

// GPU kernel
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

// Main program
int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Matrix size (1024 x 1024)
    int nx = 1 << 10;
    int ny = 1 << 10;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // Host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // Init data
    double start = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    printf("Initialization time: %f sec\n", seconds() - start);

    // CPU computation
    start = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    printf("Host computation time: %f sec\n", seconds() - start);

    // Allocate GPU memory
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // Copy to GPU
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // GPU computation
    start = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    printf("GPU computation time: %f sec\n", seconds() - start);

    // Copy results back
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // Compare results
    checkResult(hostRef, gpuRef, nxy);

    // Free memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());
    return 0;
}

```

## OUTPUT:
<img width="429" height="142" alt="image" src="https://github.com/user-attachments/assets/e968fd30-942f-48b9-a978-9ce9cd0385af" />



## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
