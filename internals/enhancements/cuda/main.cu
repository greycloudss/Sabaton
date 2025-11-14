#include <cuda_runtime.h>              // Pulls in CUDA runtime API declarations (cudaMalloc, cudaMemcpy, etc.).
                                       // Without this, the host code cannot call CUDA functions or manage GPU memory.

__global__ void set_value(int* x) {    // __global__ marks this as a kernel function that runs on the GPU,
                                       // and can be launched from the CPU. It defines a GPU entry point.
    *x = 42;                           // Dereferences the device pointer and writes 42 into that location.
                                       // This is executed on the GPU, not the CPU.
}

int main() {                           // Standard C/C++ program entry point running on the CPU (host).
    int* d_x;                          // Pointer that will point to memory allocated on the device (GPU).
    int h_x;                           // Plain CPU (host) integer that will receive the value from the GPU.

    cudaMalloc(&d_x, sizeof(int));     // Allocates sizeof(int) bytes on the GPU and writes the resulting device
                                       // address into d_x. This is essential: GPU memory is separate from CPU memory.

    set_value<<<1, 1>>>(d_x);          // Launches the GPU kernel with 1 block and 1 thread per block.
                                       // The kernel will run on the GPU and receive d_x as a device pointer.

    cudaMemcpy(&h_x,                   // Destination: address of host variable h_x (CPU side).
               d_x,                    // Source: device pointer d_x (GPU side).
               sizeof(int),            // Number of bytes to copy.
               cudaMemcpyDeviceToHost  // Direction of copy: from GPU memory to CPU memory.
    );                                 // This is how you get data back from the GPU to the CPU.

    cudaFree(d_x);                     // Frees the previously allocated GPU memory to avoid leaks on the device.

    return 0;                          // Normal program exit status. At this point, h_x holds whatever the kernel wrote.
}
