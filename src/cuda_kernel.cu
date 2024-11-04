// CUDA kernel declaration
__global__ void cudaKernel(float* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int index = y * width + x;
        buffer[index] = float(x) / width;
    }
}