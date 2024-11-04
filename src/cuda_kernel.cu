#include "cuda_kernel.h"

__global__ void generateChessboard(unsigned char* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int index = (y * width + x) * 4;
        bool isWhite = ((x / (width / 8)) + (y / (height / 8))) % 2 == 0;
        buffer[index] = buffer[index + 1] = buffer[index + 2] = isWhite ? 255 : 0;
        buffer[index + 3] = 255;  // Alpha channel
    }
}

void generateChessBoardCaller(dim3 gridSize, dim3 blockSize, unsigned char* d_textureData, int texWidth, int texHeight) {
    generateChessboard<<<gridSize, blockSize>>>(d_textureData, texWidth, texHeight);
    cudaDeviceSynchronize();
}
