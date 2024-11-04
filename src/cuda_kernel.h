#pragma once
#include <cuda_runtime.h>

void generateChessBoardCaller(dim3 gridSize, dim3 blockSize, unsigned char* d_textureData, int texWidth, int texHeight);
