/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
 
__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
	/********************************************************************
	Determine input and output indexes of each thread
	Load a tile of the input image to shared memory
	Apply the filter on the input image tile
	Write the compute values to the output image at the correct indexes
	********************************************************************/

    //INSERT KERNEL CODE HERE
    
    __shared__ float N_c[BLOCK_SIZE][BLOCK_SIZE];
    
	int row;
	int col;
	int sRow = threadIdx.y + (FILTER_SIZE / 2);
	int sCol = threadIdx.x + (FILTER_SIZE / 2);
	int gRow = blockIdx.y * TILE_SIZE + threadIdx.y;
	int gCol = blockIdx.x * TILE_SIZE + threadIdx.x;
	float pVal = 0.0;
	int i = 0;
	int j = 0;

	if (gRow >= 0 && gRow < N.height && gCol >= 0 && gCol < N.width)
		N_c[sRow][sCol] = N.elements[gRow * N.width + gCol];
	else
		N_c[sRow][sCol] = 0.0;

	if (threadIdx.y < (FILTER_SIZE / 2) && threadIdx.x < (FILTER_SIZE / 2)) {
		row = (blockIdx.y + 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x + 1) * TILE_SIZE + threadIdx.x;

		if (row < N.height && col < N.width && row >= 0 && col >= 0)
			N_c[sRow + TILE_SIZE][sCol + TILE_SIZE] = N.elements[row * N.width + col];
		else
			N_c[sRow + TILE_SIZE][sCol + TILE_SIZE] = 0.0;
	}

	if (threadIdx.y >= TILE_SIZE - (FILTER_SIZE / 2) && threadIdx.x < (FILTER_SIZE / 2)) {
		row = (blockIdx.y - 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x + 1) * TILE_SIZE + threadIdx.x; 

		if (row < N.height && col < N.width && row >= 0 && col >= 0)
			N_c[sRow - TILE_SIZE][sCol + TILE_SIZE] = N.elements[row * N.width + col];
		else
			N_c[sRow - TILE_SIZE][sCol + TILE_SIZE] = 0.0;
	}

	if (threadIdx.y < (FILTER_SIZE / 2) && threadIdx.x >= TILE_SIZE - (FILTER_SIZE / 2)) {
		row = (blockIdx.y + 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x - 1) * TILE_SIZE + threadIdx.x;

		if (row < N.height && col < N.width && row >= 0 && col >= 0)
			N_c[sRow + TILE_SIZE][sCol - TILE_SIZE] = N.elements[row * N.width + col];
		else
			N_c[sRow + TILE_SIZE][sCol - TILE_SIZE] = 0.0;
	}

	if (threadIdx.y >= TILE_SIZE - (FILTER_SIZE / 2) && threadIdx.x >= TILE_SIZE - (FILTER_SIZE / 2)) {
		row = (blockIdx.y - 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x - 1) * TILE_SIZE + threadIdx.x;

		if (row < N.height && col < N.width && row >= 0 && col >= 0)
			N_c[sRow - TILE_SIZE][sCol - TILE_SIZE] = N.elements[row * N.width + col];
		else
			N_c[sRow - TILE_SIZE][sCol - TILE_SIZE] = 0.0;
	}

	if (threadIdx.y < (FILTER_SIZE / 2)) {
		row = (blockIdx.y + 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x) * TILE_SIZE + threadIdx.x;

		if (row < N.height && col < N.width && row >= 0 && col >= 0)
			N_c[sRow + TILE_SIZE][sCol - 0] = N.elements[row * N.width + col];
		else
			N_c[sRow + TILE_SIZE][sCol - 0] = 0.0;
	}

	if (threadIdx.x < (FILTER_SIZE / 2)) {
		row = (blockIdx.y) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x + 1) * TILE_SIZE + threadIdx.x;

		if (row < N.height && col < N.width && row >= 0 && col >= 0)
			N_c[sRow][sCol + TILE_SIZE] = N.elements[row * N.width + col];
		else
			N_c[sRow][sCol + TILE_SIZE] = 0.0;
	}

	if (threadIdx.y >= TILE_SIZE - (FILTER_SIZE / 2)) {
		row = (blockIdx.y - 1) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x) * TILE_SIZE + threadIdx.x;

		if (row < N.height && col < N.width && row >= 0 && col >= 0)
			N_c[sRow - TILE_SIZE][sCol] = N.elements[row * N.width + col];
		else
			N_c[sRow - TILE_SIZE][sCol] = 0.0;
	}

	if (threadIdx.x >= TILE_SIZE - (FILTER_SIZE / 2)) {
		row = (blockIdx.y) * TILE_SIZE + threadIdx.y;
		col = (blockIdx.x - 1) * TILE_SIZE + threadIdx.x;
		
		if (row < N.height && col < N.width && row >= 0 && col >= 0)
			N_c[sRow - 0][sCol - TILE_SIZE] = N.elements[row * N.width + col];
		else
			N_c[sRow - 0][sCol - TILE_SIZE] = 0.0;
	}

	__syncthreads();

	if (gRow < P.height && gCol < P.width) {
		for (int z = -2; z < 3; z++) {
			for (int x = -2; x < 3; x++) {
				i = z + 2;
				j = x + 2;
				pVal += N_c[sRow + z][sCol + x] * M_c[i][j];
			}
		}
		P.elements[gRow * P.width + gCol] = pVal;
	}
}