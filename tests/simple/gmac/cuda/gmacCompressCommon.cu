#define PI 3.14159265358979

const size_t blockSize = 16;

__shared__ float tile[blockSize][blockSize];

__global__ void dct(float *out, float *in, size_t width, size_t height)

{
	int l = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	/* Pre-compute some values */
	float alpha, beta;
	if(k == 0) alpha = sqrtf(1.0 / width);
	else alpha = sqrtf(2.0 / width);
	if(l == 0) beta = sqrtf(1.0 / height);
	else beta = sqrtf(2.0 / height);

	float a = (PI / width) * k;
	float b = (PI / height) * l;

	float o = 0;
	for(int j = 0; j < height; j += blockDim.y) {
		for(int i = 0; i < width; i+= blockDim.x) {
			/* Calculate n and m values */
			int y = j + threadIdx.y;
			int x = i + threadIdx.x;

			/* Prefetch data in shared memory */
			if(x < width && y < height)
				tile[threadIdx.x][threadIdx.y] = in[y * width + x];
			__syncthreads();

			/* Compute the partial DCT */
			for(int m = 0; m < blockDim.y; m++) {
				for(int n = 0; n < blockDim.x; n++) {
					o += tile[m][n] * cosf(a * (n + i + 0.5)) * cosf(b * (m + j + 0.5));
				}
			}
			
			/* Done computing the DCT for the sub-block */
		}
	}

	if(k < width && l < height) {
		out[(l * width) + k] = alpha * beta * o;
	}
}

__global__ void idct(float *out, float *in, size_t width, size_t height)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	/* Pre-compute some values */
	float alpha, beta;

	float a = (PI / width) * (x + 0.5);
	float b = (PI / height) * (y + 0.5);

	float o = 0;
				
	for(int j = 0; j < height; j += blockDim.y) {
		for(int i = 0; i < width; i+= blockDim.x) {
			/* Calculate n and m values */
			int l = j + threadIdx.y;
			int k = i + threadIdx.x;

			/* Prefetch data in shared memory */
			if(i + threadIdx.x < width && j + threadIdx.y < height)
				tile[threadIdx.x][threadIdx.y] = in[l * width + k];
			__syncthreads();

			/* Compute the partial IDCT */
			for(int m = 0; m < blockDim.y; m++) {
				for(int n = 0; n < blockDim.x; n++) {
					/* Pre-compute some values */
					if((n + i) == 0) alpha = sqrtf(1.0 / width);
					else alpha = sqrtf(2.0 / width);
					if((m + j) == 0) beta = sqrtf(1.0 / height);
					else beta = sqrtf(2.0 / height);
					o += alpha * beta * tile[m][n] * cosf(a * (n + i)) * cosf(b * (m + j));
				}
			}
			
			/* Done computing the DCT for the sub-block */
		}
	}

	if(x < width && y < height) {
		out[(y * width) + x] = o;
	}

}

__global__ void quant(float *out, float *in, size_t width, size_t height, float k)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if(x < width && y < height) {
		float f = fabsf(in[y * width + x]);
		if(f > k) out[(y * width) + x] = f;
		else out[(y * width) + x] = 0.0;
	}
}

