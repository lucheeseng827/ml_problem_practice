extern "C"
{

    __global__ void matrix_ops(const float *A, const float *B, float *dot, float *cross, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n * n)
        {
            int row = idx / n;
            int col = idx % n;

            dot[idx] = 0.0f;

            for (int k = 0; k < n; k++)
            {
                dot[idx] += A[row * n + k] * B[k * n + col];
            }

            if (row < n && col < n && row != col)
            {
                cross[row * n + col] = A[row * n + (col + 1) % n] * B[(row + 1) % n * n + col] -
                                       A[row * n + col] * B[(row + 1) % n * n + (col + 1) % n];
            }
            else
            {
                cross[row * n + col] = 0;
            }
        }
    }
}
