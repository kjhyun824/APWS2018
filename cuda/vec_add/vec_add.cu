#include <stdio.h>
#include <stdlib.h>

__global__ void gpuVecAdd(float *A, float *B, float *C) {
  // TODO: write kernel code here
}

void init(float *V, int N) {
  for (int i = 0; i < N; i++) {
    V[i] = rand() % 100;
  }
}

void verify(float *A, float *B, float *C, int N) {
  for (int i = 0; i < 16384; i++) {
    if (A[i] + B[i] != C[i]) {
      printf("Verification failed! A[%d] = %d, B[%d] = %d, C[%d] = %d\n",
             i, A[i], i, B[i], i, C[i]);
      return;
    }
  }
  printf("Verification success!\n");
}

int main() {
  int N = 16384;

  float *A = (float*)malloc(sizeof(float) * N); 
  float *B = (float*)malloc(sizeof(float) * N); 
  float *C = (float*)malloc(sizeof(float) * N);

  init(A, N);
  init(B, N);

  // Memory objects of the device
  float *d_A, *d_B, *d_C;

  // TODO: allocate memory objects d_A, d_B, and d_C.

  // TODO: copy "A" to "d_A" (host to device).
  // TODO: copy "B" to "d_B" (host to device).

  // TODO: launch the kernel.

  // TODO: copy "d_C" to "C" (device to host).

  verify(A, B, C, N);

  // TODO: release d_A, d_B, and d_C.

  free(A);
  free(B);
  free(C);

  return 0;
}

