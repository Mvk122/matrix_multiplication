#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <x86intrin.h>
#include <omp.h>

#define PRINT_MATRICES 0 // Determines whether to print matrices
#define MIN 0.0 // Min value in matrix
#define MAX 1.0 // Max value in matrix
#define UNROLL 4 // Number of times to unroll loop in unrolled_matrix_multiply()
#define OMP_THREADS 4 // Number of threads to use with omp pragma
// Block size 32 is better for the combined solution
#define BLOCK_SIZE 32 // Block size when using blocked_matrix_multiply()
#define MM256_STRIDE 4 // Number of doubles operated on simultaneously in AVX instructions
#define MM512_STRIDE 8 // Double the memory stride of MM256 for AVX512
#define MEM_ALIGN 64 // Memory alignment required for _mm256 instructions (Use 32 for AVX256 and 64 for AVX512)


void simd_do_block_unrolled(int si, int sj, int sk, double **A, double **B, double **C){
    /* iterate over the rows of A in the block*/
    for (int i=si; i<si+BLOCK_SIZE; i++){
        
        // Loop unrolling by making 4 c0 variables to allow the CPU to 
        // pipeline UNROLL amount at a time.
        // MM512_STRIDE*UNROLL as due to the unrolling, UNROLL amount of
        // AVX instructions are run at a time. 
        for (int j=sj; j<sj+BLOCK_SIZE; j+=(MM512_STRIDE*UNROLL)){
            __m512d c0[UNROLL];
            // Loading each of the 4 variables into the array
            for (int u=0; u<UNROLL; u++) {
                c0[u] = _mm512_load_pd(&C[i][j+ (u*MM512_STRIDE)]);
            }
            // Doing the SIMD calculation
            for(int k=sk; k<sk+BLOCK_SIZE; k++){

                // Supposed to be faster by using multiply and add at the same time but is slower
                // c0[u] = _mm256_fmadd_pd(_mm256_load_pd(&B[k][j+ (u*MM512_STRIDE)]), 
                //                              _mm256_broadcast_sd(&A[i][k]), c0[u]);
                
                // Do the AVX512 matrix multiply and adds in a for loop
                // Loop unrolling is done by the compiler here as it will always loop UNROLL times
                for (int u=0; u<UNROLL; u++) {
                    c0[u] = _mm512_add_pd(c0[u],
                                        _mm512_mul_pd(_mm512_loadu_pd(&B[k][j+ (u*MM512_STRIDE)]), 
                                _mm512_set1_pd(A[i][k])));
                }
            }
            // Loop unrolling happens here too
            for (int u=0; u<UNROLL; u++) {
                _mm512_storeu_pd(&C[i][j+ (u*MM512_STRIDE)], c0[u]);
            }
        }
    }
}

void combined_matrix_multiply(double **A, double **B, double **C, int L, int M, int N) {
    // Multicore for the entire solution
    #pragma omp parallel for
    // Blocking solution to ensure that variables stay in the cache/register
    // without cache replacement by using the same variables until they are
    // no longer needed
    /* iterate over the rows of A */
    for(int sj=0; sj<L; sj+=BLOCK_SIZE) {
        /* Iterate over the columns of B */
        for(int si=0; si<N; si+=BLOCK_SIZE) {
            /* Iterate over the rows of B */
            for(int sk=0; sk<M; sk+=BLOCK_SIZE){
                // Actually doing the computation for the block
                simd_do_block_unrolled(si, sj, sk, A, B, C);
            }
        }
    }
}