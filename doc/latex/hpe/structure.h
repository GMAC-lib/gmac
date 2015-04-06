typedef struct {
    float *ptr;
    semaphore_t ready;
    semaphore_t reuse;
} vector_block_t;
