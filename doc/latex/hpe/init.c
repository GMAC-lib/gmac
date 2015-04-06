int allocate_vector_block(vector_block_t *v)
{
    if(sem_init(&v->ready, 0, 0) < 0) return -1;
    if(sem_init(&v->reuse, 0, 0) < 0) goto ready_cleanup;
    if(eclMalloc((void *)&v->ptr, vector_block_size) != eclSuccess)
        goto reuse_cleanup;

    return 0;

reuse_cleanup: sem_destroy(&v->reuse);
ready_cleanup: sem_destroy(&v->ready);
    return -1;
}

. . .

    int ret = -1;

    /* Local data structures */
    vector_block_t vector_a_first, vector_a_second;
    vector_block_t vector_b_first, vector_b_second;
    vector_block_t vector_c_first, vector_c_second;

    /* Allocate first input vector */
    if(allocate_vector_block(&vector_a_first) < 0)
        return -1;
    if(allocate_vector_block(&vector_a_second) < 0)
        goto vector_a_first_cleanup;

    /* Allocate second input vector */
    if(allocate_vector_block(&vector_b_first) < 0)
        goto vector_a_second_cleanup;
    if(allocate_vector_block(&vector_b_second) < 0)
        goto vector_b_first_cleanup;

    /* Allocate output vector */
    if(allocate_vector_block(&vector_c_first) < 0)
        goto vector_b_second_cleanup;
    if(allocate_vector_block(&vector_c_second) < 0)
        goto vector_c_first_cleanup;
    
