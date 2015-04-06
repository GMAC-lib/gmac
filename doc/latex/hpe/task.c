typedef struct {
    vector_block_t *first, *second;
} vector_load_args_t;

typedef struct {
    vector_block_t *a_first, *a_second;
    vector_block_t *b_first, *b_second;
    vector_block_t *c_first, *c_second;
} vector_addition_args_t;

typedef vector_store_args_t vector_load_args_t;

. . .

    pthread_t vector_load_a_tid, vector_load_b_tid;
    pthread_t vector_addition_tid, vector_store_tid;

    vector_load_args_t vector_load_a_args, vector_load_b_args;
    vector_addition_args_t vector_addition_args;
    vector_store_args_t vector_store_args;
    
    /* Task to load the first input vector */
    vector_load_a_args.first = &vector_a_first;
    vector_load_a_args.second = &vector_a_second;
    if(pthread_create(&vector_load_a_tid, NULL, vector_load,
        (void *)&vector_load_a_args) != 0) goto vector_c_second_cleanup;

    /* Task to load the second input vector */
    vector_load_b_args.first = &vector_b_first;
    vector_load_b_args.second = &vector_b_second;
    if(pthread_create(&vector_load_b_tid, NULL, vector_load,
        (void *)&vector_load_b_args) != 0) goto wait_vector_load_a;

    /* Task to load the second input vector */
    vector_addition_args.a_first = &vector_a_first;
    vector_addition_args.a_second = &vector_a_second;
    vector_addition_args.b_first = &vector_b_first;
    vector_addition_args.b_second = &vector_b_second;
    vector_addition_args.c_first = &vector_c_first;
    vector_addition_args.c_second = &vector_c_second;
    if(pthread_create(&vector_addition_tid, NULL, vector_addition,
        (void *)&vector_addition_args) != 0) goto wait_vector_load_b;

    /* Task to load the second input vector */
    vector_store_args.first = &vector_c_first;
    vector_store_args.second = &vector_c_second;
    if(pthread_create(&vector_store_tid, NULL, vector_store,
        (void *)&vector_store_args) != 0) goto wait_vector_addition;

    
