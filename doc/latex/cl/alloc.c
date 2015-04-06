    float *a, *b, *c;
    ocl_error error_code;

    /* Allocate the input and output vectors */
    error_code = clMalloc(helper.command_queues[0], (void **)&a,
                          vector_size * sizeof(float));
    if(error_code != oclSuccess) return error(error_code);
    error_code = clMalloc(helper.command_queues[0], (void **)&b,
                          vector_size * sizeof(float));
    if(error_code != oclSuccess) {
        clFree(helper.command_queues[0], a);
        return error(error_code);
    }
    error_code = clMalloc(helper.command_queues[0], (void **)&c,
                          vector_size * sizeof(float));
    if(error_code != oclSuccess) {
        clFree(helper.command_queues[0], a); clFree(helper.command_queues[0], b);
        return error(error_code);
    }

    /* Initialize the input vectors */
    if(load_vector(vector_a_file, a) < 0) {
        fprintf(stderr, "Error loading %s\n", vector_a_file);
        clFree(helper.command_queues[0], a);
        clFree(helper.command_queues[0], b);
        clFree(helper.command_queues[0], c); abort();
    }
    if(load_vector(vector_b_file, b) < 0) {
        fprintf(stderr, "Error loading %s\n", vector_b_file);
        clFree(helper.command_queues[0], a);
        clFree(helper.command_queues[0], b);
        clFree(helper.command_queues[0], c); abort();
    }
