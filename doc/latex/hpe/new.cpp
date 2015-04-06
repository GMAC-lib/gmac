    float *a, *b, *c;
    ecl_error error_code;

    /* Allocate the input and output vectors */
    a = new (ecl::allocator)float[vector_size];
    if(a == NULL) return error(eclErrorMemoryAllocation);
    b = new (ecl::allocator)float[vector_size];
    if(b == NULL) {
        ecl::free(a);
        return error(eclErrorMemoryAllocation);
    }
    c = new (ecl::allocator)float[vector_size];
    if(c == NULL) {
        ecl::free(a); ecl::free(b);
        return error(eclErrorMemoryAllocation);
    }

    /* Initialize the input vectors */
    if(load_vector(vector_a_file, a) < 0) {
        std::cerr << "Error loading " << vector_a_file << std::endl;
        ecl::free(a); ecl::free(b); ecl::free(c); abort();
    }
    if(load_vector(vector_b_file, b) < 0) {
        std::cerr << "Error loading " << vector_b_file << std::endl;
        ecl::free(a); ecl::free(b); ecl::free(c); abort();
    }
