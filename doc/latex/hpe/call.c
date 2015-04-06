    ecl_kernel kernel;
    ecl_error error_code;
    cl_uint global_size = vec_size;

    error_code = eclGetKernel("vecAdd", &kernel);
    if(error_code != eclSuccess) return error(error_code);

    error_code = eclSetKernelArgPtr(&kernel, 0, c);
    if(error_code != eclSuccess)
        return error(error_code);
    error_code = eclSetKernelArgPtr(&kernel, 1, a);
    if(error_code != eclSuccess)
        return error(error_code);
    error_code = eclSetKernelArgPtr(&kernel, 2, b);
    if(error_code != eclSuccess)
        return error(error_code);

    error_code = eclCallNDRange(&kernel, 1, NULL,
                                &global_size, NULL);
    if(error_code != eclSuccess)  return error(error_code);

