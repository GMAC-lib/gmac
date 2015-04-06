    cl_mem mem;
    cl_kernel kernel;
    cl_uint global_size = vector_size;

    kernel = clCreateKernel(helper.programs[0], "vecAdd", &error_code);
    if(error_code != CL_SUCCESS) return error(error_code);

    mem = clBuffer(context, c);
    error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem);
    if(mem == NULL || error_code != CL_SUCCESS)
        return error(error_code);
    mem = clBuffer(context, a);
    error_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem);
    if(mem == NULL || error_code != CL_SUCCESS)
        return error(error_code);
    mem = clBuffer(context, b);
    error_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem);
    if(mem == NULL || error_code != CL_SUCCESS)
        return error(error_code);

    error_code = clEnqueueNDRangeKernel(helper.command_queue[0], kernel, 1,
                                        NULL, &global_size, NULL,
                                        0, NULL, NULL);
    if(error_code != CL_SUCCESS) return error(error_code);
    error_code = clFinish(helper.command_queue[0]);
    if(error_code != CL_SUCCESS) return error(error_code);

