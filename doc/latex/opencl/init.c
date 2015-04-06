    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    FILE *fp;
    stat file_stat;
    size_t n;
    char *source_code = NULL;

    /* Create an OpenCL context containing a GPU */
    context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &error_code);
    if(error_code != CL_SUCCESS) return error(error_code);

   /* Open the file containing the kernel 
     * We need to first get the source file size to allocate
     * the necessary memory
     */
    if(stat(kernel_file, &stat) < 0) {
        clReleaseContext(context);
        return error(CL_INVALID_PROGRAM);
    }
    source_code = (char *)malloc(stat.st_size);
    if(source_code == NULL) {
        clReleaseContext(context);
        return error(CL_OUT_OF_HOST_MEMORY);
    }
    fp = fopen(kernel_file, "rt");
    if(fp == NULL) {
        free(source_code);
        clReleaseContext(context);
        return error(CL_INVALID_PROGRAM);
    }
    n = fread(source_code, stat.st_size, sizeof(char), fp);
    fclose(fp);
    if(n != stat.st_size) {
        free(source_code);
        clReleaseContext(context);
        return error(CL_INVALID_PROGRAM);
    }
    program = clCreateProgramWithSource(context, 1, &source_code,
                                        &n, &error_code);
    free(source_code);
    if(error_code != CL_SUCCESS) {
        clReleaseContext(context);
        return error(error_code);
    }

    /* We get the OpenCL device bound to the context, becase we
     * need that * device to build the program and create the
     * command queue
     */
    error_code = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES,
                                  sizeof(num_devices), &num_devices,
                                   NULL);
    if(error_code != CL_SUCCESS || num_devices == 0) {
        clReleaseProgram(program); clReleaseContext(context);
        return error(CL_DEVICE_NOT_AVAILABLE);
    }
    devices = (cl_device_id *)
              malloc(num_devices * sizeof(cl_device_id));
    if(devices == NULL) { 
        clReleaseProgram(program); clReleaseContext(context);
        return error(CL_OUT_OF_HOST_MEMORY);
    }
    error_code = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                                  num_devices * sizeof(cl_device_id),
                                  devices, NULL);
    if(error_code != CL_SUCCESS) {
        free(devices);
        clReleaseProgram(program); clReleaseContext(context);
        return error(error_code);
    }

    /* Buidl the program */
    error_code = clBuildProgram(program, 1, *devices, NULL,
                                NULL, NULL);
    if(error_code != CL_SUCCESS) {
        free(devices);
        clReleaseProgram(program); clReleaseContext(context);
        return error(error_code);
    }

    /* Finally, we create a command queue to be able to
     * call kernelss
     */
    command_queue = clCreateCommandQueue(context, *devices,
                                         NULL, &error_code);
    free(devices);
    if(error_code != CL_SUCCESS) { 
        clReleaseProgram(program); clReleaseContext(context);
        return error(error_code);
    }

