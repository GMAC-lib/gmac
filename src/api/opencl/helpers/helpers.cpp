#include <cstdio>
#include <sys/stat.h>

#include <vector>

#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#else 
#   include <CL/cl.h>
#endif

#include <include/gmac/cl.h>

static std::vector<cl_helper> helpers;

static cl_int clHelperInitPlatform(cl_platform_id platform, cl_helper &state)
{
    cl_int error_code;
    cl_uint i, num_devices = 0, num_contexts = 0, num_queues = 0;

    state.platform = platform;

    /* Get the devices */
    error_code = clGetDeviceIDs(state.platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if(error_code != CL_SUCCESS) return error_code;
    state.devices = NULL;
    state.devices = new cl_device_id[num_devices];
    if(state.devices == NULL) return CL_OUT_OF_HOST_MEMORY;
    error_code = clGetDeviceIDs(state.platform, CL_DEVICE_TYPE_GPU, num_devices, state.devices, NULL);
    if(error_code != CL_SUCCESS) return error_code;

    /* Create contexts */
    state.contexts = NULL;
    state.contexts = new cl_context[num_devices];
    if(state.contexts == NULL) return CL_OUT_OF_HOST_MEMORY;
    for(num_contexts = 0; num_contexts < num_devices; num_contexts++) {
        state.contexts[num_contexts] = clCreateContext(NULL, 1, &state.devices[num_contexts], NULL, NULL, &error_code);
        if(error_code != CL_SUCCESS) goto cleanup_contexts;
    }

    /* Create command queues */
    state.command_queues = NULL;
    state.command_queues = new cl_command_queue[num_devices];
    if(state.command_queues == NULL) {
        error_code = CL_OUT_OF_HOST_MEMORY;
        goto cleanup_contexts;
    }
    for(num_queues = 0; num_queues < num_devices; num_queues++) {
        state.command_queues[num_queues] = clCreateCommandQueue(state.contexts[num_queues], state.devices[num_queues], 0, &error_code);
        if(error_code != CL_SUCCESS) goto cleanup_queues;
    }

	/* Create programs */
	state.programs = new cl_program[num_devices];
    for(i = 0; i < num_devices; i++) state.programs[i] = NULL;

    state.num_devices = num_devices;
    return CL_SUCCESS;

cleanup_queues:
    for(i = 0; i < num_queues; i++) clReleaseCommandQueue(state.command_queues[i]);
    delete [] state.command_queues;

cleanup_contexts:
    for(i = 0; i < num_contexts; i++) clReleaseContext(state.contexts[i]);
    delete [] state.contexts;
    
    return error_code;
}

cl_int APICALL clInitHelpers(size_t *platforms)
{
    cl_int error_code;
    cl_uint num_platforms = 0;

    error_code = clGetPlatformIDs(0, NULL, &num_platforms);
    if(error_code != CL_SUCCESS) return error_code;

    *platforms = size_t(num_platforms);
    cl_platform_id *tmp_platforms = new cl_platform_id[num_platforms];

    /* Get the platoforms */
    error_code = clGetPlatformIDs(num_platforms, tmp_platforms, NULL);
    if(error_code != CL_SUCCESS) goto cleanup;

    for (size_t i = 0; i < num_platforms; i++) {
        cl_helper helper;
        helper.platform = 0;
        helper.num_devices = 0;
        helper.devices = NULL;
		helper.programs = NULL;
        helper.contexts = NULL;
        helper.command_queues = NULL;
        error_code = clHelperInitPlatform(tmp_platforms[i], helper);
        if (error_code != CL_SUCCESS) goto cleanup;
        helpers.push_back(helper);
    }

cleanup:
    delete [] tmp_platforms;

    return error_code;
}

cl_helper * APICALL clGetHelpers()
{
    cl_helper *ret = new cl_helper[helpers.size()];
    for (size_t i = 0; i < helpers.size(); i++) {
        ret[i] = helpers[i];
    }
    return ret;
}

cl_int APICALL clReleaseHelpers()
{
    cl_int error_code = CL_SUCCESS;
    for (size_t i = 0; i < helpers.size(); i++) {
        cl_helper &helper = helpers[i];
        cl_uint j;
        for(j = 0; j < helper.num_devices; j++) {
            if(helper.command_queues != NULL) {
				error_code = clReleaseCommandQueue(helper.command_queues[j]);
			}
            if(error_code != CL_SUCCESS) return error_code;
			if(helper.contexts != NULL) {
				error_code = clReleaseContext(helper.contexts[j]);
			}
            if(error_code != CL_SUCCESS) return error_code;
			if(helper.programs != NULL) {
				error_code = clReleaseProgram(helper.programs[j]);
			}
            if(error_code != CL_SUCCESS) return error_code;
        }

		if(helper.command_queues != NULL) {
            delete [] helper.command_queues;
		}
        
		if(helper.contexts != NULL) {
            delete [] helper.contexts;
		}

		if(helper.programs != NULL) {
            delete [] helper.programs;
		}

		if(helper.devices != NULL) {
            delete [] helper.devices;
		}
    }

    helpers.clear();
    
    return CL_SUCCESS;
}


static const char *build_flags = "-I.";

cl_int APICALL clHelperLoadProgramFromFile(cl_helper state, const char *file_name)
{
    /* Let's all thank Microsoft for such a great compatibility */
#if defined(_MSC_VER)
#   define stat _stat
#endif

    cl_int ret = CL_SUCCESS;
    FILE *fp;
    struct stat file_stats;
    char *buffer = NULL;
    size_t read_bytes;
    cl_uint i = 0;
	int stat_ret;
	stat_ret = stat(file_name, &file_stats);
    if(stat_ret < 0) {
		return CL_INVALID_VALUE;
	}
#if defined(_MSC_VER)
#   undef stat
#endif

    buffer = new char[file_stats.st_size];
    if(buffer == NULL) { ret = CL_OUT_OF_HOST_MEMORY; return ret; }

#if defined(_MSC_VER)
	if(fopen_s(&fp, file_name, "rt") != 0) { ret = CL_INVALID_VALUE; goto cleanup; }
#else
    fp = fopen(file_name, "rt");
    if(fp == NULL) { ret = CL_INVALID_VALUE; goto cleanup; }
#endif
    read_bytes = fread(buffer, sizeof(char), file_stats.st_size, fp);
    fclose(fp);
    if(read_bytes != (size_t)file_stats.st_size) {
        ret = CL_INVALID_VALUE;
        goto cleanup;
    }

    for(i = 0; i < state.num_devices; i++) {
        state.programs[i] = clCreateProgramWithSource(state.contexts[i], 1, (const char **)&buffer, &read_bytes, &ret);
        if(ret != CL_SUCCESS) {
            state.programs[i] = NULL;
            goto cleanup;
        }
        ret = clBuildProgram(state.programs[i], 1, &state.devices[i], build_flags, NULL, NULL);
        if(ret != CL_SUCCESS) {
            state.programs[i] = NULL;
            goto cleanup;
        }
    }


cleanup:
    delete [] buffer;

    return ret;
}
