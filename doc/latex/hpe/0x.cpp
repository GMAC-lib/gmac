    ecl::kernel kernel;
    ecl::config config(vec_size);

    error_code = eclGetKernel("vecAdd", &kernel);
    if(error_code != eclSuccess) return error(error_code);

    error_code = kernel(config)(c, a, b);
    if(error_code != eclSuccess)  return error(error_code);

