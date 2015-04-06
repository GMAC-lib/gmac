    FILE *fp;

    /* Release input vectors */
    clFree(helper.command_queues[0], a);
    clFree(helper.command_queues[0], b);

    /* Write and release the output vector */
    fp = fopen(vector_c_file, "w");
    if(fp == NULL) {
        fprintf(stderr, "Cannot write output %s\n",
                vector_c_file);
        clFree(helper.command_queues[0], c); abort();
    }
    fwrite(c, vector_size, sizeof(float), fp);
    fclose(fp);
    clFree(helper.command_queues[0], c);
    clReleaseHelpers();
