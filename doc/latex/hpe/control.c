struct thread_work {
    pthread_t tid;
    size_t offset;
}

. . .

    size_t num_gpus, items_per_thread;
    unsigned n, m;
    int ret;
    struct thread_work *threads;

    num_gpus = eclGetNumberOfAccelerators();
    items_per_thread = vector_size / num_gpus;
    if(vector_size % num_gpus) items_per_thread++;

    threads =
        (struct thread_work *)
        malloc(num_gpus * sizeof(struct thread_work));
    if(threads == NULL) {
        fprintf(stderr, "Not enough host memory\n");
        abort();
    }

    /* Spawn the threads */
    for(n = 0; n < num_gpus; n++) {
        threads[n].offset = n * items_per_thread;
        ret = pthread_create(&threads[n].tid, NULL,
                             vector_add, (void *)&threads[n]);
        if(ret != 0) {
            fprintf(stderr, "Error spawing threads\n");
            break;
        }
    }
    /* Wait for the threads */
    for(m = 0; m < n; m++) pthread_join(&threads[m].tid, NULL);

    free(threads);
    if(n != m) return -1;

    return 0;
