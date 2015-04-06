#ifndef __UTILS_IPP__
#define __UTILS_IPP__

template<typename T>
T checkError(const T * orig, const T * calc, uint32_t elems, T (*abs_fn)(T))
{
    T err = 0;

    for (uint32_t i = 0; i < elems; i++) {
        err += abs_fn(orig[i] - calc[i]);
    }

    return err;
}

template<typename T>
void vecAdd(T * c, const T * a, const T * b, uint32_t elems)
{
    for (uint32_t i = 0; i < elems; i++) {
        c[i] = a[i] + b[i];
    }
}

#endif
