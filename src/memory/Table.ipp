#ifndef GMAC_MEMORY_TABLE_IPP_
#define GMAC_MEMORY_TABLE_IPP_

namespace __impl { namespace memory  { namespace vm {

template<typename T>
T *
Table<T>::entry(size_t n) const
{
    assertion(n < nEntries);
    return (T *)((addr_t)table[n] & Mask);
}

template<typename T>
Table<T>::Table(size_t nEntries) :
    nEntries(nEntries)
{
    trace("Creating Table with %zd entries (%p)", nEntries, this);

    posix_memalign((void **) &table, getpagesize(), nEntries * sizeof(T *));

    assertion(table != NULL);
    memset(table, 0, nEntries * sizeof(T *));
    trace("Table memory @ %p", table);
}

template<typename T>
Table<T>::~Table()
{
    trace("Cleaning Table with %zd entries (%p) @ %p", nEntries, this, table);
    ::free(table);
}

template<typename T>
inline bool
Table<T>::present(size_t n) const
{
    assertion(n < nEntries);
    return (addr_t)table[n] & Present;
}

template<typename T>
void
Table<T>::create(size_t n, size_t size)
{
    enterFunction(FuncVmAlloc);
    assertion(n < nEntries);
    table[n] = (T *)((addr_t)new T(size) | Present);
    exitFunction();
}

template<typename T>
void
Table<T>::insert(size_t n, void *addr)
{
    assertion(n < nEntries);
    table[n] = (T *)((addr_t)addr | Present);
}

template<typename T>
void
Table<T>::remove(size_t n)
{
    assertion(n < nEntries);
    table[n] = (T *)0;
}

template<typename T>
inline T &
Table<T>::get(size_t n) const
{
    return *entry(n);
}

template<typename T>
inline T *
Table<T>::value(size_t n) const
{
    return entry(n);
}

template<typename T>
inline size_t
Table<T>::size() const
{
    return nEntries;
}


}}}

#endif
