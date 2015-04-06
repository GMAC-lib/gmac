/* Copyright (c) 2009, 2010, 2011 University of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */


#ifndef GMAC_CONFIG_VISIBILITY_H_
#define GMAC_CONFIG_VISIBILITY_H_

/* Generic helper definitions for shared library support */
#if defined _WIN32 || defined __CYGWIN__
#  define GMAC_DLL_IMPORT __declspec(dllimport)
#  define GMAC_DLL_EXPORT __declspec(dllexport)
#  define GMAC_DLL_LOCAL
#  define APICALL __stdcall
#else
#  if __GNUC__ >= 4
#    define GMAC_DLL_IMPORT __attribute__ ((visibility("default")))
#    define GMAC_DLL_EXPORT __attribute__ ((visibility("default")))
#    define GMAC_DLL_LOCAL  __attribute__ ((visibility("hidden")))
#  else
#    define GMAC_DLL_IMPORT
#    define GMAC_DLL_EXPORT
#    define GMAC_DLL_LOCAL
#  endif
#  define APICALL
#endif

/* Now we use the generic helper definitions above to define GMAC_API and
 * GMAC_LOCAL.
 * GMAC_API is used for the public API symbols. It either DLL imports or DLL
 * exports (or does nothing for static build).
 * GMAC_LOCAL is used for non-api symbols.
 */

#if defined(GMAC_DLL)
/* compiled as a non-debug shared library */
#  ifdef GMAC_DLL_EXPORTS           /* defined if we're building the library
                                    * instead of using it                     */
#    define GMAC_API GMAC_DLL_EXPORT
#  else
#    define GMAC_API GMAC_DLL_IMPORT
#  endif  /* GMAC_DLL_EXPORTS */
#  define GMAC_LOCAL GMAC_DLL_LOCAL
#else  /* GMAC_DLL is not defined, meaning we're in a static library */
#  define GMAC_API
#  define GMAC_LOCAL
#endif  /* GMAC_DLL */



#endif  /* GMAC_CONFIG_VISIBILITY_H */
