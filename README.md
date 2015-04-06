= GMAC: Global Memory for ACcelerators =

GMAC is a user-level library that implements an Asymmetric Distributed Shared Memory model to be used by CUDA programs. An ADSM model allows CPU code to access data hosted in accelerator (GPU) memory.

GMAC is being developed by the [http://www.gso.ac.upc.edu Operating System Group] at the [http://www.upc.edu Universitat Politecnica de Catalunya] and the [http://impact.crhc.illinois.edu IMPACT Research Group] at the [http://www.illinois.edu University of Illinois].

== Supported Systems ==

  * Any GNU/Linux ia32 and amd64
  * Mac OS X
  * Windows

== Supported Back-ends ==

  * CUDA
  * OpenCL

== Overview ==

Reduce the complexity of your code. GMAC transparently takes care of the consistency and coherency of the data both used in the CPU and in the GPU, so your code gets simpler and cleaner. This means that you only have to use a pointer per allocation to access data both in the CPU and the GPU.

<img src="http://adsm.googlecode.com/files/CUDA.png" width="100%"/>
<img src="http://adsm.googlecode.com/files/GMAC.png" width="100%"/>

== Using GMAC ==
This is a quick-guide to start using GMAC. Please, read the [Main documentation] for *[Examples examples]* and further information. If you need further information, please send an e-mail to the appropriate Google group:
  * [http://groups.google.com/group/adsm-users ADSM Users]: public group (anybody can send e-mails) to ask questions regarding how to use GMAC.
  * [http://groups.google.com/group/adsm-developers ADSM Developers]: restricted group (ask for an invitation if you plan developing code for GMAC) to discuss about the development of new features in GMAC.

If you find a bug or you want to request a new feature, please use [http://code.google.com/p/adsm/issues/list the Issues Tab] and add a new issue filling the necessary fields.

=== Get the library ===
You can download pre-built Debian packages [http://code.google.com/p/adsm/downloads/list here].

=== Get the code ===
You can use the package tar.gz file from [http://code.google.com/p/adsm/downloads/list here] or get the latest source code cloning the Mercurial repository

`hg clone https://adsm.googlecode.com/hg/ adsm`

=== Compile (use your favorite flags) ===
You need to create the configuration scripts and compile the source code. Additionally, you can also install the GMAC library in your system. The new build system is based on CMake, so you need to have CMake installed to compile GMAC.
{{{
cd adsm
mkdir build
cd build
../configure
make all install
}}}
There are many configuration options for the `configure` script. For example, in order to activate the support for OpenCL, you have to use `--enable-opencl`.

=== Use the GMAC [API] ===
These are the guidelines to port your CUDA code to GMAC:

 # Use `#include <gmac/cuda.h>` instead of `#include <cuda.h>`
 # Use `gmacMalloc()` instead of `cudaMalloc()` and `gmacFree()` instead of `cudaFree()`
 # Use the _device pointer_ in your CPU code
 # Comment out any call to `cudaMemcpy()`.
 # Change any call to CUDA for the analogous GMAC call (e.g., `cudaMemcpyToSymbol()` -> `gmacMemcpyToSymbol()`

Your application is ready to use GMAC.

The GMAC API for OpenCL uses the 'ecl' prefix instead of 'gmac'. You can find examples in the [https://code.google.com/p/adsm/source/browse/#hg%2Ftests%2Fsimple%2Fgmac%2Fopencl%2Fcpp tests] folder.

=== Documentation ===
Documentation can be found in our [Main Wiki]
