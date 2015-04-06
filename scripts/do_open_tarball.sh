#!/bin/bash -e
# --------------------------------------------
# do_open_tarball.sh Creates a tarball without the proprietary bits
# --------------------------------------------
# File:   do_open_tarball.sh
# Author: Javier Cabezas <jcabezas in ac upc edu>
#

# Remove svn entries
find . -name .svn -exec rm -Rf {} \;

# Remove OpenCL bits
find . -name opencl -exec rm -Rf {} \; -exec mkdir {} \; -exec touch {}/CMakeLists.txt \;
rm -f CMakeLists-opencl.txt

# Remove Windows bits
find . -name windows -exec rm -Rf {} \; -exec mkdir {} \; -exec touch {}/CMakeLists.txt \;
rm -f CMakeLists-windows.txt

# vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab:
