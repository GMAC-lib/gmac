#!/bin/bash

SIZE=$(( 1024 * 1024 ))
MAX=$(( 512 * 1024 * 1024 ))

cd ..

while [ $SIZE -lt $(( $MAX + 1 )) ]
do
    echo "Computing $SIZE"
    GMAC_TYPE=1 GMAC_TRANSFER=$SIZE ./transferBlocks > results/normal-in-$SIZE
    GMAC_TYPE=1 GMAC_TRANSFER=$SIZE GMAC_PAGE_LOCKED=1 ./transferBlocks > results/locked-in-$SIZE
    GMAC_TYPE=2 GMAC_TRANSFER=$SIZE ./transferBlocks > results/normal-out-$SIZE
    GMAC_TYPE=2 GMAC_TRANSFER=$SIZE GMAC_PAGE_LOCKED=1 ./transferBlocks > results/locked-out-$SIZE
    SIZE=$(( $SIZE * 2 ))
done

cd -
