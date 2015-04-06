#!/bin/bash

check_guard()
{
    HEADER=$1
    NAME=`basename $HEADER`
    DIR=`dirname $HEADER`
    GUARDS=`grep "#define" $HEADER | cut -f2 -d" "`

    for GUARD in $GUARDS
    do
        i=0
        if [ -n "$GUARD" ]
        then
            RESULTS=`grep "#ifndef $GUARD" $HEADER`
            if [ -n "$RESULTS" ]
            then
                CORRECT_DIR=${DIR/src\//}
                CORRECT_DIR=${CORRECT_DIR//\//_}
                CORRECT_DIR=${CORRECT_DIR^^[a-z]}
                CORRECT_NAME=`echo $NAME | cut -f 1 -d "."`
                CORRECT_NAME=${CORRECT_NAME//-/_}
                CORRECT_NAME=${CORRECT_NAME^^[a-z]}
                CORRECT="GMAC_${CORRECT_DIR}_${CORRECT_NAME}_H_"
                
                if [ $CORRECT != $GUARD ]
                then
                    echo "Bad guard. Current: $GUARD Correct: $CORRECT"
                fi
                i=$(( i + 1 ))
            fi
        fi
        if [ $i -gt 1 ]
        then
            echo Script internal error
            exit
        fi
    done
}

check_license()
{
    HEADER=$1
    LICENSE=`grep "Permission is hereby granted, free of charge" $HEADER`
    if [ ! -n "$LICENSE" ]
    then
        echo "Bad license"
    fi
}

SRC_HEADERS=`find src/ -name *.h` 

for HEADER in $SRC_HEADERS
do
    GUARD_ERROR_STRING=`check_guard $HEADER`
    LICENSE_ERROR_STRING=`check_license $HEADER`

    if [ -n "$GUARD_ERROR_STRING" -o -n  "$LICENSE_ERROR_STRING" ]
    then
        echo "Error in file $HEADER"
        
        if [ -n "$GUARD_ERROR_STRING" ]
        then
            echo "$GUARD_ERROR_STRING"
        fi
        if [ -n "$LICENSE_ERROR_STRING" ]
        then
            echo "$LICENSE_ERROR_STRING"
        fi
    fi
done

TEST_HEADERS=`find "tests" -name *.h` 

for HEADER in $TEST_HEADERS
do
    LICENSE_ERROR_STRING=`check_license $HEADER`

    if [ -n  "$LICENSE_ERROR_STRING" ]
    then
        echo "Error in file $HEADER"
        echo "$LICENSE_ERROR_STRING"
    fi
done


# vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab:
