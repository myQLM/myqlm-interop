#!/usr/bin/env sh

######################################################################
# @author      : Reda Drissi (mohamed-reda.drissi@atos.net)
# @file        : run
# @created     : Wednesday Nov 28, 2018 16:17:32 CET
#
# @description : kind of test
#####################################################################

cwd=`dirname $(readlink -f "$0")`
COMPILER=oqasm2circ
for f in `ls ${cwd}/*qasm`; do
    echo "----------Executing File `basename $f` ----------------"
    $COMPILER $f
    sleep 5
done
