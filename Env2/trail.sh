#!/bin/bash
# Try a command until it executes successfully.
# Optionnally takes the number of times to try until it is successful.
# Author : Mashi (http://bbs.archlinux.org/viewtopic.php?id=56646)
COUNT=-1
if [[ "$1" =~ ^[0-9]+$ ]]; then
COUNT=$1
shift
fi
STATUS=0
while test $COUNT -ne 0; do
let COUNT-=1
$*
STATUS=$?
if test "$STATUS" -eq 0 ; then
exit $STATUS
fi
done
exit $STATUS