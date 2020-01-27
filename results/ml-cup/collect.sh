#!/bin/bash

echo "----------"
date
echo "----------"

for fold in exp*; do
    file=$fold/ml-cup.txt
    if [ -e $file ]; then
        tsloss=$(grep Chosen -A1 $file | tail -n1 | awk '{print($NF)}' | tr -d '\r')
        vlline=$(grep Chosen -A1 $file | head -n2 | sed 's/^/'$fold'\t/' | tail -n1 | tr -d '\r')
        line="$vlline $tsloss"
        echo $line | awk '{print $(NF-1),$0}'
    fi
done | sort | cut -f2- -d' '