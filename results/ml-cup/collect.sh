#! sh
date
for fold in exp*; do grep Chosen -m1 -A1 $fold/ml-cup.txt | tail -n1 | sed 's/^/'$fold'\t/' | awk '{print $NF,$0}'; done | sort | cut -f2- -d' '
