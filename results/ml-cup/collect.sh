#! sh

fn=comparison.txt
echo '----------------------------------' >> $fn
date >> $fn
for fold in exp*; do grep Chosen -A1 $fold/ml-cup.txt | tail -n1 | sed 's/^/'$fold'\t/'; done | sort --field-separator='}' -k2 >> $fn
