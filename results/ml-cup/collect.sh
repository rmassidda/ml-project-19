#! sh

echo '----------------------------------' >> comment.txt
date >> comment.txt
for fold in exp*; do grep Chosen -A1 $fold/ml-cup.txt | tail -n1 | sed 's/^/'$fold'\t/'; done | sort --field-separator='}' -k2 >> comment.txt
