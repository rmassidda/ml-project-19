#! bash

TEAM=cosmas
CODE=${TEAM}_code_$(git rev-parse --short HEAD)

rm -r $TEAM.zip deliver 2&>/dev/null
mkdir -p deliver

# ML-CUP results
cp results/ml-cup/exp_final/$TEAM*.csv deliver/

# Abstract
cat report/report.md| grep abstract -A2 -m1 | tail -n2 >deliver/${TEAM}_abstract.txt

# Code
mkdir -p deliver/$CODE
mkdir -p deliver/$CODE/data
mkdir -p deliver/$CODE/data/monks
cp activation.py ml-cup.py monks.py network.py screening.py utils.py validation.py requirements.txt deliver/$CODE
cp data/ML-CUP19* deliver/$CODE/data
cp data/monks/* deliver/$CODE/data/monks

# Report
cd report
make
cp report.pdf ../deliver/${TEAM}_report.pdf

# Package
cd ../deliver
zip -r ../$TEAM.zip *
