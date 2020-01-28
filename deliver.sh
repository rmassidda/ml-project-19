#! bash

TEAM=cosmas
PACKAGE=${TEAM}_code_$(git rev-parse --short HEAD).zip

rm -r deliver 2&>/dev/null
mkdir -p deliver

# ML-CUP results
cp results/ml-cup/exp_final/$TEAM*.csv deliver/

# Abstract
cat report/report.md| grep abstract -A1 -m1 | tail -n1 >deliver/${TEAM}_abstract.txt

# Code
zip $PACKAGE activation.py grid.py ml-cup.py monks.py network.py screening.py utils.py validation.py requirements.txt data/{ML-CUP19,monks}*
mv $PACKAGE deliver/

# Report
cd report
make
cp report.pdf ../deliver/${TEAM}_report.pdf
