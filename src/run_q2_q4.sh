#!/bin/bash
# Q2 and Q4 on the local machine: new truths, diffusion-only fits, then flow fits (sparse LU, fewer in parallel).
: "${BT_DATA:?set BT_DATA to the data package folder (see README)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
LOG=../results/nd/logs
python3 -c "
import q2_flow_detect as q2, q1_error_budget as q1, nd_common as nd
S = nd.Setup()
for k in sorted({j.split('__')[0] for j in q2.jobs()}):
    q1.truth_trajectory(S, k)
" > $LOG/q2_truths.log 2>&1
{ python3 q2_flow_detect.py list | grep '__diff__' | sed 's/^/q2_flow_detect.py /';
  python3 q4_real_data.py list | grep '__diff$' | sed 's/^/q4_real_data.py /'; } \
  | xargs -P 10 -I{} sh -c 'set -- {}; python3 $1 $2 > '$LOG'/$(basename $1 .py)_$2.log 2>&1'
{ python3 q2_flow_detect.py list | grep '__flow-' | sed 's/^/q2_flow_detect.py /';
  python3 q4_real_data.py list | grep '__flow-' | sed 's/^/q4_real_data.py /'; } \
  | xargs -P 7 -I{} sh -c 'set -- {}; python3 $1 $2 > '$LOG'/$(basename $1 .py)_$2.log 2>&1'
echo ALL_DONE
