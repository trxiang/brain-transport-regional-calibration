#!/bin/bash
# Q1: generate truths, then run every case (local machine, single-threaded BLAS per process).
: "${BT_DATA:?set BT_DATA to the data package folder (see README)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
LOG=../results/nd/logs
printf "base\nr0\nvinje\n" | xargs -P 3 -I{} sh -c 'python3 q1_error_budget.py truths {} > '$LOG'/truth_{}.log 2>&1'
python3 -c "import q1_error_budget as q; print('\n'.join(k for k in q.TRUTHS if k.startswith('flow_')))" \
  | xargs -P 4 -I{} sh -c 'python3 q1_error_budget.py truths {} > '$LOG'/truth_{}.log 2>&1'
python3 q1_error_budget.py list | grep -v '^A0_perfect$' \
  | xargs -P 10 -I{} sh -c 'python3 q1_error_budget.py {} > '$LOG'/q1_{}.log 2>&1'
echo ALL_DONE
