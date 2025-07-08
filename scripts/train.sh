#!/bin/bash

EXP_NAME=CrossSegExp1
MLFLOW_DB_URI=/
SEED=42
EPOCHS=100
DEVICES=(0 1 2 3)
GRAD_ACCUM=1

python ../main.py --train --accumulate_grad_batches $GRAD_ACCUM \
                  --epochs $EPOCHS --devices "${DEVICES[@]}" \
                  --exp_name $EXP_NAME --mlflow_db_uri $MLFLOW_DB_URI \
                  --seed $SEED