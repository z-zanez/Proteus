#!/usr/bin/env bash
# run_pp4_tp8.sh
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

vocab_size=40478

# 并行度
mp_deg=8           # Tensor Parallel (TP)
pp_deg=1           # Pipeline Parallel (PP)
dp_deg=1           # Data Parallel 固定为 1

# 集群文件
cluster=clusters/dgx1_v100_1ib/n4_g8.json
bs=32

# 运行模拟
mpirun -np 8 \
python megatron_gpt.py \
    -model gpt \
    -global-bs $bs \
    -n-macro-batch 1 \
    -nlayer 6 \
    -seq-length 256 \
    -hidden-size 256 \
    -nheads 16 \
    -vocab-size $vocab_size \
    -ps dp \
    -mp-deg $mp_deg \
    -pp-deg $pp_deg \
    -zero 0 \
    --no-seq-first \
    -cluster $cluster \
    --profile-iters 1