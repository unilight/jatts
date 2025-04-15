#!/usr/bin/env bash

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

accelerate_conf=conf/single-node-1-gpu.yaml
conf=conf/matcha_tts.mas.v1.yaml

# dataset configuration
# db_root=downloads
db_root=/data/group1/z44476r/Corpora/hi-fi-captain/ja-JP/female
dumpdir=dump                # directory to dump full features

# data preparation related
julius_clean=false
create_histogram=false

# text related setting
token_type="phn"
token_column="phonemes"
g2p=pyopenjtalk         # g2p method.
oov="\<unk\>"           # Out of vocabrary symbol.
blank="\<blank\>"       # CTC blank symbol.
sos_eos="\<sos/eos\>"   # sos and eos symbols.
nlsyms_txt=none         # Non-linguistic symbol list (needed if existing).
cleaner=none            # text cleaner.

# pretrained model related
pretrained_model=           # NOTE(unilight): for future use

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
           
# decoding related setting
outdir=                     # In case not evaluation not executed together with decoding & synthesis stage
voc=PWG                     # vocoder used (GL or PWG)
griffin_lim_iters=64        # number of iterations of Griffin-Lim
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

# evaluation related setting
eval_metrics="mcd sheet asr"

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

train_set="train"
dev_set="dev"
test_set="test"

token_listdir="${dumpdir}/token_list/${train_set}_${token_type}"
if [ "${cleaner}" != none ]; then
    token_listdir+="_${cleaner}"
fi
if [ "${token_type}" = phn ]; then
    token_listdir+="_${g2p}"
fi
token_list="${token_listdir}/tokens.txt"

# ========================== Main stages start from here. ==========================


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data preparation"

    mkdir -p "data"

    log "Making csv files"
    python local/data_prep.py \
        --train_set "${train_set}" \
        --dev_set "${dev_set}" \
        --test_set "${test_set}" \
        --db_root "${db_root}" \
        --outdir "data"
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Feature extraction"

    # extract features
    pids=()
    for name in "${train_set}" "${dev_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/csvs" ] && mkdir -p "${dumpdir}/${name}/csvs"
        [ ! -e "${dumpdir}/${name}/feats" ] && mkdir -p "${dumpdir}/${name}/feats"
        log "Splitting ${name} set"
        python utils/split_csv.py \
            --csv "data/${name}.csv" \
            --n_splits "${n_jobs}" \
            --outdir "${dumpdir}/${name}/csvs"
        log "Feature extraction start. See the progress via ${dumpdir}/${name}/preprocessing.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/preprocessing.JOB.log" \
            preprocess.py \
                --config "${conf}" \
                --csv "${dumpdir}/${name}/csvs/${name}.JOB.csv" \
                --dumpdir "${dumpdir}/${name}/feats" \
                --f0_path "conf/f0.yaml" \
                --verbose "${verbose}"
        log "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && log "$0: ${i} background jobs are failed." && exit 1;
    log "Successfully finished feature extraction."

    utils/combine_csv.py --csv_dir "${dumpdir}/${train_set}/csvs" --out "data/${train_set}_raw_feat.csv"
    utils/combine_csv.py --csv_dir "${dumpdir}/${dev_set}/csvs" --out "data/${dev_set}_raw_feat.csv"

    # calculate statistics for normalization
    log "Statistics computation start. See the progress via ${dumpdir}/${train_set}/compute_statistics.log."
    ${train_cmd} "${dumpdir}/${train_set}/compute_statistics.log" \
        compute_statistics.py \
            --csv "data/${train_set}_raw_feat.csv" \
            --out "${dumpdir}/${train_set}/stats.h5" \
            --verbose "${verbose}"
    log "Successfully finished calculation of statistics."

fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Token list generation"

    ${train_cmd} "${token_listdir}/generate_token_list.log" \
        generate_token_list.py \
            --csv "data/${train_set}.csv" \
            --out "${token_list}" \
            --column "${token_column}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"
    log "Successfully finished token list generation."
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${token_type}_${cleaner}_$(basename ${conf%.*})
else
    expname=${train_set}_${token_type}_${cleaner}_${tag}
fi
expdir=exp/${expname}
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Stage 3: Network training"

    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "${dumpdir}/${train_set}/stats.h5" "${expdir}/"
    cp "${token_list}" "${expdir}/tokens.txt"
    
    log "Training start. See the progress via ${expdir}/train.log."
    VENV_PYTHON="$MAIN_ROOT/tools/venv/bin/python"
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        $VENV_PYTHON -m accelerate.commands.launch --config_file "${accelerate_conf}" ../../../jatts/bin/e2tts_train.py \
            --config "${conf}" \
            --train-csv "data/${train_set}_raw_feat.csv" \
            --dev-csv "data/${dev_set}_raw_feat.csv" \
            --stats "${expdir}/stats.h5" \
            --token-list "${expdir}/tokens.txt" \
            --token-column "${token_column}" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}"
    log "Successfully finished training."
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Stage 4: Network decoding"

    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${test_set}"; do
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        log "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            e2tts_decode.py \
                --csv "data/${name}.csv" \
                --stats "${expdir}/stats.h5" \
                --token-list "${expdir}/tokens.txt" \
                --token-column "${token_column}" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}"
        log "Successfully finished decoding of ${name} set."
    done
    log "Successfully finished decoding."
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "stage 5: Objective Evaluation"

    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    for name in "${test_set}"; do
        wavdir="${outdir}/${name}/wav"
        log "Evaluation start. See the progress via ${outdir}/${name}/evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/evaluation.log" \
            evaluate.py \
                --csv "data/${test_set}.csv" \
                --wavdir "${wavdir}" \
                --f0_path "conf/f0.yaml" \
                --metrics ${eval_metrics}
        grep "INFO: Mean" "${outdir}/${name}/evaluation.log"
    done
fi
