#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

workdir=data/julius              # Directory to save temporal files of processing.
train_set="train"
dev_set="dev"
clean=false

. utils/parse_options.sh

# make dir
tempdir="${workdir}/tmp"
if ${clean}; then
    log "Removing the temp dir ${tempdir}"
    rm -rf "${tempdir}"
fi
mkdir -p "${tempdir}"

# copy wav and prepare .txt files
for _set in "${train_set}" "${dev_set}"; do
    log "Prapring for set ${_set}"
    python utils/prepare_julius.py \
        --csv "data/${_set}.pre_julius.csv" \
        --outdir "${tempdir}"
done

# 3. run julius
log "Run Julius"
perl utils/segment_julius.pl "${tempdir}" > "${workdir}/julius.log" 2>&1

