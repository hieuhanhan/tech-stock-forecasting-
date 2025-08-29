#!/usr/bin/env bash
set -euo pipefail

############################################
#                CONFIG                    #
############################################

# --- AWS/S3 ---
AWS_PROFILE="default"
S3_BUCKET="s3://tech-stock-data-2025"
S3_PREFIX="tuning_results/lstm_tier2"
S3_CODE_URI="${S3_BUCKET}/${S3_PREFIX}/code"
S3_RESULTS_URI="${S3_BUCKET}/${S3_PREFIX}/results"
S3_LOGS_URI="${S3_BUCKET}/${S3_PREFIX}/logs"

# --- REPO / LOCAL ---
LOCAL_ROOT="/Users/duybui/Documents/Hieu-Han-Github/Personal-Project-/tech-stock-forecasting"
LOCAL_CODE_DIR="${LOCAL_ROOT}"                      # repo root
LOCAL_RESULTS_DIR="${LOCAL_ROOT}/_tuning_results"

# --- EC2 ---
EC2_HOST="ec2-3-75-218-159.eu-central-1.compute.amazonaws.com"      # TODO: change
EC2_KEY="/Users/duybui/Documents/Hieu-Han-Github/tech-stock-key.pem"        # TODO: change
REMOTE_ROOT="/home/ec2-user/quant_runs"
REMOTE_RUN_ID="$(date +%Y%m%d_%H%M%S)"
REMOTE_WORK="${REMOTE_ROOT}/${REMOTE_RUN_ID}"
REMOTE_PYTHON="python3"
REMOTE_AWS_PROFILE="default"

# --- JOB (Tier-2 LSTM) ---
PY_SCRIPT="tier2_lstm_tuning.py"
FOLDS_JSON="data/processed_folds/final/lstm/lstm_tuning_folds_final_paths.json"
TIER1_BACKBONE_JSON="data/tuning_results/jsons/tier1_lstm_backbone.json"
FEATURES_JSON="data/processed_folds/lstm_feature_columns.json"

RETRAIN_INTERVALS="10,20,42"
POP_SIZE=25
NGEN=20
BO_ITERS=5

TIER2_CSV="data/tuning_results/csv/tier2_lstm.csv"
TIER2_JSON="data/tuning_results/jsons/tier2_lstm.json"

# [Optional] limit folds per run (0 = all)
MAX_FOLDS=2

############################################
#        PRECHECKS (local machine)          #
############################################
command -v aws >/dev/null 2>&1 || { echo "ERROR: aws cli not found"; exit 1; }
command -v ssh >/dev/null 2>&1 || { echo "ERROR: ssh not found"; exit 1; }
mkdir -p "${LOCAL_RESULTS_DIR}"

############################################
#     1) PUSH CODE & DATA TO S3 (LOCAL)    #
############################################
echo "==> Syncing code/data to ${S3_CODE_URI}"
aws --profile "${AWS_PROFILE}" s3 sync "${LOCAL_CODE_DIR}/" "${S3_CODE_URI}/" \
  --exclude ".git/*" \
  --exclude ".venv/*" \
  --exclude "__pycache__/*" \
  --delete

############################################
#     2) RUN REMOTE JOB (ON EC2)           #
############################################
echo "==> Starting remote job on EC2: ${EC2_HOST}"
ssh -i "${EC2_KEY}" -o StrictHostKeyChecking=no "${EC2_HOST}" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail

# ====== REMOTE CONFIG (templated) ======
REMOTE_AWS_PROFILE="{{REMOTE_AWS_PROFILE}}"
S3_CODE_URI="{{S3_CODE_URI}}"
S3_RESULTS_URI="{{S3_RESULTS_URI}}"
S3_LOGS_URI="{{S3_LOGS_URI}}"
REMOTE_WORK="{{REMOTE_WORK}}"
REMOTE_PYTHON="{{REMOTE_PYTHON}}"

PY_SCRIPT="{{PY_SCRIPT}}"
FOLDS_JSON="{{FOLDS_JSON}}"
TIER1_BACKBONE_JSON="{{TIER1_BACKBONE_JSON}}"
FEATURES_JSON="{{FEATURES_JSON}}"

RETRAIN_INTERVALS="{{RETRAIN_INTERVALS}}"
POP_SIZE="{{POP_SIZE}}"
NGEN="{{NGEN}}"
BO_ITERS="{{BO_ITERS}}"

TIER2_CSV="{{TIER2_CSV}}"
TIER2_JSON="{{TIER2_JSON}}"
MAX_FOLDS="{{MAX_FOLDS}}"

echo "==> [EC2] Creating workdir: \${REMOTE_WORK}"
mkdir -p "\${REMOTE_WORK}"
cd "\${REMOTE_WORK}"

echo "==> [EC2] Pulling latest code from S3"
aws --profile "\${REMOTE_AWS_PROFILE}" s3 sync "\${S3_CODE_URI}/" "./" --delete

# # (Optional) venv
# python3 -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt

cat > per_fold_runner.py <<'PYEOF'
import json, os, subprocess, sys

FOLDS_JSON = os.environ["FOLDS_JSON"]
TIER1_BACKBONE_JSON = os.environ["TIER1_BACKBONE_JSON"]
FEATURES_JSON = os.environ.get("FEATURES_JSON", "")
RETRAIN_INTERVALS = os.environ["RETRAIN_INTERVALS"]
POP_SIZE = int(os.environ["POP_SIZE"])
NGEN = int(os.environ["NGEN"])
BO_ITERS = int(os.environ["BO_ITERS"])
TIER2_CSV = os.environ["TIER2_CSV"]
TIER2_JSON = os.environ["TIER2_JSON"]
PY_SCRIPT = os.environ["PY_SCRIPT"]
MAX_FOLDS = int(os.environ.get("MAX_FOLDS","0"))
REMOTE_AWS_PROFILE = os.environ.get("REMOTE_AWS_PROFILE","default")
S3_RESULTS_URI = os.environ["S3_RESULTS_URI"]

with open(FOLDS_JSON, "r") as f:
    folds = json.load(f)

if isinstance(folds, dict) and "selected_folds" in folds:
    folds = folds["selected_folds"]
elif not isinstance(folds, list):
    raise SystemExit("Unexpected folds JSON format")

done = 0
for rec in folds:
    gid = rec.get("global_fold_id") or rec.get("fold_id")
    if gid is None:
        continue

    tmp_json = f"_tmp_fold_{gid}.json"
    with open(tmp_json, "w") as g:
        json.dump([rec], g)

    cmd = [
        sys.executable, PY_SCRIPT,
        "--folds-path", tmp_json,
        "--tier1-backbone-json", TIER1_BACKBONE_JSON,
        "--feature-path", FEATURES_JSON if FEATURES_JSON else "data/processed_folds/lstm_feature_columns.json",
        "--retrain-intervals", RETRAIN_INTERVALS,
        "--pop-size", str(POP_SIZE),
        "--ngen", str(NGEN),
        "--bo-iters", str(BO_ITERS),
        "--tier2-json", TIER2_JSON,
        "--tier2-csv", TIER2_CSV,
        "--seed", "42",
        "--skip-existing"
    ]
    print(f"[RUN] Fold {gid} -> {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Fold {gid} failed: {e}", flush=True)

    # sync kết quả sau mỗi fold (CSV/JSON)
    subprocess.run([
        "aws","--profile", REMOTE_AWS_PROFILE, "s3","sync",
        "data/tuning_results", f"{S3_RESULTS_URI}/data_tuning_results",
        "--exclude","*","--include","*.csv","--include","*.json"
    ], check=False)

    done += 1
    if MAX_FOLDS > 0 and done >= MAX_FOLDS:
        break
PYEOF

echo "==> [EC2] Running per-fold runner"
REMOTE_LOG="remote_run_\$(date +%Y%m%d_%H%M%S).log"
( REMOTE_AWS_PROFILE="\${REMOTE_AWS_PROFILE}" \
  S3_RESULTS_URI="\${S3_RESULTS_URI}" \
  FOLDS_JSON="\${FOLDS_JSON}" \
  TIER1_BACKBONE_JSON="\${TIER1_BACKBONE_JSON}" \
  FEATURES_JSON="\${FEATURES_JSON}" \
  RETRAIN_INTERVALS="\${RETRAIN_INTERVALS}" \
  POP_SIZE="\${POP_SIZE}" \
  NGEN="\${NGEN}" \
  BO_ITERS="\${BO_ITERS}" \
  TIER2_CSV="\${TIER2_CSV}" \
  TIER2_JSON="\${TIER2_JSON}" \
  PY_SCRIPT="\${PY_SCRIPT}" \
  MAX_FOLDS="\${MAX_FOLDS}" \
  "\${REMOTE_PYTHON}" per_fold_runner.py ) 2>&1 | tee "\${REMOTE_LOG}"

aws --profile "\${REMOTE_AWS_PROFILE}" s3 cp "\${REMOTE_LOG}" "\${S3_LOGS_URI}/\${REMOTE_LOG}"
echo "==> [EC2] Done. Logs: \${S3_LOGS_URI}/\${REMOTE_LOG}"
REMOTE_SCRIPT

############################################
#  3) PULL RESULTS BACK TO LOCAL (NEW)     #
############################################
echo "==> Pulling results back to LOCAL: ${LOCAL_RESULTS_DIR}"
mkdir -p "${LOCAL_RESULTS_DIR}"
aws --profile "${AWS_PROFILE}" s3 sync "${S3_RESULTS_URI}/data_tuning_results" "${LOCAL_RESULTS_DIR}/data_tuning_results" \
  --exclude "*" --include "*.csv" --include "*.json"

aws --profile "${AWS_PROFILE}" s3 sync "${S3_LOGS_URI}/" "${LOCAL_RESULTS_DIR}/logs" \
  --exclude "*" --include "*.log"

echo "==> Done. Results are in: ${LOCAL_RESULTS_DIR}"