#!/bin/bash

# 1. Set simple variables
S3_BUCKET="s3://tech-stock-data-2025/tuning_results"
CHECKPOINT_FILE="final_moga_gru_results.json"
PYTHON_SCRIPT="run_moga_optimization_gru.py"
LOCAL_DIR="data/tuning_results"

# 2. Read parameters
POP_SIZE=$1
N_GEN=$2
MAX_NEW_FOLDS=$3  

# 3. Check parameters
if [ -z "$POP_SIZE" ] || [ -z "$N_GEN" ]; then
  echo "Usage: ./resume_tuning_gru.sh <POP_SIZE> <N_GEN> [MAX_NEW_FOLDS]"
  exit 1
fi

# 4. Make local dir if not exists
mkdir -p $LOCAL_DIR

# 5. Pull checkpoint from S3
echo "Pulling checkpoint from S3..."
aws s3 cp "$S3_BUCKET/$CHECKPOINT_FILE" "$LOCAL_DIR/$CHECKPOINT_FILE"

# 6. Start tuning
echo "Running $PYTHON_SCRIPT with pop_size=$POP_SIZE, n_gen=$N_GEN, max_new_folds=${MAX_NEW_FOLDS:-ALL}..."
if [ -z "$MAX_NEW_FOLDS" ]; then
  python "$PYTHON_SCRIPT" --pop-size "$POP_SIZE" --n-gen "$N_GEN"
else
  python "$PYTHON_SCRIPT" --pop-size "$POP_SIZE" --n-gen "$N_GEN" --max-new-folds "$MAX_NEW_FOLDS"
fi

# 7. Upload updated results to S3
echo "Uploading updated results to S3..."
aws s3 cp "$LOCAL_DIR/$CHECKPOINT_FILE" "$S3_BUCKET/$CHECKPOINT_FILE"

echo "All done. Checkpoint updated and uploaded."