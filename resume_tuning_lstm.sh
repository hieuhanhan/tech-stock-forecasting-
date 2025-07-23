#!/bin/bash

# 1. Set simple variable
S3_BUCKET="s3://tech-stock-data-2025/tuning_results"
CHECKPOINT_FILE="final_moga_lstm_results.json"
PYTHON_SCRIPT="run_moga_optimization_lstm.py"
LOCAL_DIR="data/tuning_results"
FOLDS_DIR="data/processed_folds"

# 2. Read parameter
POP_SIZE=$1
N_GEN=$2
MAX_NEW_FOLDS=$3  

# 3. Check parameter
if [ -z "$POP_SIZE" ] || [ -z "$N_GEN" ]; then
  echo "Usage: ./resume_tuning_lstm.sh <POP_SIZE> <N_GEN> [MAX_NEW_FOLDS]"
  exit 1
fi

# 4. Prepare folders and logging
mkdir -p $LOCAL_DIR $FOLDS_DIR logs
START_TIME=$(date)
LOG_FILE="logs/tuning_log_$(date '+%Y%m%d_%H%M%S').log"

exec > "$LOG_FILE" 2>&1

echo "[INFO] Started at: $START_TIME"
echo "[INFO] Pop size: $POP_SIZE | N gen: $N_GEN | Max folds: ${MAX_NEW_FOLDS:-ALL}"

# 5. Download checkpoint file
echo "Pulling checkpoint from S3..."
aws s3 cp "$S3_BUCKET/$CHECKPOINT_FILE" "$LOCAL_DIR/$CHECKPOINT_FILE"

# 5.1 Download processed folds
echo "Pulling processed folds from S3..."
aws s3 sync s3://tech-stock-data-2025/data/processed_folds $FOLDS_DIR
if [ "$(ls -A $FOLDS_DIR)" ]; then
  echo "Fold data successfully synced."
else
  echo "Fold data not found in $FOLDS_DIR. Please check S3 path."
  exit 1
fi

# 6. Start tuning
echo "Found $(find $FOLDS_DIR -name '*.json' | wc -l) fold files."
echo "Running $PYTHON_SCRIPT with pop_size=$POP_SIZE, n_gen=$N_GEN, max_new_folds=${MAX_NEW_FOLDS:-ALL}..."

if [ -z "$MAX_NEW_FOLDS" ]; then
  python "$PYTHON_SCRIPT" --pop-size "$POP_SIZE" --n-gen "$N_GEN"
else
  python "$PYTHON_SCRIPT" --pop-size "$POP_SIZE" --n-gen "$N_GEN" --max-new-folds "$MAX_NEW_FOLDS"
fi

#7. Check files 
PYTHON_EXIT_CODE=$?
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
  echo " Python script failed with exit code $PYTHON_EXIT_CODE"
  exit $PYTHON_EXIT_CODE
fi

# 8. Upload results to S3
echo "Uploading updated results to S3..."
aws s3 cp "$LOCAL_DIR/$CHECKPOINT_FILE" "$S3_BUCKET/$CHECKPOINT_FILE"

if [ $? -eq 0 ]; then
  echo "Upload successful."
else
  echo "Upload failed!"
  exit 1
fi

END_TIME=$(date)
echo "Finished at: $END_TIME"