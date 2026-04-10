#!/bin/bash

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="../../venv/bin/python3.11"
DATASET_DIR="./dataset"


MODELS=(
    "GPR.py"
    #"GPR_rbf.py"
    #"Kernel_SVR.py"
    #"SVR.py"
    #"KNN_regressor.py"
    #"Ridge.py"
    #"ElasticNet.py"
    #"XGBoost.py"
    #"vae_enc_reg_head.py"
)

# --- TROVA TUTTI I CSV NELLA CARTELLA DATASET ---
DATASET_FILES=($(ls ${DATASET_DIR}/*.csv))

TOTAL=$(( ${#MODELS[@]} * ${#DATASET_FILES[@]} ))
CURRENT=0
SUCCESS=0
FAILED=0

echo "============================================"
echo "  Dataset trovati: ${#DATASET_FILES[@]}"
for D in "${DATASET_FILES[@]}"; do
    echo "    - $(basename $D)"
done
echo "  Modelli: ${#MODELS[@]}"
echo "  Totale esperimenti: $TOTAL"
echo "============================================"

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASET_FILES[@]}"; do

        #label da nomefile
        FILENAME=$(basename "$DATASET" .csv)
        if [ "$FILENAME" = "dataset_all" ]; then
            LABEL="clean"
        else
            LABEL="${FILENAME#dataset_all_}"
        fi

        CURRENT=$((CURRENT + 1))
        MODEL_NAME="${MODEL%.py}"
        EXP_NAME="${MODEL_NAME}_${LABEL}"
        LOG_FILE="${LOG_DIR}/${EXP_NAME}_log.txt"

        echo ""
        echo "[$CURRENT/$TOTAL] $EXP_NAME"
        echo "  Modello: $MODEL"
        echo "  Dataset: $(basename $DATASET)"

        $PYTHON "$MODEL" "$DATASET" "$LABEL" > "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            echo "  ✅ OK"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "  ❌ FAILED (vedi $LOG_FILE)"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo ""
echo "============================================"
echo "  FINITO!"
echo "  Success: $SUCCESS / $TOTAL"
echo "  Failed:  $FAILED"
echo "============================================"
