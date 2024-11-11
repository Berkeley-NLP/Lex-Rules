LANGS=("afrikaans" "tamil" "telugu" "galician" "hindi" "armenian" "japanese" "farsi" "latvian")
LANG_IDS=("en-af" "en-ta" "en-te" "en-gl" "en-hi" "en-hy" "en-ja" "en-fa" "en-lv")
for (( i=0; i<9; i++ )); do
    python computeOverallAccuracy.py \
        --results_file="data/${LANG_IDS[$i]}/results.csv" \
        --output_file="accuracy.csv" \
        --language=${LANGS[$i]} 
done