python generate_lexical_rules.py \
    --model="gpt-4-turbo" \
    --source_language="English" \
    --target_language="Afrikaans" \
    --source_concepts="data/en-af/af.words" \
    --out_file="lexical_rules/gpt-4-turbo-afrikaans.csv" \
    --use_incontext_sentences \
    --num_sentences=25 \
    --train_dataset="data/en-af/train_dataset.csv" \
    --max_retries=3