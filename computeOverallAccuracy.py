import argparse
import csv
import pandas as pd
import math
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    return parser.parse_args()

def process_data(df):
    col_to_result = {}
    for col in df.columns:
        if col in {"concept", "test-count"}:
            continue

        total_concepts = 0
        total_counts = sum(df["test-count"].astype(int))

        macro_average_total = []
        micro_average_total = []

        for i, row in df.iterrows():
            if not math.isnan(row[col]):
                total_concepts += 1
                macro_average_total.append(row[col])
                micro_weight = row['test-count'] / total_counts
                micro_average_total.append(row[col] * micro_weight)

        if total_concepts > 0:
            macro_average = sum(macro_average_total) / total_concepts
            micro_average = sum(micro_average_total)
            col_to_result[col] = micro_average
            print(f'{col} Macro: {macro_average}')
            print(f'{col} Micro: {micro_average}')
    
    return col_to_result

def update_output_file(output_file, language, col_to_result):
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        df = pd.DataFrame(columns=[
            'language', 'frequency', 'GPT4', 
            'GPT4-NoRules', 'Llama-3-8b-GPT4Rules', 'Llama-3-8b', 
            'Llama-3-8b-NoRules', 'Gemma-7b-GPT4Rules', 'Gemma-7b', 
            'Gemma-7b-NoRules', 'MADLAD400-10b', 'NLLB-200-3.3b'
        ])

    for col, result in col_to_result.items():
        if col not in df.columns:
            continue

        found = df[df['language'] == language]
        if not found.empty:
            df.loc[df['language'] == language, col] = result
        else:
            new_row = {col: result, 'language': language}
            new_row_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_row_df], ignore_index=True)
            print(f"Added new row for {language}")

    df.to_csv(output_file, index=False)

def main():
    args = parse_arguments()
    df = pd.read_csv(args.results_file)
    
    col_to_result = process_data(df)
    update_output_file(args.output_file, args.language, col_to_result)

if __name__ == "__main__":
    main()

    