import random
import argparse
import os
import json
import re
import operator
import pprint

import pandas as pd
import openai
from openai import OpenAI
from dotenv import load_dotenv
from Levenshtein import ratio
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate model on test dataset.')
    parser.add_argument("--model", type=str, choices=['gpt-4-turbo', 'llama-3-8b', 'gemma-7b'], default="gpt-4-turbo",
                        help='Model to use for evaluation.')
    parser.add_argument("--target_language", type=str, help='Target language for translation.')
    parser.add_argument("--max_retries", type=int, default=1, help='Maximum number of retries for generating output.')
    parser.add_argument("--test_dataset", type=str, help='Path to test dataset CSV file.')
    parser.add_argument("--out_file", type=str, help='Path to output CSV file.')
    parser.add_argument("--eval_mode", type=str, choices=['lexical_rules', 'baseline'], default='baseline',
                        help='Evaluation mode.')
    parser.add_argument("--seed", type=int, default=1, help='Random seed for shuffling variations.')
    parser.add_argument("--lexical_rules", type=str, default=None, help='Path to lexical rules CSV file.')
    parser.add_argument("--results_file", type=str, default=None, help='Path to results CSV file.')
    parser.add_argument("--results_file_column", type=str, choices=['GPT4','GPT4-NoRules','Llama-3-8b-GPT4Rules','Llama-3-8b', 'Llama-3-8b-NoRules','Gemma-7b-GPT4Rules','Gemma-7b', 'Gemma-7b-NoRules'], help='Column name in results file to store results.')
    parser.add_argument("--shuffle_order", action='store_true', help='Whether to shuffle the order of variations.')
    args = parser.parse_args()
    return args

def load_model(args):
    """
    Load the specified model and return the model, tokenizer, terminators, and client (if applicable).
    """
    model = None
    tokenizer = None
    terminators = None
    client = None
    
    if args.model == "llama-3-8b":
        # Load Llama 3-8b model
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
    elif args.model == "gemma-7b":
        # Load Gemma 7b model
        model_id = "google/gemma-1.1-7b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        terminators = [tokenizer.eos_token_id]
        
    elif args.model == "gpt-4-turbo":
        # Load GPT-4 Turbo model using OpenAI API
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI()
        
    return model, tokenizer, terminators, client

def load_lexical_rules(lexical_rules_path):
    """
    Load lexical rules from CSV file and return a dictionary mapping concepts to rules.
    """
    concept_to_rules = {}
    generated_rules = pd.read_csv(lexical_rules_path)
    for i in range(len(generated_rules)):
        concept = generated_rules["concept"][i]
        rules = generated_rules["lexical_rules"][i]
        concept_to_rules[concept] = rules
    return concept_to_rules

def construct_baseline_prompt(concept, source_text, variations):
    """
    Construct the prompt for baseline evaluation mode.
    """
    prompt = (f'Please select the best translation of "{concept}" in "{source_text}" '
              f'from the following list: {variations}. Carefully explain your reasoning first '
              f'and then enclose your final answer like this ```answer```.')
    msg = [{'role': 'user', 'content': prompt}]
    return msg

def construct_lexical_rules_prompt(concept, source_text, variations, target_language, concept_to_rules, model_name):
    """
    Construct the prompt for lexical_rules evaluation mode.
    """
    rules = concept_to_rules[concept]
    system_prompt = f'Here are rules for how to translate "{concept}" in {target_language}:{rules}'
    user_prompt = (f'Based on the provided rules, please select the best translation of "{concept}" '
                   f'in "{source_text}" from the following list: {variations}. Carefully explain your reasoning first '
                   f'and then enclose your final answer like this ```answer```.')
    if model_name == "gemma-7b":
        # Concatenate system and user prompt for models that do not support system prompts
        prompt = f'{system_prompt}\n{user_prompt}'
        msg = [{'role': 'user', 'content': prompt}]
    else:
        msg = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
    return msg

def generate_output(args, msg, model, tokenizer, terminators, client):
    """
    Generate output from the model given the message.
    """
    output = ""
    if args.model in ["llama-3-8b", "gemma-7b"]:
        # Generate output using the specified model
        input_ids = tokenizer.apply_chat_template(
            msg,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=False
        )
        output_ids = outputs[0][input_ids.shape[-1]:]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
    elif args.model == "gpt-4-turbo":
        # Generate output using OpenAI API
        response = client.chat.completions.create(
            model=args.model,
            messages=msg,
            temperature=0,
            n=1,
            max_tokens=1024)
        output = response.choices[0].message.content
    return output

def compute_prediction(prediction, label):
    """
    Compare prediction with label to determine correctness.
    """
    predictions = prediction.split("/")
    labels = label.split("/")
    for pred in predictions:
        for gold in labels:
            if pred.strip() == gold.strip():
                print(prediction, label, "correct")
                return 1
    print(prediction, label, "wrong")
    return 0

def process_output(output, variations_list, label):
    """
    Process the output to extract the prediction and evaluate correctness.
    """
    # Ensure output contains exactly two occurrences of '```'
    if len(re.findall('```', output)) != 2:
        raise ValueError("Please enclose your selected translation with 3 back ticks.")
    
    # Extract the content between backticks
    preds = output.split('```')[1]
    if preds.strip() == "answer":
        raise ValueError("Please enclose your selected translation with 3 back ticks.")
    
    preds = preds.replace("answer", "")
    preds = preds.split('/')
    
    # Compute Levenshtein ratios between predictions and variations
    variations_ratio = {}
    for variation in variations_list:
        variations_ratio[variation] = 0
        translations = variation.split('/')
        for translation in translations:
            for pred in preds:
                lev_ratio = ratio(pred.strip(), translation.strip())
                if lev_ratio > variations_ratio[variation]:
                    variations_ratio[variation] = lev_ratio
    
    # Select the variation with the highest ratio
    prediction, max_ratio = max(variations_ratio.items(), key=operator.itemgetter(1))
    
    # Check if the max_ratio is above threshold
    if max_ratio < 0.7:
        correct = 0
        prediction = None
    else:
        correct = compute_prediction(prediction, label)
    return correct, prediction

def process_example(i, row, args, model, tokenizer, terminators, client, concept_to_rules):
    """
    Process a single example from the dataset.
    """
    source_text = row["source language text"]
    label = row["label"]
    concept = row["concept"]
    variations_str = row["variations"]
    variations_list = json.loads(variations_str.replace("'", '"'))
    
    # Shuffle variations list if shuffle_order is True
    if args.shuffle_order:
        random.seed(args.seed * i)  # Ensures reproducibility
        random.shuffle(variations_list)
    else:
        random.seed(args.seed * i)
        random.shuffle(variations_list)
    
    variations = str(variations_list)
    
    # Construct the prompt based on eval_mode
    if args.eval_mode == "baseline":
        msg = construct_baseline_prompt(concept, source_text, variations)
    elif args.eval_mode == "lexical_rules":
        msg = construct_lexical_rules_prompt(concept, source_text, variations, args.target_language, concept_to_rules, args.model)
    
    # Generate output using the model, with retries
    for j in range(args.max_retries):
        try:
            output = generate_output(args, msg, model, tokenizer, terminators, client)
            # Process the output to extract the prediction
            correct, prediction = process_output(output, variations_list, label)
            break  # Break out of the retries loop
        except Exception as e:
            if j == args.max_retries - 1:
                correct = 0
                prediction = None
            msg[0]['content'] += f" {e}"
            correct = 0
            prediction = None
    return correct, prediction

def update_results_file(args, concept_list, concept_accuracy):
    """
    Update the results CSV file with the new accuracy scores.
    """
    if os.path.exists(args.results_file):
        df = pd.read_csv(args.results_file)
    else:
        # Initialize dataframe with appropriate columns
        columns = ['concept','GPT4','GPT4-NoRules','Llama-3-8b-GPT4Rules','Llama-3-8b', 'Llama-3-8b-NoRules','Gemma-7b-GPT4Rules','Gemma-7b', 'Gemma-7b-NoRules','MADLAD400-10b','NLLB-200-3.3b','test-count']
        df = pd.DataFrame(columns=columns)
    for concept in concept_list:
        found = False
        for idx in range(len(df)):
            concept_in_df = df.loc[idx, 'concept']
            concept_no_pos = concept_in_df.split('_')[0] if '_' in concept_in_df else concept_in_df
            if (concept == df.loc[idx, 'concept'] or concept == concept_no_pos) and (concept_accuracy[concept]["total"] != 0):
                df.at[idx, args.results_file_column] = concept_accuracy[concept]["accuracy"]
                found = True
                break
        if not found:
            new_row = {col: None for col in df.columns}
            new_row[args.results_file_column] = concept_accuracy[concept]["accuracy"]
            new_row["concept"] = concept
            new_row["test-count"] = concept_accuracy[concept]["total"]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"Added new row for concept: {concept}")
    df.to_csv(args.results_file, index=False)

def evaluate(args):
    """
    Main evaluation function.
    """
    # Load the test dataset
    dataset = pd.read_csv(args.test_dataset)
    
    # Get the set of concepts from the dataset
    concept_list = set(dataset["concept"])
    
    # Load the specified model
    model, tokenizer, terminators, client = load_model(args)
    
    # Load lexical rules if in lexical_rules evaluation mode
    concept_to_rules = None
    if args.lexical_rules and args.eval_mode == "lexical_rules":
        concept_to_rules = load_lexical_rules(args.lexical_rules)
        
    # Initialize predictions list
    if os.path.exists(args.out_file):
        predictions = pd.read_csv(args.out_file)["predictions"]
    else:
        predictions = [""] * len(dataset)
        # Create output directory if it does not exist
        dir_path = os.path.dirname(args.out_file)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
    # Initialize variables for tracking accuracy
    overall_total = 0
    overall_correct = 0
    not_found = 0
    concept_accuracy = {concept: {"total": 0, "correct": 0, "accuracy": 0} for concept in concept_list}
    
    # Iterate over each example in the dataset
    for i, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        # Process each row
        correct, prediction = process_example(i, row, args, model, tokenizer, terminators, client, concept_to_rules)
        
        # Update predictions and accuracy
        predictions[i] = prediction
        concept = row["concept"]
        concept_accuracy[concept]["correct"] += correct
        concept_accuracy[concept]["total"] += 1
        concept_accuracy[concept]["accuracy"] = round(concept_accuracy[concept]["correct"]/concept_accuracy[concept]["total"], 3)
        overall_correct += correct
        overall_total += 1
        print(f"Accuracy: {overall_correct/overall_total}")
        
        if correct == 0 and prediction is None:
            not_found += 1
    
    # Update results file if specified
    if args.results_file:
        update_results_file(args, concept_list, concept_accuracy)
    
    # Save predictions to output file
    dataset["predictions"] = predictions
    dataset.to_csv(args.out_file, index=False)

def main():
    """
    Main function to parse arguments and run evaluation.
    """
    # Parse command-line arguments
    args = parse_arguments()
        
    # Capitalize target language
    if args.target_language:
        args.target_language = args.target_language.capitalize()
    
    # Evaluate the model
    evaluate(args)

if __name__ == "__main__":
    main()