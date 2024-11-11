import json
import pandas as pd
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse
from tqdm import tqdm
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate lexical rules for translating concepts.")
    parser.add_argument("--model", type=str, choices=['gpt-4-turbo', 'llama-3-8b', 'gemma-7b'], default="gpt-4-turbo",
                        help="Model to use for generating lexical rules.")
    parser.add_argument("--source_language", type=str, required=True,
                        help="Source language of the concepts.")
    parser.add_argument("--target_language", type=str, required=True,
                        help="Target language for translation.")
    parser.add_argument("--max_retries", type=int, default=10,
                        help="Maximum number of retries for LLM interaction.")
    parser.add_argument("--source_concepts", type=str, required=True,
                        help="Path to CSV file containing source concepts.")
    parser.add_argument("--out_file", type=str, required=True,
                        help="Path to output CSV file.")
    parser.add_argument("--use_incontext_sentences", action="store_true",
                        help="Whether to use in-context sentences.")
    parser.add_argument("--num_sentences", type=int, default=10,
                        help="Number of sentences to use per translation when using in-context sentences.")
    parser.add_argument("--train_dataset", type=str,
                        help="Path to train dataset CSV file.")
    return parser.parse_args()


def initialize_model(args):
    """
    Initialize the model or OpenAI client based on the specified model in arguments.
    """
    if args.model == "llama-3-8b":
        # Initialize LLaMA 3-8B model
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        return {'model': model, 'tokenizer': tokenizer, 'terminators': terminators}
    elif args.model == "gemma-7b":
        # Initialize GEMMA 7B model
        model_id = "google/gemma-1.1-7b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        terminators = [tokenizer.eos_token_id]
        return {'model': model, 'tokenizer': tokenizer, 'terminators': terminators}
    else:
        # Initialize OpenAI client
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        model_name = args.model
        client = OpenAI()
        # Initialize GPT-2 tokenizer for tokenization approximation
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return {'client': client, 'model_name': model_name, 'tokenizer': tokenizer}


def read_source_concepts(file_path):
    """
    Read source concepts from a CSV file.
    """
    source_concepts = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            source_concepts.append(row)
    return source_concepts


def prepare_output_dataframe(source_concepts):
    """
    Prepare a DataFrame to store the output lexical rules and prompts.
    """
    data = {
        "concept": [row[0] for row in source_concepts],
        "variations": [row[2] for row in source_concepts],
        "lexical_rules": [""] * len(source_concepts),
        "prompt": [""] * len(source_concepts)
    }
    return pd.DataFrame(data)


def read_train_dataset(file_path):
    """
    Read the training dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def format_concepts_string(translations):
    """
    Format the list of translations into a string suitable for inclusion in the prompt.
    """
    if len(translations) == 1:
        return f'"{translations[0]}"'
    elif len(translations) == 2:
        return f'"{translations[0]}" and "{translations[1]}"'
    else:
        # For more than two translations, join them with commas and 'and' before the last one
        return ', '.join(f'"{t}"' for t in translations[:-1]) + f', and "{translations[-1]}"'


def top_k_longest_strings(strings, num_sentences, feasible_strings=None):
    """
    Select the top k longest strings from a list of strings, optionally filtering by feasible strings.
    """
    if feasible_strings:
        strings = [string for string in strings if string in feasible_strings]
    sorted_strings = sorted(strings, key=len, reverse=True)
    return sorted_strings[:num_sentences]


def collect_incontext_sentences(source_word, train_dataset, translations, num_sentences, tokenizer):
    """
    Collect representative sentences for each lexical variation.
    """
    # Filter dataset for the current concept
    filtered_dataset = train_dataset[train_dataset['concept'] == source_word]

    # Initialize a dictionary to hold sentences per translation
    sentences_per_translation = {translation: [] for translation in translations}

    # Collect sentences for each translation
    for idx, row in filtered_dataset.iterrows():
        label = row['label']
        if label in sentences_per_translation:
            sentences_per_translation[label].append(row['source language text'])

    # For each translation, select top_k_longest_strings
    for translation in sentences_per_translation:
        candidate_sentences = sentences_per_translation[translation]
        feasible_strings = set()
        for sent in candidate_sentences:
            tokenized_sent = tokenizer(sent, return_tensors='pt').input_ids
            if tokenized_sent.shape[1] <= 50:  # remove texts greater than 50 tokens
                feasible_strings.add(sent)
        sentences_per_translation[translation] = top_k_longest_strings(
            candidate_sentences, num_sentences, feasible_strings=feasible_strings)
    return sentences_per_translation


def generate_prompt(source_word, concepts_string, source_language, target_language, translations, use_incontext_sentences, train_dataset, num_sentences, tokenizer):
    """
    Generate the prompt to be sent to the LLM.
    """
    if use_incontext_sentences:
        sentences_per_translation = collect_incontext_sentences(
            source_word, train_dataset, translations, num_sentences, tokenizer
        )
        prompt = (
            f'When translating the concept "{source_word}" from {source_language} to {target_language}, '
            f'what is the difference in meaning between {concepts_string} and in which contexts should they be used? '
            f'Here are sentences where each word is used in context to help you: {sentences_per_translation}'
        )
    else:
        prompt = (
            f'When translating the concept "{source_word}" from {source_language} to {target_language}, '
            f'what is the difference in meaning between {concepts_string} and in which contexts should they be used?'
        )
    return prompt


def generate_system_prompt(translations, source_language, target_language):
    """
    Generate the system prompt that instructs the LLM on the expected output format.
    """
    return (
        f"Please only return a json with the following keys {translations} and no other text. "
        f"For each key the value should be a string in {source_language} explaining how the meaning "
        f"and usage of that {target_language} word is different from the others. The string should also "
        f"include a brief example in {target_language} of the word being used with an {source_language} translation. "
        f"Please include the transliteration from {target_language} to Latin characters if necessary."
    )


def prepare_messages(system_prompt, prompt, model_name):
    """
    Prepare the messages to be sent to the LLM, handling models that do not support system prompts.
    """
    if model_name == "gemma-7b":  # Gemma does not support system prompt
        full_prompt = f'{system_prompt} {prompt}'
        messages = [{'role': 'user', 'content': full_prompt}]
    else:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]
    return messages


def get_llm_response(messages, args, model_data, max_retries, translations):
    """
    Interact with the LLM to get the response, handling retries and output validation.
    """
    for attempt in range(max_retries):
        try:
            if args.model != "gpt-4-turbo":
                # Using local model
                tokenizer = model_data['tokenizer']
                model = model_data['model']
                terminators = model_data['terminators']
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=2048,
                    eos_token_id=terminators,
                    do_sample=False,
                    temperature=0,
                    no_repeat_ngram_size=6
                )
                output_ids = outputs[0][input_ids.shape[-1]:]
                output = tokenizer.decode(output_ids, skip_special_tokens=True)
                if '```' in output:
                    output = output.split('```')[1]
            else:
                # Using OpenAI API
                client = model_data['client']
                model_name = model_data['model_name']
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    n=1,
                    max_tokens=2048
                )
                output = response.choices[0].message.content
            # Process the output
            if "json" in output:
                output = output.split("json")[1].split("```")[0].strip()
            # Attempt to load the output as JSON
            meaning_dict = json.loads(output)
            # Validate that all translations are present as keys
            assert all(translation in meaning_dict for translation in translations), \
                f"json does not contain keys {translations}"
            assert len(meaning_dict) == len(translations), \
                f"the number of keys {len(meaning_dict)} does not match the number of variations {len(translations)}"
            # If everything is fine, return the output
            print(output)
            return output
        except Exception as e:
            # Append error message to the system prompt for the next attempt
            messages[0]["content"] += f" Please avoid the following error in your output: {e}"
            print(e)
            print('-----------------------------------')
    print(output)
    return output


def main():
    # Parse command-line arguments
    args = parse_arguments()
    args.target_language = args.target_language.capitalize()

    # Initialize the model or OpenAI client
    model_data = initialize_model(args)

    # Read source concepts
    source_concepts = read_source_concepts(args.source_concepts)

    # Prepare the output DataFrame
    output_lexical_rules = prepare_output_dataframe(source_concepts)

    # If in-context sentences are to be used, read the train dataset
    if args.use_incontext_sentences:
        train_dataset = read_train_dataset(args.train_dataset)
    else:
        train_dataset = None

    # For each concept, generate the prompt and get the LLM response
    for i in tqdm(range(len(source_concepts))):
        source_word = source_concepts[i][0]
        variations = source_concepts[i][2]
        translations = variations.split(";")
        concepts_string = format_concepts_string(translations)

        # Generate the prompt
        prompt = generate_prompt(
            source_word, concepts_string, args.source_language,
            args.target_language, translations, args.use_incontext_sentences,
            train_dataset, args.num_sentences, model_data.get('tokenizer')
        )

        # Store the prompt in the DataFrame
        output_lexical_rules.at[i, 'prompt'] = prompt

        # Generate the system prompt
        system_prompt = generate_system_prompt(
            translations, args.source_language, args.target_language
        )

        # Prepare messages for the LLM
        messages = prepare_messages(
            system_prompt, prompt, args.model
        )

        # Get the LLM response
        output = get_llm_response(
            messages, args, model_data, args.max_retries, translations
        )

        # Process the output and store in the DataFrame
        output_lexical_rules.at[i, 'lexical_rules'] = output

        # Save the output to CSV after each iteration
        output_lexical_rules.to_csv(args.out_file, index=False)


if __name__ == "__main__":
    main()
