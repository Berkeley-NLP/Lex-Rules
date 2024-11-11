import torch
import numpy as np
import transformers
import accelerate
import pandas as pd
import ast
import os
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from Levenshtein import ratio
import stanza
import operator
import argparse


def main():
    """
    Main function to parse arguments and initiate the evaluation.
    """
    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate translation models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=['madlad400-10b', 'nllb-200-3.3b'],
        default="madlad400-10b",
        help="NMT model to use."
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Path to the test dataset CSV file."
    )
    parser.add_argument(
        "--target_lang_id",
        type=str,
        choices=["af", "ta", "te", "gl", "hi", "hy", "ja", "fa", "lv"],
        required=True,
        help="Target language ID."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Path to the results CSV file."
    )
    parser.add_argument(
        "--column_name",
        type=str,
        choices=['MADLAD400-10b', 'NLLB-200-3.3b'],
        help="Column name for storing results."
    )
    args = parser.parse_args()

    # Start the evaluation process
    evaluate(args)


def load_model_and_tokenizer(model_name, target_lang):
    """
    Load the appropriate translation model and tokenizer based on the model name and target language.

    Parameters:
        model_name (str): The name of the model to load.
        target_lang (str): The target language ID.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        translator: The translator pipeline (only for certain models).
    """
    flores_mapping = {
        "af": "afr_Latn",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "hy": "hye_Armn",
        "gl": "glg_Latn",
        "lv": "lvs_Latn",
        "fa": "pes_Arab",
        "ja": "jpn_Jpan",
        "hi": "hin_Deva"
    }

    if model_name == "madlad400-10b":
        model_id = 'google/madlad400-10b-mt'
        model = T5ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto"
        )
        tokenizer = T5Tokenizer.from_pretrained(model_id)
        return model, tokenizer, None
    elif model_name == "nllb-200-3.3b":
        model_id = "facebook/nllb-200-3.3B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map="auto"
        )
        translator = pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            src_lang='eng_Latn',
            tgt_lang=flores_mapping[target_lang],
            do_sample=False
        )
        return model, tokenizer, translator
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def load_nlp_pipeline(lang_id):
    """
    Initialize the Stanza NLP pipeline for the given language ID.

    Parameters:
        lang_id (str): The language ID for the NLP pipeline.

    Returns:
        nlp: The initialized Stanza NLP pipeline.
    """
    # Download models for the specified language
    stanza.download(lang_id)

    try:
        return stanza.Pipeline(lang=lang_id, processors='tokenize,mwt,pos,lemma')
    except Exception:
        return stanza.Pipeline(lang=lang_id, processors='tokenize,pos,lemma')


def translate_text(model_name, model, tokenizer, translator, source_text, target_lang_id):
    """
    Translate the source text using the specified model.

    Parameters:
        model_name (str): The name of the translation model.
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        translator: The translator pipeline (for models that use a pipeline).
        source_text (str): The source text to translate.
        target_lang_id (str): The target language ID.

    Returns:
        str: The translated text in lowercase.
    """
    if model_name == "madlad400-10b":
        input_text = f'<2{target_lang_id}> {source_text}'
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(input_ids=input_ids, do_sample=False, max_length=1024)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    elif model_name == "nllb-200-3.3b":
        outputs = translator(source_text, max_length=1024)
        return outputs[0]['translation_text'].lower()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def find_exact_match(translated_text, variations):
    """
    Find an exact match between the translated text and any of the variations.

    Parameters:
        translated_text (str): The translated text.
        variations (list): A list of variation strings.

    Returns:
        str or None: The matched variation or None if no exact match is found.
    """
    for variation in variations:
        # Variations can contain multiple translations separated by '/'
        for translation in variation.split('/'):
            if translation in translated_text:
                return variation
    return None


def find_approximate_match(nlp, translated_text, variations):
    """
    Find an approximate match between the translated text and variations using Levenshtein ratio.

    Parameters:
        nlp: The Stanza NLP pipeline.
        translated_text (str): The translated text.
        variations (list): A list of variation strings.

    Returns:
        tuple: (best_matching_variation, max_ratio)
    """
    processed_text = nlp(translated_text)
    variations_ratio = {}

    # Iterate over each variation
    for variation in variations:
        max_variation_ratio = 0
        # Each variation can have multiple translations separated by '/'
        for trans in variation.split('/'):
            # Iterate over all words in the processed text
            for sentence in processed_text.sentences:
                for word in sentence.words:
                    lev_ratio = ratio(word.text.lower(), trans)
                    lemma_lev_ratio = ratio(word.lemma.lower(), trans)
                    max_ratio = max(lev_ratio, lemma_lev_ratio)
                    if max_ratio > max_variation_ratio:
                        max_variation_ratio = max_ratio
        variations_ratio[variation] = max_variation_ratio

    # Find the variation with the maximum ratio
    prediction, max_ratio = max(variations_ratio.items(), key=operator.itemgetter(1))
    return prediction, max_ratio


def compute_prediction(prediction, label):
    """
    Determine if the prediction matches the label.

    Parameters:
        prediction (str): The predicted label(s), possibly containing multiple labels separated by '/'.
        label (str): The true label(s), possibly containing multiple labels separated by '/'.

    Returns:
        int: 1 if there is a match, 0 otherwise.
    """
    predictions = prediction.split("/")
    labels = label.split("/")
    for pred in predictions:
        for gold in labels:
            if pred == gold:
                return 1  # Correct prediction
    return 0  # Incorrect prediction


def update_concept_accuracy(concept_accuracy, concept, correct):
    """
    Update the accuracy statistics for a given concept.

    Parameters:
        concept_accuracy (dict): Dictionary containing accuracy statistics for each concept.
        concept (str): The concept to update.
        correct (int): 1 if the prediction was correct, 0 otherwise.
    """
    concept_data = concept_accuracy[concept]
    concept_data["correct"] += correct
    concept_data["total"] += 1
    concept_data["accuracy"] = round(concept_data["correct"] / concept_data["total"], 3)


def save_results(concept_list, concept_accuracy, results_file, column_name):
    """
    Save the evaluation results to a CSV file.

    Parameters:
        concept_list (set): Set of all concepts.
        concept_accuracy (dict): Dictionary containing accuracy statistics for each concept.
        results_file (str): Path to the results CSV file.
        column_name (str): Column name for storing results.
    """
    if results_file:
        # Check if the results file already exists
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
        else:
            # Create a new DataFrame with predefined columns
            df = pd.DataFrame(columns=[
                'concept', 'GPT4', 'GPT4-NoRules', 'Llama-3-8b-GPT4Rules', 'Llama-3-8b',
                'Llama-3-8b-NoRules', 'Gemma-7b-GPT4Rules', 'Gemma-7b',
                'Gemma-7b-NoRules', 'MADLAD400-10b', 'NLLB-200-3.3b', 'test-count'
            ])

        # Update the DataFrame with new results
        for concept in concept_list:
            concept_acc = concept_accuracy[concept]["accuracy"]
            test_count = concept_accuracy[concept]["total"]
            if concept_acc != 0:
                df = update_or_add_row(df, concept, concept_acc, test_count, column_name)
        df.to_csv(results_file, index=False)


def update_or_add_row(df, concept, accuracy, test_count, column_name):
    """
    Update an existing row or add a new row in the DataFrame for a given concept.

    Parameters:
        df (DataFrame): The results DataFrame.
        concept (str): The concept to update or add.
        accuracy (float): The accuracy for the concept.
        test_count (int): The number of tests for the concept.
        column_name (str): Column name for storing results.

    Returns:
        DataFrame: Updated DataFrame.
    """
    found = False
    for i in df.index:
        concept_in_df = df.at[i, 'concept']
        if concept == concept_in_df or concept == concept_in_df.split("_")[0]:
            df.at[i, column_name] = accuracy
            df.at[i, 'test-count'] = test_count
            found = True
            break
    if not found:
        # Add a new row
        new_row = {
            'concept': concept,
            column_name: accuracy,
            'test-count': test_count
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def evaluate(args):
    """
    Main evaluation function to process the dataset and compute accuracies.

    Parameters:
        args: Parsed command-line arguments.
    """
    # Load the dataset
    dataset = pd.read_csv(args.test_dataset)
    concept_list = set(dataset["concept"])

    # Load models and pipelines
    model, tokenizer, translator = load_model_and_tokenizer(args.model, args.target_lang_id)
    nlp = load_nlp_pipeline(args.target_lang_id)

    # Initialize accuracy statistics
    overall_total = 0
    overall_correct = 0
    not_found = 0
    concept_accuracy = {
        concept: {"total": 0, "correct": 0, "accuracy": 0} for concept in concept_list
    }

    # Iterate over the dataset
    for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        source_text = row["source language text"]
        label = row["label"]
        concept = row["concept"]
        variations = ast.literal_eval(row["variations"])

        # Translate the source text
        translated_text = translate_text(
            args.model, model, tokenizer, translator, source_text, args.target_lang_id
        )

        # Find exact or approximate matches
        prediction = find_exact_match(translated_text, variations)

        if prediction is None:
            prediction, max_ratio = find_approximate_match(nlp, translated_text, variations)
            if max_ratio >= 0.7:
                correct = compute_prediction(prediction, label)
            else:
                not_found += 1
                correct = 0  # No match found
        else:
            correct = compute_prediction(prediction, label)

        # Update accuracy statistics
        update_concept_accuracy(concept_accuracy, concept, correct)
        overall_total += 1
        overall_correct += correct

    # Print overall accuracy
    print(f'Accuracy: {overall_correct / overall_total:.3f}')

    # Save results
    save_results(concept_list, concept_accuracy, args.results_file, args.column_name)


if __name__ == "__main__":
    main()