import os
import pandas as pd
import re

from word_fixations import *

def clean_aoi_sentences_files(sentences_path, question_version):
    """
    Cleans the AOI sentences files by removing removing location and char columns and deduplicating words.
    """
    # Go through all files in the sentences_path directory
    for filename in os.listdir(sentences_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(sentences_path, filename)
            df = pd.read_csv(file_path)

            # Remove rows with None values in the 'word' column
            df = df[df['word'].notna()]

            # Remove rows with different question_image_version
            df = df[(df['question_image_version'] == question_version) | (df['question_image_version'].isna())]
            
            # Remove unnecessary columns
            df = df.drop(columns=["char_idx", "char", "top_left_x", "top_left_y", "width", "height", "char_idx_in_line", "line_idx", "word_idx_in_line"])
            
            # Deduplicate words
            df = df.drop_duplicates(subset=['word_idx', 'page'])
            
            # Save the cleaned dataframe
            df.to_csv(file_path, index=False)


def create_aoi_sentences_dict(sentences_path):
    """
    Creates a dictionary with keys as file names and pages, and values a dictionary with keys the word_idx and the word as value.
    """
    aoi_sentences_dict = {}
    
    # Go through all files in the sentences_path directory
    for filename in os.listdir(sentences_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(sentences_path, filename)
            df = pd.read_csv(file_path)

            # Get the file name without the extension
            file_name = os.path.splitext(filename)[0]
            # Remove _aoi from the file name
            file_name = file_name.replace('_aoi', '')

            # Go through all pages in the file
            for page in df['page'].unique():
                # Get the words for that page
                words = df[df['page'] == page]['word'].tolist()
                # Create a dictionary with the word_idx as key and the word as value
                words_dict = {idx: word for idx, word in zip(df[df['page'] == page]['word_idx'], words)}
                # Add to the dictionary
                dict_key = f"{file_name}_{page}"
                aoi_sentences_dict[dict_key] = words_dict
    return aoi_sentences_dict


def create_stimuli_experiment_sentences_dict(stimuli_experiment_path):
    """
    Creates a dictionary with keys as file names and pages, and values as a dictionary with keys text, which contains a string of all words on that page, sentences, which contains a list of sentences.
    """
    stimuli_sentences_dict = {}
    df = pd.read_excel(stimuli_experiment_path)

    # Group py stimulus_id
    grouped = df.groupby('stimulus_id')

    # Go through each group
    for name, group in grouped:
        stimulus_name = group['stimulus_name'].iloc[0]
        stimulus_id = group['stimulus_id'].iloc[0]
        # Go through all page_* columns
        for col in group.columns:
            if col.startswith('page_'):
                dict_key = f"{stimulus_name.lower()}_{stimulus_id}_{col.lower()}"
                # Get the words from the column
                words = group[col].iloc[0]
                # Check if the stimulus has that page
                if pd.isna(words):
                    continue
                # Add to the dictionary
                if dict_key not in stimuli_sentences_dict:
                    stimuli_sentences_dict[dict_key] = {}
                stimuli_sentences_dict[dict_key]['text'] = words
                # Split the words into sentences: \n, ., ?, !
                sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', words)
                # Remove empty sentences
                sentences = [s.strip() for s in sentences if s.strip()]
                # Add the sentences to the dictionary
                stimuli_sentences_dict[dict_key]['sentences'] = sentences
    return stimuli_sentences_dict


def map_words_to_sentences(aoi_sentences_dict, stimuli_sentences_dict):
    """
    Maps the words in the AOI sentences to the sentences in the stimuli experiment.
    """
    # Create a dictionary to store the mapping
    word_to_sentence_mapping = {}

    # Go through each key in the AOI sentences dictionary
    for key, words_dict in aoi_sentences_dict.items():
        sentence_idx = 0
        # Check if the key is in the stimuli sentences dictionary
        if key in stimuli_sentences_dict:
            # Get the sentences from the stimuli sentences dictionary
            sentences = stimuli_sentences_dict[key]['sentences']
            # Add key to the mapping dictionary
            word_to_sentence_mapping[key] = {}
            # Go through each word
            for word_idx, word in words_dict.items():
                found = False
                clean_word = word.strip('.,!?:„”()')
                # If there is any punctuation mark in the word, put None for setence
                if any(p in clean_word for p in ['.', '!', '?', ':', '„', '”']):
                    word_to_sentence_mapping[key][word_idx] = {
                        'word': clean_word,
                        'sentence': None,
                        'sentence_idx': None
                    }
                    continue
                # Search for the word in the current sentence
                # If not found, go to the next sentence
                while sentence_idx < len(sentences) and not found:
                    sentence = sentences[sentence_idx]
                    if clean_word in sentence:
                        # If the word is found, add it to the mapping
                        if key not in word_to_sentence_mapping:
                            word_to_sentence_mapping[key] = {}
                        if word_idx not in word_to_sentence_mapping[key]:
                            word_to_sentence_mapping[key][word_idx] = {}
                            word_to_sentence_mapping[key][word_idx]['word'] = clean_word
                        word_to_sentence_mapping[key][word_idx]['sentence'] = sentence
                        word_to_sentence_mapping[key][word_idx]['sentence_idx'] = sentence_idx
                        found = True
                    else:
                        # If the word is not found, go to the next sentence
                        sentence_idx += 1
                        print(f"Word '{clean_word}' not found in sentence '{sentence}'. Moving to next sentence.")
                # If the word is not found in any sentence, add it to the mapping with None
                if not found:
                    if key not in word_to_sentence_mapping:
                        word_to_sentence_mapping[key] = {}
                    if word_idx not in word_to_sentence_mapping[key]:
                        word_to_sentence_mapping[key][word_idx] = {}
                    word_to_sentence_mapping[key][word_idx]['word'] = clean_word
                    word_to_sentence_mapping[key][word_idx]['sentence'] = None
                    word_to_sentence_mapping[key][word_idx]['sentence_idx'] = None
        
    return word_to_sentence_mapping


def get_word_sentence_fixations_dict_from_csv(csv_path):
    """
    Read the CSV file in word_sentence_fixations and return words_dict.
    """
    df = pd.read_csv(csv_path)
    words_dict = {}

    for _, row in df.iterrows():
        stimulus = row['stimulus']
        subject_id = row['subject_id']
        word_idx = row['word_index']
        word = row['word']
        sentence = row['sentence']
        fixations_list = row['fixations_list']
        fixations_TRT = row['fixations_TRT']

        # Transform fixations_list from string to list of numbers
        fixations_list = [int(x.strip()) for x in fixations_list.strip('[]').split(',') if x != '']

        if stimulus not in words_dict:
            words_dict[stimulus] = {}
        if subject_id not in words_dict[stimulus]:
            words_dict[stimulus][subject_id] = {}
        if word_idx not in words_dict[stimulus][subject_id]:
            words_dict[stimulus][subject_id][word_idx] = {
                'word': word,
                'sentence': sentence,
                'fixations': WordFixations(fixations=fixations_list, TRT=fixations_TRT)
            }

    return words_dict
