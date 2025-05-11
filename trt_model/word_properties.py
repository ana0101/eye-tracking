from wordfreq import word_frequency
import pyphen
from transformers import AutoTokenizer, AutoModelForMaskedLM
from surprisal import AutoHuggingFaceModel
import torch
import math
import pandas as pd
import os

class WordProperties:
    """
    Class to store the properties of a word.
    """

    def __init__(self, word, sentence, word_idx_in_sentence, properties_dir):
        self.word = word
        self.sentence = sentence
        self.word_idx_in_sentence = word_idx_in_sentence
        self.properties_dir = properties_dir
        self.length = -1
        self.frequency = -1
        self.syllables = -1
        self.vowels = -1
        self.consonants = -1
        self.vowel_ratio = -1
        self.synsets = -1
        # the number of hypernyms of the first synset
        self.hypernyms = -1
        # the number of hyponyms of all synsets
        self.hypernyms_all = -1
        # the number of hyponyms of the first synset
        self.hyponyms = -1
        # the number of hyponyms of all synsets
        self.hyponyms_all = -1
        self.syntactic_relations = -1
        self.spacy_embedding = -1
        self.surprisal = -1
        self.transformer_embedding_first_layer = -1
        self.transformer_embedding_middle_layer = -1
        self.transformer_embedding_last_layer = -1
        self.transformer_embedding_mean = -1
        self.surprisal_2 = -1

    def compute_length(self):
        if self.length == -1:
            self.length = len(self.word)
        
    def compute_frequency(self):
        if self.frequency == -1:
            self.frequency = word_frequency(self.word, 'ro')
        
    def compute_syllables(self):
        if self.syllables == -1:
            pyphenator = pyphen.Pyphen(lang='ro')
            self.syllables = len(pyphenator.inserted(self.word).split('-'))
        
    def compute_vowels(self):
        if self.vowels == -1:
            self.vowels = sum(1 for char in self.word if char in 'aeiouAEIOU')
        
    def compute_consonants(self):
        if self.consonants == -1:
            self.consonants = sum(1 for char in self.word if char.isalpha() and char not in 'aeiouAEIOU')
        
    def compute_vowel_ratio(self):
        if self.vowel_ratio == -1:
            self.vowel_ratio = self.vowels / self.length if self.length > 0 else 0


    def compute_surprisal(self):
        if self.properties_dir is not None:
            # Check if properties_dir contains surprisal.csv file
            if os.path.exists(self.properties_dir + '/surprisal.csv'):
                print("Loading surprisal from CSV file...")
                df = pd.read_csv(self.properties_dir + '/surprisal.csv')
                df = df[df['sentence'] == self.sentence]
                df = df[df['word_index_in_sentence'] == self.word_idx_in_sentence]
                # Check if exactly one row is returned
                if len(df) == 1:
                    self.surprisal = float(df['surprisal'].values[0])
                    return
                else:
                    print(f"Expected one row for sentence '{self.sentence}' and word index {self.word_idx_in_sentence}, but got {len(df)} rows.")
                    return

        if self.surprisal == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
            model = AutoModelForMaskedLM.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1").to(device)

            # Ensure word_idx_in_sentence is an integer
            word_idx_in_sentence = int(self.word_idx_in_sentence)

            # Replace only the occurrence of the word at word_idx_in_sentence with the mask token
            words = self.sentence.split()
            if word_idx_in_sentence < 0 or word_idx_in_sentence >= len(words):
                print(f"Word index {word_idx_in_sentence} is out of bounds for the sentence '{self.sentence}'.")
                return
            words[word_idx_in_sentence] = tokenizer.mask_token  # Replace the target word with the mask token
            masked_context = ' '.join(words)

            # Tokenize the masked context
            inputs = tokenizer(masked_context, return_tensors="pt").to(device)

            # Get the model's predictions for the masked token
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits

            # Find the probability of the original word at the masked position
            mask_token_index = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            word_id = tokenizer.convert_tokens_to_ids(self.word)
            word_prob = torch.softmax(predictions[0, mask_token_index, :], dim=-1)[0, word_id].item()

            # Calculate surprisal
            self.surprisal = -math.log2(word_prob)


    def compute_properties(self):
        self.compute_length()
        self.compute_frequency()
        self.compute_syllables()
        self.compute_vowels()
        self.compute_consonants()
        self.compute_vowel_ratio()
        self.compute_surprisal()


def transformer_embedding(word_ids, embeddings, offsets):
    """
    Compute the transformer embeddings for a list of words based on their tokenized offsets.
    """
    word_embeddings = dict()
    current_word_index = 0 
    current_word_id = word_ids[current_word_index]
    
    for i, offset in enumerate(offsets):
        current_word_embeddings = []
        last_token_end = 0

        for j, (current_token_start, current_token_end) in enumerate(offset):
            if current_token_start == 0 and current_token_end == 0:
                continue

            if current_token_start <= last_token_end:
                current_word_embeddings.append(embeddings[i, j])
            else:
                word_embedding = torch.mean(torch.stack(current_word_embeddings), dim=0)
                word_embeddings[current_word_id] = word_embedding.cpu().detach().numpy()
                current_word_index += 1
                if current_word_index < len(word_ids):
                    current_word_id = word_ids[current_word_index]
                current_word_embeddings = [embeddings[i, j]]
            last_token_end = current_token_end

        word_embedding = torch.mean(torch.stack(current_word_embeddings), dim=0)
        word_embeddings[current_word_id] = word_embedding.cpu().detach().numpy()
        current_word_index += 1
        if current_word_index < len(word_ids):
            current_word_id = word_ids[current_word_index]

    return word_embeddings
    