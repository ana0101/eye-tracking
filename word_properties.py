from wordfreq import word_frequency
import pyphen

class WordProperties:
    """
    Class to store the properties of a word.
    """

    def __init__(self, word, sentence):
        self.length = 0
        self.frequency = 0
        self.syllables = 0
        self.vowels = 0
        self.consonants = 0
        self.vowel_ratio = 0
        self.synsets = 0
        # the number of hypernyms of the first synset
        self.hypernyms = 0
        # the number of hyponyms of all synsets
        self.hypernyms_all = 0
        # the number of hyponyms of the first synset
        self.hyponyms = 0
        # the number of hyponyms of all synsets
        self.hyponyms_all = 0
        self.syntactic_relations = 0
        self.spacy_embedding = 0
        self.roberta_embedding_first_layer = 0
        self.roberta_embedding_middle_layer = 0
        self.roberta_embedding_last_layer = 0
        self.roberta_embedding_mean = 0
        self.gpt2_surprisal = 0

    def compute_length(self):
        self.length = len(self.word)
        
    def compute_frequency(self):
        self.frequency = word_frequency(self.word, 'en')
        
    def compute_syllables(self):
        pyphenator = pyphen.Pyphen(lang='ro')
        self.syllables = len(pyphenator.inserted(self.word).split('-'))
        
    def compute_vowels(self):
        self.vowels = sum(1 for char in self.word if char in 'aeiouAEIOU')
        
    def compute_consonants(self):
        self.consonants = sum(1 for char in self.word if char.isalpha() and char not in 'aeiouAEIOU')
        
    def compute_vowel_ratio(self):
        self.vowel_ratio = self.vowels / self.length if self.length > 0 else 0