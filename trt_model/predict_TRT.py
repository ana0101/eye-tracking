import re
from transformers import AutoTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
import joblib

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCALER_PATH = os.path.join(BASE_DIR, 'trt_model', 'scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'trt_model', 'best_regression_model.pth')

from trt_model.word_properties import *
from trt_model.regression_model import *

def get_words_dict_from_input_text(input_text):
    """
    This function takes an input text and returns a dictionary with words as keys and sentences and other things as values (similar to words_dict).
    """
    words_dict = {}
    # Split the input text into sentences
    sentences = sentences = re.split(r'(?<=[.!?:])\s+|(?<=\n)\s*|(?<=\.\”)', input_text)
    # Go through each sentence
    word_id = 0
    for sentence in sentences:
        # Split the sentence into words
        words = sentence.split()
        # Go through each word
        word_idx = 0
        for word in words:
            words_dict[word_id] = {
                'word': word,
                'sentence': sentence,
                'word_indx_in_sentence': word_idx,
                'properties': WordProperties(word=word, sentence=sentence, word_idx_in_sentence=word_idx, properties_dir=None)
            }
            word_idx += 1
            word_id += 1
    return words_dict


def compute_properties_for_input_words(words_dict):
    """
    This function computes the properties for each word in the words_dict.
    """
    for word_id, word_info in words_dict.items():
        words_dict[word_id]['properties'].compute_properties()

    # Compute sentence-level properties (transformer embeddings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
    model = BertModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
    model.to(device)

    sentences = [word_info['sentence'] for word_info in words_dict.values()]
    seen = set()
    unique_sentences = [s for s in sentences if not (s in seen or seen.add(s))]
    sentences = unique_sentences
    word_ids = list(words_dict.keys())

    model.eval()
    with torch.no_grad():
        encoded_inputs = tokenizer(sentences, return_tensors='pt', padding=True, return_offsets_mapping=True, truncation=False)
        offsets = encoded_inputs.pop("offset_mapping")
        outputs = model(**encoded_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        embeddings_first_layer = hidden_states[1]

        embeddings_middle_layer = hidden_states[6]
        embeddings_last_layer = hidden_states[-1]
        embeddings_avg = torch.mean(torch.stack(hidden_states), dim=0)

        bert_embeddings_first_layer = transformer_embedding(word_ids=word_ids, embeddings=embeddings_first_layer, offsets=offsets)
        bert_embeddings_middle_layer = transformer_embedding(word_ids=word_ids, embeddings=embeddings_middle_layer, offsets=offsets)
        bert_embeddings_last_layer = transformer_embedding(word_ids=word_ids, embeddings=embeddings_last_layer, offsets=offsets)
        bert_embeddings_avg = transformer_embedding(word_ids=word_ids, embeddings=embeddings_avg, offsets=offsets)

    for word_id, word_info in words_dict.items():
        words_dict[word_id]['properties'].transformer_embedding_first_layer = bert_embeddings_first_layer[word_id]
        words_dict[word_id]['properties'].transformer_embedding_middle_layer = bert_embeddings_middle_layer[word_id]
        words_dict[word_id]['properties'].transformer_embedding_last_layer = bert_embeddings_last_layer[word_id]
        words_dict[word_id]['properties'].transformer_embedding_avg = bert_embeddings_avg[word_id]

    return words_dict


def predict_trt_for_input_words(words_dict):
    """
    This function predicts the TRT for each word in the words_dict.
    """

    # Convert to numpy arrays
    length = np.array([word_info['properties'].length for word_info in words_dict.values()])
    freq = np.array([word_info['properties'].frequency for word_info in words_dict.values()])
    surprisal = np.array([word_info['properties'].surprisal for word_info in words_dict.values()])
    transformer_embedding_avg = np.array([word_info['properties'].transformer_embedding_avg for word_info in words_dict.values()])
    X = np.column_stack((length, freq, surprisal, transformer_embedding_avg))
    
    # Scale the features
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
    
    # Prepare the data for the model
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    X_dataset = TensorDataset(X_tensor)
    X_loader = DataLoader(X_dataset, batch_size=1, shuffle=False)

    # Load the model
    model = RegressionModel(input_dim=X.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH))

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Predict the TRT
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in X_loader:
            batch = batch[0].to(device)
            output = model(batch)
            predictions.append(output.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    predictions = predictions.flatten()

    # Add the predictions to the words_dict
    for i, word_id in enumerate(words_dict.keys()):
        words_dict[word_id]['predicted_trt'] = predictions[i]

    # Return a list of dictionaries with the word, its id and its predicted TRT
    results = []
    for word_id, word_info in words_dict.items():
        results.append({
            'word': word_info['word'],
            'word_id': word_id,	
            'trt': round(word_info['predicted_trt'], 2) if word_info['predicted_trt'] > 0 else 0
        })
    return results
