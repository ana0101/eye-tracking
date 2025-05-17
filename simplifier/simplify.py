import re
from transformers import pipeline, AutoTokenizer

model = 'dumitrescustefan/bert-base-romanian-cased-v1'
tokenizer = AutoTokenizer.from_pretrained(model)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)


def get_word_replacements(word_idx_in_sentence, sentence, top_k=5):
    """
    Replace the N-th word (excluding punctuation) in the sentence with a mask and return top_k replacements.
    """
    # Split sentence into tokens: words and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", sentence, flags=re.UNICODE)

    # Map word index (ignoring punctuation) to token index
    word_count = 0
    target_token_index = None
    for i, token in enumerate(tokens):
        if re.match(r"\w+", token):  # It's a word
            if word_count == word_idx_in_sentence:
                target_token_index = i
                break
            word_count += 1

    if target_token_index is None:
        raise IndexError(f"word_idx_in_sentence {word_idx_in_sentence} is out of range for the sentence.")

    original_word = tokens[target_token_index].lower()
    tokens[target_token_index] = tokenizer.mask_token

    # Rebuild sentence with appropriate spacing
    masked_sentence = ""
    for i, t in enumerate(tokens):
        if i > 0 and re.match(r"\w+|"+re.escape(tokenizer.mask_token), t):
            masked_sentence += " "
        masked_sentence += t

    print("Masked sentence:", masked_sentence)

    # Get predictions and filter
    predictions = fill_mask(masked_sentence, top_k=top_k + 5)
    replacements = []
    for p in predictions:
        cand = p['token_str'].strip()
        if cand.lower() != original_word and cand != tokenizer.mask_token and cand not in replacements:
            replacements.append(cand)
        if len(replacements) >= top_k:
            break

    return replacements

def get_word_replacements_2(word_idx_in_sentence, sentence, top_k=5):
    """
    Uses a sentence pair: [MASKED] [SEP] [ORIGINAL] format.
    """
    tokens = re.findall(r"\w+|[^\w\s]", sentence, flags=re.UNICODE)

    word_count = 0
    target_token_index = None
    for i, token in enumerate(tokens):
        if re.match(r"\w+", token):
            if word_count == word_idx_in_sentence:
                target_token_index = i
                break
            word_count += 1

    if target_token_index is None:
        raise IndexError(f"word_idx_in_sentence {word_idx_in_sentence} is out of range for the sentence.")

    original_word = tokens[target_token_index].lower()
    tokens[target_token_index] = tokenizer.mask_token

    masked_sentence = ""
    for i, t in enumerate(tokens):
        if i > 0 and re.match(r"\w+|"+re.escape(tokenizer.mask_token), t):
            masked_sentence += " "
        masked_sentence += t

    sentence_pair = f"{masked_sentence} {tokenizer.sep_token} {sentence}"
    predictions = fill_mask(sentence_pair, top_k=top_k + 5)

    replacements = []
    for p in predictions:
        candidate = p['token_str'].strip()
        if candidate.lower() != original_word and candidate != tokenizer.mask_token and candidate not in replacements:
            replacements.append(candidate)
        if len(replacements) >= top_k:
            break

    return replacements


def get_word_replacements_3(word_idx_in_sentence, sentence, top_k=5):
    """
    Uses a reversed sentence pair: [ORIGINAL] [SEP] [MASKED].
    """
    tokens = re.findall(r"\w+|[^\w\s]", sentence, flags=re.UNICODE)

    word_count = 0
    target_token_index = None
    for i, token in enumerate(tokens):
        if re.match(r"\w+", token):
            if word_count == word_idx_in_sentence:
                target_token_index = i
                break
            word_count += 1

    if target_token_index is None:
        raise IndexError(f"word_idx_in_sentence {word_idx_in_sentence} is out of range for the sentence.")

    original_word = tokens[target_token_index].lower()
    tokens[target_token_index] = tokenizer.mask_token

    masked_sentence = ""
    for i, t in enumerate(tokens):
        if i > 0 and re.match(r"\w+|"+re.escape(tokenizer.mask_token), t):
            masked_sentence += " "
        masked_sentence += t

    sentence_pair = f"{sentence} {tokenizer.sep_token} {masked_sentence}"
    predictions = fill_mask(sentence_pair, top_k=top_k + 5)

    replacements = []
    for p in predictions:
        candidate = p['token_str'].strip()
        if candidate.lower() != original_word and candidate != tokenizer.mask_token and candidate not in replacements:
            replacements.append(candidate)
        if len(replacements) >= top_k:
            break

    return replacements
    

# sentence = "Ana citește cartea și apoi povestește povestea."
# candidates = get_word_replacements(6, sentence, top_k=5)
# print(candidates)

# sentence = "Ana citește cartea și apoi povestește povestea."
# candidates = get_word_replacements_2(6, sentence, top_k=5)
# print(candidates)

# sentence = "Ana citește cartea și apoi povestește povestea."
# candidates = get_word_replacements_3(6, sentence, top_k=5)
# print(candidates)