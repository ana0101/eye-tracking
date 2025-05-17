from transformers import pipeline, AutoTokenizer

from trt_interface import *
from simplifier.simplify import *

def simplify_word(idx, sentence, top_k=3):
    candidates = get_word_replacements_2(idx, sentence, top_k=top_k)
    original_trts = estimate_trt(sentence)
    print(f"Original TRTs: {original_trts}")

    # Get the original TRT
    for word_dict in original_trts:
        if word_dict['word_id'] == idx:
            original_trt = word_dict['trt']
            original_word = word_dict['word']
            break

    print(f"Original TRT for word {idx}: {original_trt}")

    # Get the TRTs for the candidates
    candidate_trts = []
    for candidate in candidates:
        modified_sentence = sentence.replace(sentence.split()[idx], candidate)
        modified_trts = estimate_trt(modified_sentence)
        for word_dict in modified_trts:
            if word_dict['word_id'] == idx:
                candidate_trt = word_dict['trt']
                break
        candidate_trts.append(candidate_trt)

    print(f"Candidate TRTs: {candidate_trts}")
    
    # Keep the first candidate that has a lower TRT than the original
    for i, candidate in enumerate(candidates):
        if candidate_trts[i] < original_trt:
            print(f"Selected candidate: {candidate} with TRT: {candidate_trts[i]}")
            return candidate, candidate_trts[i]
    print("No suitable candidate found.")
    return original_word, original_trt
