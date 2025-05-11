import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trt_model.predict_trt import *

def estimate_trt(text):
    words_dict = get_words_dict_from_input_text(text)
    words_dict = compute_properties_for_input_words(words_dict)
    results = predict_trt_for_input_words(words_dict)
    return results
