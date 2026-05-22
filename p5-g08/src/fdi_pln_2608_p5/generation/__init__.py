"""Generación de texto e inferencia NER."""

from fdi_pln_2608_p5.generation.generate import generate_text, load_model_and_tokenizer
from fdi_pln_2608_p5.generation.ner_predict import (
    load_ner_model,
    predict_entities_from_file,
    predict_entities_from_text,
    simple_tokenize_words,
)

__all__ = [
    "generate_text",
    "load_model_and_tokenizer",
    "load_ner_model",
    "predict_entities_from_file",
    "predict_entities_from_text",
    "simple_tokenize_words",
]
