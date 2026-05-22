"""Paquete principal de la práctica 5 de PLN."""

from fdi_pln_2608_p5.generation.generate import generate_text
from fdi_pln_2608_p5.generation.ner_predict import (
    predict_entities_from_file,
    predict_entities_from_text,
)
from fdi_pln_2608_p5.model.attention import Attention
from fdi_pln_2608_p5.model.transformer import MiniLLM
from fdi_pln_2608_p5.tokenizer import BPETokenizer
from fdi_pln_2608_p5.training.train_causal import train_model
from fdi_pln_2608_p5.training.train_ner import train_ner_model

__all__ = [
    "Attention",
    "BPETokenizer",
    "MiniLLM",
    "generate_text",
    "predict_entities_from_file",
    "predict_entities_from_text",
    "train_model",
    "train_ner_model",
]
