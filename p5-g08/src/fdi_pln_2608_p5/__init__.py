"""Paquete principal de la práctica 5 de PLN."""

from fdi_pln_2608_p5.modules.attention import Attention
from fdi_pln_2608_p5.modules.model import MiniLLM
from fdi_pln_2608_p5.modules.tokenizer import BPETokenizer
from fdi_pln_2608_p5.modules.train import train_model
from fdi_pln_2608_p5.modules.train_ner import train_ner_model
from fdi_pln_2608_p5.modules.generate import generate_text
from fdi_pln_2608_p5.modules.ner_predict import (
    predict_entities_from_file,
    predict_entities_from_text,
)

__all__ = [
    "Attention",
    "MiniLLM",
    "BPETokenizer",
    "train_model",
    "train_ner_model",
    "generate_text",
    "predict_entities_from_file",
    "predict_entities_from_text",
]
