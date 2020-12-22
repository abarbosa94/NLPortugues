from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (GRU, Attention, Concatenate, Dense,
                                            Embedding, TextVectorization)
from tensorflow.python.keras.models import Model
from transformers import BertTokenizerFast, TFBertModel


def tokenize_encoder(
    input_encoder: np.array, bert_model_name: str, encoder_seq_length: int
):
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    flat_list_encoder = [item for sublist in input_encoder.tolist() for item in sublist]
    tokenized_bert = bert_tokenizer(
        text=flat_list_encoder,
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=encoder_seq_length,
    )
    return tokenized_bert["input_ids"]


def define_decoder_tokenization_layers(
    decoder_train: np.array,
    max_decoder_seq_length: int,
) -> Tuple[TextVectorization, TextVectorization]:
    tokenizer_layer_decoder = TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        ngrams=None,
        output_mode="int",
        output_sequence_length=max_decoder_seq_length,
        name="decoder_vectorizer",
    )
    # Setup tokenizer layer for inference
    tokenizer_layer_decoder_inference = TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        ngrams=None,
        output_mode="int",
        output_sequence_length=None,  # As it is auto-regressive, I cant pad
        name="decoder_vectorizer_inference",
    )

    tokenizer_layer_decoder.adapt(decoder_train)
    tokenizer_layer_decoder_inference.adapt(decoder_train)

    return (
        tokenizer_layer_decoder,
        tokenizer_layer_decoder_inference,
    )


def define_full_model(
    input_text_encoder: Input,
    encoder_model: TFBertModel,
    input_text_decoder: Input,
    dec_emb: Embedding,
    decoder_model: GRU,
    attention_layer: Attention,
    decoder_dense: Dense,
) -> Model:
    # encoder part
    bert_encoder_output = encoder_model(input_text_encoder)

    # Decoder definition
    encoder_pooler_output = bert_encoder_output.pooler_output
    encoder_hidden_output = bert_encoder_output.last_hidden_state
    decoder_output, decoder_h_state = decoder_model(
        dec_emb, initial_state=[encoder_pooler_output]
    )
    attn_context_vector = attention_layer([decoder_output, encoder_hidden_output])
    decoder_concat_input = Concatenate(name="concat_layer")(
        [decoder_output, attn_context_vector]
    )
    dense_outputs = decoder_dense(decoder_output)

    # Define the model
    model = Model(
        [input_text_encoder, input_text_decoder],
        [dense_outputs],
    )
    return model
