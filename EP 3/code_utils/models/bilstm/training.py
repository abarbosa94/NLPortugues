from typing import Tuple

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (LSTM, Attention, Bidirectional,
                                     Concatenate, Dense, Embedding)
from tensorflow.keras.layers.experimental.preprocessing import \
    TextVectorization


def define_tokenization_layers(
    encoder_train: np.ndarray,
    decoder_train: np.ndarray,
    max_encoder_seq_length: int,
    max_decoder_seq_length: int,
) -> Tuple[TextVectorization, TextVectorization, TextVectorization]:
    tokenizer_layer_encoder = TextVectorization(
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        ngrams=None,
        output_mode="int",
        output_sequence_length=max_encoder_seq_length,
        name="encoder_vectorizer",
    )
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

    tokenizer_layer_encoder.adapt(encoder_train)
    tokenizer_layer_decoder.adapt(decoder_train)
    tokenizer_layer_decoder_inference.adapt(decoder_train)

    return (
        tokenizer_layer_encoder,
        tokenizer_layer_decoder,
        tokenizer_layer_decoder_inference,
    )


def define_full_model(
    input_text_encoder: Input,
    enc_emb: Embedding,
    encoder_lstm: Bidirectional,
    input_text_decoder: Input,
    dec_emb: Embedding,
    decoder_lstm: LSTM,
    attention_layer: Attention,
    decoder_dense: Dense,
) -> Model:
    # encoder part
    # encoder lstm 1
    encoder_output, state_h_for, state_c_for, state_h_back, state_c_back = encoder_lstm(
        enc_emb
    )
    state_h = Concatenate(name="concatenate_h")([state_h_for, state_h_back])
    state_c = Concatenate(name="concatenate_c")([state_c_for, state_c_back])

    # decoder part
    # encoder lstm 1
    decoder_output, decoder_h_state, decoder_c_state = decoder_lstm(
        dec_emb, initial_state=[state_h, state_c]
    )
    # Attention layer
    attn_context_vector = attention_layer([decoder_output, encoder_output])
    # Concat attention input and decoder LSTM output
    decoder_concat_input = Concatenate(name="concat_layer")(
        [decoder_output, attn_context_vector]
    )
    # dense layer
    dense_outputs = decoder_dense(decoder_concat_input)

    # Define the model
    model = Model(
        [input_text_encoder, input_text_decoder],
        [dense_outputs],
    )
    return model
