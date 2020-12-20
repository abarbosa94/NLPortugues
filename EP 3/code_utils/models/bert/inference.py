from typing import Union

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import (GRU, LSTM, Attention, Concatenate,
                                            Dense, Embedding)
from tensorflow.python.keras.models import Model
from transformers import TFBertModel


def encoder_inference_model(
    input_text_encoder: Input, encoder_model: TFBertModel
) -> Model:
    # Inputs from the encoder
    bert_encoder_output = encoder_model(input_text_encoder)
    encoder_output = bert_encoder_output.pooler_output
    encoder_hidden_states = bert_encoder_output.last_hidden_state
    encoder_model = Model(input_text_encoder, [encoder_hidden_states, encoder_output])
    return encoder_model


def decoder_inference_model(
    decoder_embedding: Embedding,
    target_text: Input,
    latent_dim: int,
    encoder_sequence_length: int,
    decoder_model: Union[LSTM, GRU],
    attention_layer: Attention,
    decoder_dense: Dense,
) -> Model:
    # Inputs from the encoder
    decoder_state_input_h = Input(shape=(latent_dim,), name="inference_h")
    encoder_output_inf = Input(
        shape=(encoder_sequence_length, latent_dim), name="output_from_encoder"
    )

    decoder_inference, state_h_inference = decoder_model(
        decoder_embedding, initial_state=[decoder_state_input_h]
    )

    attn_context_inference = attention_layer([decoder_inference, encoder_output_inf])

    decoder_concat_inference = Concatenate(name="concat_layer")(
        [decoder_inference, attn_context_inference]
    )
    dense_inference = decoder_dense(decoder_concat_inference)
    decoder_model = Model(
        [target_text, encoder_output_inf, decoder_state_input_h],
        [dense_inference, state_h_inference],
    )
    return decoder_model
