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
    encoder_pooler_output = bert_encoder_output.pooler_output
    encoder_hidden_output = bert_encoder_output.last_hidden_state
    encoder_model = Model(input_text_encoder, [encoder_pooler_output])
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
    encoder_pooler_output = Input(shape=(latent_dim,), name="inference_h")

    decoder_output, decoder_h_state = decoder_model(
        decoder_embedding, initial_state=[encoder_pooler_output]
    )

    dense_outputs = decoder_dense(decoder_output)
    decoder_model = Model(
        [target_text, encoder_pooler_output],
        [dense_outputs, decoder_h_state],
    )
    return decoder_model
