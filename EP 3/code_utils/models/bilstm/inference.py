from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (LSTM, Attention, Bidirectional,
                                     Concatenate, Dense, Embedding)


def encoder_inference_model(
    input_text_encoder: Input, encoder_lstm: Bidirectional, embedding_encoder: Embedding
) -> Model:
    # Inputs from the encoder
    encoder_output, state_h_for, state_c_for, state_h_back, state_c_back = encoder_lstm(
        embedding_encoder
    )
    state_h = Concatenate()([state_h_for, state_h_back])
    state_c = Concatenate()([state_c_for, state_c_back])
    encoder_model = Model(input_text_encoder, [encoder_output, state_h, state_c])
    return encoder_model


def decoder_inference_model(
    decoder_embedding: Embedding,
    target_text: Input,
    latent_dim: int,
    encoder_sequence_length: int,
    decoder_lstm: LSTM,
    attention_layer: Attention,
    decoder_dense: Dense,
) -> Model:
    # Inputs from the encoder
    decoder_state_input_h = Input(shape=(latent_dim * 2,), name="inference_h")
    decoder_state_input_c = Input(shape=(latent_dim * 2,), name="inference_c")
    encoder_output_inf = Input(
        shape=(encoder_sequence_length, latent_dim * 2), name="output_from_encoder"
    )

    decoder_inference, state_h_inference, state_c_inference = decoder_lstm(
        decoder_embedding, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )

    attn_context_inference = attention_layer([decoder_inference, encoder_output_inf])

    decoder_concat_inference = Concatenate(name="concat_layer")(
        [decoder_inference, attn_context_inference]
    )
    dense_inference = decoder_dense(decoder_concat_inference)
    decoder_model = Model(
        [target_text, encoder_output_inf, decoder_state_input_h, decoder_state_input_c],
        [dense_inference, state_h_inference, state_c_inference],
    )
    return decoder_model
