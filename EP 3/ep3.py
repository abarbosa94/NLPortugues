from typing import Tuple, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import \
    TextVectorization
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers import (GRU, LSTM, Attention,
                                            Bidirectional, Dense, Embedding)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
from transformers import  TFBertModel

from code_utils.metrics import generate_sequences, report_linguistic_metrics
from code_utils.models import bert, bilstm
from code_utils.preprocessing.data_preprocessing import process_data

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)

BERT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
REVIEW_TITLE = "review_title"
REVIEW_TEXT = "review_text"
ENCODER_SEQ_LENGTH = 117
DECODER_SEQ_LENGTH = 8
BATCH_SIZE = 128
EPOCHS = 1
EMBED_DIM = 256
BERT_DIM = 768


def plot_metrics(history: History, model_name):
    # summarize history for loss
    # Inspired from here:
    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    plt.plot(history.history["loss"])
    epochs = range(len(history.history["loss"]))
    plt.plot(epochs, history.history["val_loss"])
    plt.title(f"Gráfico de loss do modelo {model_name}")
    plt.ylabel("loss")
    plt.xlabel("época")
    plt.legend(["loss de treinamento", "loss de validação"], loc="upper right")
    plt.savefig(f"data/{model_name}/loss_curve.png")
    return None


def bilstm_model_definition(
    tokenizer_layer_encoder: TextVectorization,
    tokenizer_layer_decoder: TextVectorization,
    tokenizer_layer_decoder_inference: TextVectorization,
):
    vocab_size_encoder = len(tokenizer_layer_encoder.get_vocabulary()) + 2
    vocab_size_decoder = len(tokenizer_layer_decoder.get_vocabulary()) + 2

    # Encoder definition
    input_text_encoder = Input(shape=(1,), dtype=tf.string, name="input_text")
    emb_enc_layer = Embedding(vocab_size_encoder, EMBED_DIM, name="encoder_embedding")
    encoder_lstm = Bidirectional(
        LSTM(EMBED_DIM, return_sequences=True, return_state=True, name="encoder_rnn")
    )

    # Decoder definition
    input_text_decoder = Input(shape=(None,), dtype=tf.string, name="decoder_input")
    emb_dec_layer = Embedding(vocab_size_decoder, EMBED_DIM, name="decoder_embedding")
    decoder_lstm = LSTM(
        EMBED_DIM * 2, return_sequences=True, return_state=True, name="decoder_rnn"
    )
    attention_layer = Attention(name="attention", causal=True)
    decoder_dense = Dense(vocab_size_decoder, activation="softmax", name="dense_layer")

    # Preprocessing step
    tokenized_input = tokenizer_layer_encoder(input_text_encoder)  # Tokenizer
    tokenized_decoder = tokenizer_layer_decoder(input_text_decoder)
    # embedding layer
    enc_emb = emb_enc_layer(tokenized_input)
    dec_emb = emb_dec_layer(tokenized_decoder)

    model = bilstm.define_full_model(
        input_text_encoder,
        enc_emb,
        encoder_lstm,
        input_text_decoder,
        dec_emb,
        decoder_lstm,
        attention_layer,
        decoder_dense,
    )

    encoder_inference = bilstm.encoder_inference_model(
        input_text_encoder, encoder_lstm, enc_emb
    )

    tokenized_decoder_inference = tokenizer_layer_decoder_inference(input_text_decoder)
    dec_emb_embedding_inference = emb_dec_layer(tokenized_decoder_inference)
    decoder_inference = bilstm.decoder_inference_model(
        decoder_embedding=dec_emb_embedding_inference,
        target_text=input_text_decoder,
        latent_dim=EMBED_DIM,
        encoder_sequence_length=ENCODER_SEQ_LENGTH,
        decoder_lstm=decoder_lstm,
        attention_layer=attention_layer,
        decoder_dense=decoder_dense,
    )

    return model, encoder_inference, decoder_inference


def bert_model_definition(
    bert_model_name: str,
    bert_dim: int,
    tokenizer_layer_decoder: TextVectorization,
    tokenizer_layer_decoder_inference: TextVectorization,
) -> Tuple[Model, Model, Model]:
    vocab_size_decoder = len(tokenizer_layer_decoder.get_vocabulary()) + 2

    # Encoder definition
    input_text_encoder = Input(
        shape=(ENCODER_SEQ_LENGTH,), dtype=tf.int32, name="input_text"
    )
    encoder_model = TFBertModel.from_pretrained(
        bert_model_name,
        output_hidden_states=False,
        output_attentions=False,
        from_pt=True,
    )

    # Decoder definition
    input_text_decoder = Input(shape=(None,), dtype=tf.string, name="decoder_input")
    emb_dec_layer = Embedding(vocab_size_decoder, EMBED_DIM, name="decoder_embedding")
    decoder_model = GRU(
        bert_dim, return_sequences=True, return_state=True, name="decoder_rnn"
    )
    attention_layer = Attention(name="attention", causal=True)
    decoder_dense = Dense(vocab_size_decoder, activation="softmax", name="dense_layer")

    ## Preprocessing step
    tokenized_decoder = tokenizer_layer_decoder(input_text_decoder)
    # embedding layer
    dec_emb = emb_dec_layer(tokenized_decoder)

    model = bert.define_full_model(
        input_text_encoder,
        encoder_model,
        input_text_decoder,
        dec_emb,
        decoder_model,
        attention_layer,
        decoder_dense,
    )

    encoder_inference = bert.encoder_inference_model(input_text_encoder, encoder_model)

    tokenized_decoder_inference = tokenizer_layer_decoder_inference(input_text_decoder)
    dec_emb_embedding_inference = emb_dec_layer(tokenized_decoder_inference)
    decoder_inference = bert.decoder_inference_model(
        decoder_embedding=dec_emb_embedding_inference,
        target_text=input_text_decoder,
        latent_dim=EMBED_DIM,
        encoder_sequence_length=ENCODER_SEQ_LENGTH,
        decoder_model=decoder_model,
        attention_layer=attention_layer,
        decoder_dense=decoder_dense,
    )

    return model, encoder_inference, decoder_inference


def execute_train(
    encoder_train: np.ndarray,
    decoder_train: np.ndarray,
    decoder_label_train: np.ndarray,
    model: Model,
    tokenizer_layer_decoder: TextVectorization,
    validation_split: float = 0.1,
) -> History:
    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=2)]

    # decay by 1/2 after 4 epochs; 1/3 by epoch 8 and so on
    steps_per_epoch = len(encoder_train) // BATCH_SIZE
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001, decay_steps=steps_per_epoch * 4, decay_rate=1, staircase=False
    )
    opt = tf.keras.optimizers.Adam(lr_schedule)
    model.compile(
        opt,
        loss={"dense_layer": "sparse_categorical_crossentropy"},
        metrics={"dense_layer": "accuracy"},
    )
    history = model.fit(
        [encoder_train, decoder_train],
        tokenizer_layer_decoder(decoder_label_train),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=my_callbacks,
        validation_split=validation_split,
    )

    return history


def experiment_pipeline(
    encoder_train: np.ndarray,
    decoder_train: np.ndarray,
    decoder_label_train: np.ndarray,
    model_name: str,
) -> Tuple[Model, Model, Model, TextVectorization, TextVectorization]:
    (
        model,
        encoder_inference,
        decoder_inference,
        tokenizer_layer_decoder,
        tokenizer_layer_decoder_inference,
    ) = (None, None, None, None, None)

    if model_name == "bilstm":

        (
            tokenizer_layer_encoder,
            tokenizer_layer_decoder,
            tokenizer_layer_decoder_inference,
        ) = bilstm.define_tokenization_layers(
            encoder_train, decoder_train, ENCODER_SEQ_LENGTH, DECODER_SEQ_LENGTH
        )

        model, encoder_inference, decoder_inference = bilstm_model_definition(
            tokenizer_layer_encoder,
            tokenizer_layer_decoder,
            tokenizer_layer_decoder_inference,
        )
    elif model_name == "bert":
        (
            tokenizer_layer_decoder,
            tokenizer_layer_decoder_inference,
        ) = bert.define_decoder_tokenization_layers(decoder_train, DECODER_SEQ_LENGTH)

        model, encoder_inference, decoder_inference = bert_model_definition(
            BERT_MODEL_NAME,
            BERT_DIM,
            tokenizer_layer_decoder,
            tokenizer_layer_decoder_inference,
        )

    plot_model(
        model,
        to_file=f"data/{model_name}/{model_name}_model.png",
        show_shapes=True,
        show_layer_names=True,
    )

    history = execute_train(
        encoder_train,
        decoder_train,
        decoder_label_train,
        model,
        tokenizer_layer_decoder,
    )
    plot_metrics(history, model_name=model_name)

    return (
        model,
        encoder_inference,
        decoder_inference,
        tokenizer_layer_decoder,
        tokenizer_layer_decoder_inference,
    )


def generate_final_results(
    model_definition: str,
    input_text: np.ndarray,
    target_text: np.ndarray,
    target_text_label: np.ndarray,
    full_model: Model,
    encoder_inference: Model,
    decoder_inference: Model,
    tokenizer_layer_decoder: TextVectorization,
    tokenizer_layer_decoder_inference: TextVectorization,
    n_samples: int,
) -> None:
    _, test_loss, test_acc = full_model.evaluate(
        [input_text, target_text],
        tokenizer_layer_decoder(target_text_label),
        batch_size=128,
    )

    sentences = generate_sequences(
        input_text,
        target_text,
        n_samples,
        encoder_inference,
        decoder_inference,
        DECODER_SEQ_LENGTH,
        tokenizer_layer_decoder_inference,
    )
    bleu_metric, nist_metric, meteor_metric = report_linguistic_metrics(sentences)
    final_dataframe = pd.DataFrame(
        [test_acc, bleu_metric, nist_metric, meteor_metric]
    ).T.rename(
        columns={0: "accuracy", 1: "bleu score", 2: "nist score", 3: "meteor score"}
    )
    sentences.to_csv(
        f"data/{model_definition}/sample_sentences.csv", index_label="index_column"
    )
    final_dataframe.to_csv(f"data/{model_definition}/final_metrics.csv", index=False)
    return None


@click.command()
@click.option(
    "--model_definition",
    prompt=True,
    required=True,
    type=click.Choice(["bilstm", "bert"]),
)
def execute_experiment(model_definition: str) -> None:
    (
        encoder_train,
        encoder_test,
        decoder_train,
        decoder_test,
        decoder_label_train,
        decoder_label_test,
    ) = process_data(DATA_PATH, REVIEW_TEXT, REVIEW_TITLE)

    if model_definition == "bert":
        encoder_train = bert.tokenize_encoder(
            encoder_train, BERT_MODEL_NAME, ENCODER_SEQ_LENGTH
        )
        encoder_test = bert.tokenize_encoder(
            encoder_test, BERT_MODEL_NAME, ENCODER_SEQ_LENGTH
        )

    (
        full_model,
        encoder_inference,
        decoder_inference,
        tokenizer_layer_decoder,
        tokenizer_layer_decoder_inference,
    ) = experiment_pipeline(
        encoder_train, decoder_train, decoder_label_train, model_definition
    )
    generate_final_results(
        model_definition,
        encoder_test,
        decoder_test,
        decoder_label_test,
        full_model,
        encoder_inference,
        decoder_inference,
        tokenizer_layer_decoder,
        tokenizer_layer_decoder_inference,
        1000,
    )
    return None


if __name__ == "__main__":
    DATA_PATH = "data/b2w-10k.csv"
    execute_experiment()
