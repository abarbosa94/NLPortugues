from typing import Tuple

import nltk
import numpy as np
from nltk.translate import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist
from pandas import DataFrame
from tensorflow import reshape
from tensorflow.keras import Model
from tensorflow.python.keras.layers import TextVectorization
from tqdm import tqdm

from code_utils.preprocessing.data_preprocessing import SpecialTokens

nltk.download("wordnet")


# generate target given source sequence
def decode_text(
    infenc,
    infdec,
    src,
    n_steps,
    tokenizer_layer_decoder_inference,
):
    # start of sequence input
    state = None
    out_inf = None
    if infdec.get_layer("decoder_rnn").__class__.__name__ is "GRU":
        src = reshape(src, shape=(1, -1))
        state_h_inf = infenc.predict(src)
        state = [state_h_inf]
    elif infdec.get_layer("decoder_rnn").__class__.__name__ is "LSTM":
        out_inf, state_h_inf, state_c_inf = infenc.predict([src])
        state = [state_h_inf, state_c_inf]
    target_seq = np.array([SpecialTokens.START_TOKEN.value])
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        h, c, yhat = None, None, None
        if infdec.get_layer("decoder_rnn").__class__.__name__ is "GRU":
            yhat, h = infdec.predict([target_seq] + state)
            # update state
            state = [h]
        elif infdec.get_layer("decoder_rnn").__class__.__name__ is "LSTM":
            yhat, h, c = infdec.predict([target_seq] + [out_inf] + state)
            state = [h, c]
        # store prediction
        next_item = yhat[0, 0, :].argmax()
        word = tokenizer_layer_decoder_inference.get_vocabulary()[next_item - 2]
        word = word.decode("utf-8")
        output.append(word)
        if word == "xxend":
            break
        # update target sequence
        target_seq = np.array([word])
    return " ".join(output)


# spot check some examples
def generate_sequences(
    input_text: np.ndarray,
    target_text: np.ndarray,
    random_samples: int,
    encoder_inference: Model,
    decoder_inference: Model,
    decoder_seq_length: int,
    tokenizer_layer_decoder_inference: TextVectorization,
) -> DataFrame:
    preprocessed_original = np.char.lower(target_text.astype("str"))
    results = []
    sample_idx = np.random.choice(input_text.shape[0], random_samples)
    for idx in tqdm(sample_idx, desc="Gerando Predições"):
        target = decode_text(
            encoder_inference,
            decoder_inference,
            input_text[idx],
            decoder_seq_length,
            tokenizer_layer_decoder_inference,
        )
        results.append((preprocessed_original[idx], target))
    final_df = DataFrame(
        results,
        columns=["target_processed", "predicted"],
    ).set_index(sample_idx)

    return final_df


def report_linguistic_metrics(sentences: DataFrame) -> Tuple[float, float, float]:
    meteor_metric = []
    bleu_metric = corpus_bleu(
        sentences["target_processed"].values.tolist(),
        sentences["predicted"].values.tolist(),
    )
    nist_metric = corpus_nist(
        sentences["target_processed"].values.tolist(),
        sentences["predicted"].values.tolist(),
    )
    for idx, row in sentences[["target_processed", "predicted"]].iterrows():
        single_metric = meteor_score.single_meteor_score(
            row["target_processed"][0], row["predicted"], wordnet=None
        )
        meteor_metric.append(single_metric)
    meteor_metric = np.mean(meteor_metric)
    return float(bleu_metric), float(nist_metric), float(meteor_metric)
