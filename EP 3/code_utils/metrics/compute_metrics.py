from typing import Tuple

import nltk
import numpy as np
from nltk.translate import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist
from pandas import DataFrame
from tensorflow.keras import Model
from tensorflow.python.keras.layers import TextVectorization
from tqdm import tqdm

from code_utils.preprocessing.data_preprocessing import SpecialTokens

nltk.download("wordnet")


# generate target given source sequence
def decode_text(
    infenc: Model,
    infdec: Model,
    source: str,
    n_steps: int,
    tokenizer_layer_decoder_inference: TextVectorization,
) -> str:
    # start of sequence input
    out_inf, state_h_inf, state_c_inf = infenc.predict([source])
    state = [state_h_inf, state_c_inf]
    target_seq = np.array([SpecialTokens.START_TOKEN.value])
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + [out_inf] + state)
        # store prediction
        next_item = yhat[0, 0, :].argmax()
        word = tokenizer_layer_decoder_inference.get_vocabulary()[next_item]
        word = word.decode("utf-8")
        output.append(word)
        if word == "xxend":
            break
        # update state
        state = [h, c]
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
        results.append(
            (input_text[idx], target_text[idx], preprocessed_original[idx], target)
        )
    return DataFrame(
        results,
        columns=["original", "target_original", "target_processed", "predicted"],
    ).set_index(sample_idx)


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
