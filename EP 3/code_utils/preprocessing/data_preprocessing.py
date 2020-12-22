import string
from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class SpecialTokens(Enum):
    START_TOKEN = "xxtsart"
    END_TOKEN = "xxend"


def strip_accents(numpy_array: np.ndarray) -> np.ndarray:
    return np.char.strip(np.array(numpy_array).astype("str"), string.punctuation)


def preprocess_embedding_from_NILC(embedding_path_origin: str, embed_dim: int, N: int):
    with open(f"data/cbow_s{embed_dim}.txt", "r") as file:
        head = [next(file) for x in range(N)]

    head[0] = str(N - 1) + " " + f"{embed_dim}" + "\n"  # Conserta contagem de palavras
    vocab = set()
    with open(embedding_path_origin, "w") as file:
        i = 0
        iter_rows = iter(head)
        next(iter_rows)
        for line in tqdm(
            iter_rows, total=N, desc="Preprocessing Embeddings from NILC file"
        ):
            word_vector = line.split()
            word = word_vector[0]
            embedding = word_vector[1:]
            if len(embedding) != embed_dim:
                print(f"word line {i} is corrupt, skipping it.")
                continue
            i += 1
            if word in vocab:
                continue
            vocab.add(word)
            file.write(line)
    return None


def generate_embeddings(
    path: str, total_lines: int, embed_dim=50
) -> Tuple[np.ndarray, list]:
    """
    This function gets the NILC embeddings and store it
    as a numpy matrix
    All words are going to be initialized as zero and then
    they are modified according to specific vector
    """
    with open(path, "r") as file:
        i = 0
        embed_matrix = [np.zeros((2, embed_dim))]
        embed_vocab = []
        for line in tqdm(file, total=total_lines, desc="Generating Embedding Matrix"):
            splitted_words = line.split()
            embed_matrix.append(np.array(splitted_words[1:]).reshape(1, -1))
            embed_vocab.append(splitted_words[0])
            i += 1
        embed_matrix = np.vstack(embed_matrix)
    return embed_matrix, embed_vocab


def process_data(
    path: str, review_text: str, review_title: str, sep: str = ","
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=sep, low_memory=False)
    input_text = df[[review_text]]
    target_text = df[[review_title]]

    encoder_train, encoder_test, decoder_train, decoder_test = train_test_split(
        input_text, target_text, random_state=42, test_size=0.2
    )

    decoder_label_train = (
        decoder_train.astype(str) + f" {SpecialTokens.END_TOKEN.value}"
    )
    decoder_label_test = decoder_test.astype(str) + f" {SpecialTokens.END_TOKEN.value}"

    decoder_train = (
        f"{SpecialTokens.START_TOKEN.value} "
        + decoder_train.astype(str)
        + f" {SpecialTokens.END_TOKEN.value}"
    )
    decoder_test = (
        f"{SpecialTokens.START_TOKEN.value} "
        + decoder_test.astype(str)
        + f" {SpecialTokens.END_TOKEN.value}"
    )

    encoder_train = encoder_train.values
    encoder_test = encoder_test.values
    decoder_train = decoder_train.values
    decoder_test = decoder_test.values
    decoder_label_train = decoder_label_train.values
    decoder_label_test = decoder_label_test.values

    encoder_train = strip_accents(encoder_train).reshape(-1, 1)
    encoder_test = strip_accents(encoder_test).reshape(-1, 1)
    decoder_train = strip_accents(decoder_train).reshape(-1, 1)
    decoder_test = strip_accents(decoder_test).reshape(-1, 1)
    decoder_label_train = strip_accents(decoder_label_train).reshape(-1, 1)
    decoder_label_test = strip_accents(decoder_label_test).reshape(-1, 1)

    return (
        encoder_train,
        encoder_test,
        decoder_train,
        decoder_test,
        decoder_label_train,
        decoder_label_test,
    )
