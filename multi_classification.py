import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pprint import pprint as print
from langdetect import detect
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sns.set_theme(style="darkgrid")


def distribution(
    df: pd.DataFrame, column: str, show: bool, kde: bool, save_location: str
) -> None:
    """
        Distribution plot for a column in a dataframe

    Args:
        df (pd.DataFrame): Dataframe
        column (str): Column from which the distribution is plotted
        show (bool): Flag to show plot
        kde (bool): Flag for showing kde in plot
        save_location (str): Path to where the plot should be saved
    """
    sns.displot(data=df, x=column, kde=kde, height=10)
    plt.savefig(save_location)
    if show:
        plt.show()


def distributions(
    df: pd.DataFrame,
    dist_class: str,
    column: str,
    show: bool,
    save_location: str,
) -> None:
    """
        Plot distribution of the same x variable for multiple classes

    Args:
        df (pd.DataFrame): Data
        dist_class (str): How to split the values from df
        column (str): Column of dataframe
        show (bool): Flag to show plot
        hist (bool): Hist type flag
        kde (bool): Kde type flag
        save_location (str): Path to where the plot should be saved
    """
    sns.displot(x=df[column], hue=df[dist_class], kind="kde", clip=(1.0, 8.0))
    plt.savefig(save_location)
    if show:
        plt.show()


def language_detection_and_cleaning(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
        Creates lang columns which maps based on the text_column for each row,
        after that the text is lemmatized and added into a new text column.
        Returns dataframe.

    Args:
        df (pd.DataFrame): Dataframe
        text_column (str): Column where the posts/texts are stored, text cleaning and preparation
        methods are applied on it

    Returns:
        df(pd.DataFrame): Cleaned and updated dataframe0
    """
    df["lang"] = df["posts"].apply(lambda x: detect(x) if x.strip() != "" else "")

    def text_processing(txt: str):
        """
            Remove english stopwords, and lemmanize the text

        Args:
            txt (str): Text that we send from the column in the lambda function

        Returns:
            txt (str): Cleaned text
        """
        txt = re.sub(r"[^\w\s]", " ", str(txt).lower().strip())
        txt = txt.split()
        nltk.corpus.stopwords.words("english")
        txt = [
            word for word in txt if word not in nltk.corpus.stopwords.words("english")
        ]
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        txt = [lem.lemmatize(word) for word in txt]
        txt = " ".join(txt)
        return txt

    df["text"] = df[text_column].apply(lambda x: text_processing(x))
    return df


def length_params(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
        Add parameters based on the text length, word count and sentence count.

    Args:
        df (pd.DataFrame): Dataframe
        text_column (str): Which column to apply len methods

    Returns:
        pd.DataFrame: Dataframe with added parameters
    """
    df["word_count"] = df[text_column].apply(lambda x: len(str(x).split(" ")))
    df["letter_count"] = df[text_column].apply(
        lambda x: sum(len(word) for word in str(x).split(" "))
    )
    df["sentence_count"] = df[text_column].apply(lambda x: len(str(x).split(".")))
    df["avg_word_len"] = df["letter_count"] / df["word_count"]
    df["avg_sentence_len"] = df["word_count"] / df["sentence_count"]
    return df


# df = pd.read_csv("./data/mbti_1.csv", index=False)
# distribution(
#     df,
#     "type",
#     False,
#     True,
#     "./graphs/myers_briggs_multiclassification/personality_distribution.png",
# )

# df = pd.read_csv("./data/cleaned_mbti.csv")
# df = language_detection_and_cleaning(df)

# df = pd.read_csv("./data/cleaned_mbti.csv")
# df = length_params(df, "posts")
# df.to_csv("./data/cleaned_mbti.csv", index=False)
df = pd.read_csv("./data/cleaned_mbti.csv")
distributions(
    df,
    "type",
    "avg_word_len",
    True,
    "./graphs/myers_briggs_multiclassification/avg_word_len_distribution.png",
)
print(df.info())