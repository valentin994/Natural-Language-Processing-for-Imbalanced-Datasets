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
    df["word_count"] = df[text_column].apply(lambda x: len(str(x).split(" ")))
    return df


# df = pd.read_csv("./data/mbti_1.csv")
# distribution(
#     df,
#     "type",
#     False,
#     True,
#     "./graphs/myers_briggs_multiclassification/personality_distribution.png",
# )

# df = pd.read_csv("./data/cleaned_mbti.csv")
# df = language_detection_and_cleaning(df)

df = pd.read_csv("./data/cleaned_mbti.csv")
print(length_params(df, "text"))