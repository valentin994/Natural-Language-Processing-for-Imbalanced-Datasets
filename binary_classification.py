import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pprint import pprint as print
from langdetect import detect
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def text_prepare(txt):
    """Apply text cleaning and lemmization to a row of dataframe

    Args:
        txt (string): Full input sentence

    Returns:
        string: Cleaned, lemminized text
    """
    print(txt)
    txt = re.sub(r"[^\w\s]", " ", str(txt).lower().strip())
    txt = txt.split()
    nltk.corpus.stopwords.words("english")
    txt = [word for word in txt if word not in nltk.corpus.stopwords.words("english")]
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    txt = [lem.lemmatize(word) for word in txt]
    txt = " ".join(txt)
    return txt


# Setup

sns.set_theme(style="darkgrid")
# df = pd.read_csv("./data/tweets.csv")
df = pd.read_csv("./twitter_set/data.csv")

# Target Distribution
"""
ax = sns.barplot(df["target"].unique(), df["target"].value_counts())
plt.savefig("./graphs/binary_classification_graphs/target_distribution.png")
"""

# Create language column
# lang = []
# for text in df["text"]:
#    try:
#        lang.append(detect(text))
#    except:
#        lang.append(np.nan)
#    #
# df["lan"] = lang

# df.to_csv("./twitter_set/data.csv", index=False)


# Graf prikazuje koliko se koji jezik koristi u tweetovima
""" 
plt.figure(figsize=(20, 20))
ax = sns.barplot(x=df["lan"].value_counts().index, y=df["lan"].value_counts())
ax.set(xlabel="Jezični kod", ylabel="Broj tekstova")
plt.savefig("./graphs/binary_classification_graphs/language_distribution.png")
plt.show()
"""

# Mozda dropat lokaciju i sve sto nije na engleskom, provjeriti josñ

# df = df.drop("location", axis=1)
# df = df[df["lan"] == "en"]

# Graf koji prikazuje usporedbu prvih pet najkoristenijih rijeci za oba slucaja
"""
top_five_positive = df[df["target"] == 1]["keyword"].value_counts()[0:5]
top_five_negative = df[df["target"] == 0]["keyword"].value_counts()[0:5]
tfnp = df[(df["keyword"].isin(top_five_positive.index) & (df["target"] == 0))][
    "keyword"
].value_counts()
tfpn = df[(df["keyword"].isin(top_five_negative.index) & (df["target"] == 1))][
    "keyword"
].value_counts()

ax = sns.lineplot(data=top_five_positive)
bx = sns.lineplot(data=tfnp)
ax.legend(labels=["Target pozitivan", "Target negativan"])
plt.savefig(
    "./graphs/binary_classification_graphs/top_five_keywords_for_target_positive_comparison.png"
)
plt.clf()
ax = sns.lineplot(data=top_five_negative)
bx = sns.lineplot(data=tfpn)
ax.legend(labels=["Target negativna", "Target pozitivan"])
plt.savefig(
    "./graphs/binary_classification_graphs/top_five_keywords_for_target_negative_comparison.png"
)
"""

# df["clean_text"] = df["text"].apply(lambda x: text_prepare(x))
# df.to_csv("./twitter_set/data.csv", index=False)

# sid = SentimentIntensityAnalyzer()
# df["scores"] = df["text"].apply(lambda x: sid.polarity_scores(x))
# df["scores"] = df["scores"].apply(lambda x: x["compound"])

# df.to_csv("./twitter_set/data.csv", index=False)

print(df)