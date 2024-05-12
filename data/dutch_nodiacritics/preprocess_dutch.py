import pandas as pd
import panphon.sonority
from panphon import FeatureTable
import unicodedata

df = pd.read_csv("original_nodiacritics.csv")

# tokenize diphthongs, triphthongs together
son = panphon.sonority.Sonority()
ft = FeatureTable()
VOWEL_SONORITY = 8


def preprocess(ipa):
    # NFD form to ensure length of input is the same number of Unicode chars as panphon ipa_segs output
    ipa = unicodedata.normalize('NFD', ipa)
    tokens = ipa.split(' ')
    final_tokens = []

    for token in tokens:
        if len(token) == 0:
            continue

        final_tokens.append(token)

    return ' '.join(final_tokens)


df["IPA"] = df["IPA"].map(preprocess, na_action='ignore')
print(df.head())

# filter out Frisian
with_frisian = len(df["Language_ID"].unique())
df = df[~df["Language_ID"].str.endswith("Fr")]
without_frisian = len(df["Language_ID"].unique())
print("removed", with_frisian - without_frisian, "Frisian varieties")

# remove Belgian dialects
dutch_langs = pd.read_csv('dutch-subset.csv')
dutch_langs = set(dutch_langs['lang'].tolist())
df = df[df["Language_ID"].isin(dutch_langs)]
without_belgian = len(df["Language_ID"].unique())
print("removed", without_frisian - without_belgian, "Belgian varieties")
print(without_belgian, "dialects now")

df.to_csv("orig.tsv", sep="\t", index=None)
