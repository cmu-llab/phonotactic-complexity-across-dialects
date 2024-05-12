import pandas as pd
import re
import unicodedata


df = pd.read_csv("min_pimentel_format.csv")

# column name - 地區 -> Language_ID
df = df.rename(columns={
    "地區": "Language_ID",
})


def preprocess(ipa):
    chao_tone_mappings = {
        #"1": "˩",
        #"2": "˨",
        #"3": "˧",
        #"4": "˦",
        #"5": "˥",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
    }
    # keep the tones together
    chars = []
    for segments, tone in re.findall('(\D*)(\d+)', ipa):
        # Chao tone letters
        for tone_num, chao_tone in chao_tone_mappings.items():
            tone = tone.replace(tone_num, chao_tone)

        # normalize the transcriptions

        # add affricate ligatures for ts, tʃ so that panphon tokenizes correctly
        segments = segments.replace("ts", "t͡s")
        segments = segments.replace("tʃ", "t͡ʃ")
        segments = segments.replace("ʦ", "t͡s")
        segments = segments.replace("tɕ", "t͡ɕ")
        segments = segments.replace("dʒ", "d͡ʒ")
        segments = segments.replace("dz", "d͡z")
        segments = segments.replace("dʑ", "d͡ʑ")

        # unsure of what notation is used - ᴂ
        segments = segments.replace("ᴂ", "æ")
        # reverse line feed ''
        segments = segments.replace("", "")
        segments = segments.replace("\ufffb", "")
        # use IPA g
        segments = segments.replace("g", "ɡ")
        # replace small tilde with combining tilde
        segments = segments.replace("˜", "̃")
        segments = segments.replace("", "̃")
        segments = segments.replace("", "̃")

        # Sinological IPA
        segments = segments.replace("ᴇ", "ɛ̝")
        segments = segments.replace("∅", "ʔ")
        segments = segments.replace("Ǿ", "ʔ")
        segments = segments.replace("ɿ", "z̩")
        # technically n̠ʲ but panphon will lose either of the 2 diacritics
        segments = segments.replace("ȵ", "nʲ")

        # aspiration (h whenever it is not the start of the sequence)
        segments = re.sub("(.+)(h)(.*)", r"\1ʰ\3", segments)
        # NFD form to ensure length of input is the same number of Unicode chars as panphon ipa_segs output
        segments = unicodedata.normalize('NFD', segments)

        # group affricates and aspiration together during tokenization
        tokens = []
        # TODO: semivowel marker
        diacritics = {"ʰ", "̃", "̩", "̝"}
        for idx, char in enumerate(segments):
            if char in diacritics:
                tokens[-1] += char
            elif idx > 0 and segments[idx - 1] == '͡':
                # merge with previous
                tokens[-2] += ('͡' + char)
                tokens.pop()
            else:
                tokens.append(char)

        tokens.append(tone)
        chars.append(' '.join(tokens))
    return ' '.join(chars)


df["IPA"] = df["IPA"].map(preprocess, na_action='ignore')
print(df.head())

df.to_csv("orig.tsv", sep="\t", index=None)
