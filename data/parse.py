# Adapted from Pimentel et al 2020
# https://github.com/tpimentelms/phonotactic-complexity/blob/master/data_layer/parse.py

# Note: Dataset 0 is PAD 1 is SOW and 2 is EOW
import pandas as pd
import numpy as np
import pickle

import sys
sys.path.append('./')
import argparse
import pathlib

from syllabifier import SyllabifySerializer, DutchSyllabify, ChineseSyllabify
from panphon import FeatureTable
import panphon.sonority
from collections import namedtuple


def read_src_data(ffolder):
    filename = '%s/orig.tsv' % ffolder
    df = pd.read_csv(filename, sep='\t')
    
    if 'Glottocode' in df:
        del df['Glottocode']
    df = df[df.Language_ID != 'cmn']
    df = df.dropna()
    return df


def get_languages(df):
    return df.Language_ID.unique()


def get_phrases(df):
    phrases = df.Concept_ID.unique()
    np.random.shuffle(phrases)
    return phrases


def separate_df(df, train_set, val_set, test_set):
    train_df = df[df['Concept_ID'].isin(train_set)]
    val_df = df[df['Concept_ID'].isin(val_set)]
    test_df = df[df['Concept_ID'].isin(test_set)]

    return train_df, val_df, test_df


def separate_train(df):
    phrases = get_phrases(df)

    num_sentences = phrases.size

    train_size = int(num_sentences * .8)
    val_size = int(num_sentences * .1)
    test_size = num_sentences - train_size - val_size

    train_set = phrases[:train_size]
    val_set = phrases[train_size:-test_size]
    test_set = phrases[-test_size:]
    data_split = (train_set, val_set, test_set)

    train_df, val_df, test_df = separate_df(df, train_set, val_set, test_set)
    return train_df, val_df, test_df, data_split


def separate_per_language(train_df, val_df, test_df, languages):
    languages_df_train = separate_per_language_single_df(train_df, languages)
    languages_df_val = separate_per_language_single_df(val_df, languages)
    languages_df_test = separate_per_language_single_df(test_df, languages)

    languages_df = {
        lang: {
            'train': languages_df_train[lang],
            'val': languages_df_val[lang],
            'test': languages_df_test[lang],
        } for lang in languages_df_train.keys()}
    return languages_df


def separate_per_language_single_df(df, languages):
    languages_df = {lang: df[df['Language_ID'] == lang] for lang in languages}
    return languages_df


def get_tokens(df):
    tokens = set()
    for index, x in df.iterrows():
        try:
            tokens |= set(x.IPA.split(' '))
        except:
            continue

    tokens = sorted(list(tokens))
    token_map = {x: i + 3 for i, x in enumerate(tokens)}
    token_map['PAD'] = 0
    token_map['SOW'] = 1
    token_map['EOW'] = 2

    return token_map


def get_concept_ids(df):
    concepts = df.Concept_ID.unique()
    concept_ids = {x: i for i, x in enumerate(concepts)}
    strings_concepts = pd.Series(df.Concept_ID.values, index=df.index).to_dict()
    IPA_to_concept = {k: concept_ids[x] for k, x in strings_concepts.items()}

    return concept_ids, IPA_to_concept


def process_languages(languages_df, token_map, args):
    folder = '%s/preprocess/' % args.ffolder
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    for lang, df in languages_df.items():
        process_language(df, token_map, lang, args)


def process_language(dfs, token_map, lang, args):
    for mode in ['train', 'val', 'test']:
        process_language_mode(dfs[mode], token_map, lang, mode, args)


def process_language_mode(df, token_map, lang, mode, args):
    syllabifier = None
    if args.syllabify:
        if 'dutch' in args.data:
            syllabifier = DutchSyllabify()
        elif 'min' in args.data:
            syllabifier = ChineseSyllabify()
        else:
            syllabifier = SyllabifySerializer()
    data_phonemes, data_constituents = parse_data(df, token_map, syllabifier)
    save_data(data_phonemes, data_constituents, lang, mode, args.ffolder)


def postprocess_syllables(syllables, ft, son):
    Syl = namedtuple('Syl', ['ons', 'nuc', 'cod'])
    VOWEL_SONORITY = 8

    # merge diphthongs
    final_syllables = []
    for curr_syl in syllables:
        if len(final_syllables) > 0 and not (final_syllables[-1].cod + curr_syl.ons):

            final_syllables[-1] = Syl(final_syllables[-1].ons, final_syllables[-1].nuc + curr_syl.nuc, curr_syl.cod)
        else:
            final_syllables.append(curr_syl)

        if final_syllables[-1].cod and son.sonority(ft.ipa_segs(final_syllables[-1].cod)[0]) >= VOWEL_SONORITY:
            # need to use ipa_segs to do the segmentation because a vowel could consist of 2 Unicode characters (e.g. rhoticized vowels)
            coda_segments = ft.ipa_segs(final_syllables[-1].cod)
            final_syllables[-1] = Syl(final_syllables[-1].ons, final_syllables[-1].nuc + coda_segments[0],
                                      ''.join(coda_segments[1:]))

    return final_syllables


def parse_data(df, token_map, syllabifier):
    max_len = df.IPA.map(lambda x: len(x.split(' '))).max()
    data_phonemes = np.zeros((df.shape[0], max_len + 3))        # 0 = <PAD>
    if syllabifier:
        data_constituents = np.zeros((df.shape[0], max_len + 3))
        ft = FeatureTable()
        son = panphon.sonority.Sonority()
        # 0 = non-constituent (<PAD> / <BOS> / <EOS>)
        constituent_vocab = {
            'onset': 1,
            'nucleus': 2,
            'coda': 3,
            'tone': 4,
        }
    else:
        data_constituents = None

    for i, (index, x) in enumerate(df.iterrows()):
        try:
            # note: this assumes orig.tsv already has the phoneme tokenization
            # tones are currently treated as one token in the phoneme tokenization
            instance = x.IPA.split(' ')

            data_phonemes[i, 0] = 1  # <BOS> / SOW
            data_phonemes[i, 1:len(instance) + 1] = [token_map[z] for z in instance]
            data_phonemes[i, len(instance) + 1] = 2  # <EOS> / EOW
            data_phonemes[i, -1] = index

            if syllabifier and len(instance) > 0:
                syllables = syllabifier.syl_parse(''.join(instance))
                syllables = postprocess_syllables(syllables, ft, son)

                pointer = 1
                # postprocess to add tone as label
                for syllable in syllables:
                    # each constituent could have more than 1 phoneme (e.g. "vɾ")
                    for _ in ft.ipa_segs(syllable.ons):
                        data_constituents[i, pointer] = constituent_vocab['onset']
                        pointer += 1
                    for _ in ft.ipa_segs(syllable.nuc):
                        data_constituents[i, pointer] = constituent_vocab['nucleus']
                        pointer += 1
                    for _ in ft.ipa_segs(syllable.cod):
                        data_constituents[i, pointer] = constituent_vocab['coda']
                        pointer += 1
                    if syllable.tone:
                        # the entire tone, even if it is a tone contour, will be treated as one token
                        # ex: Mandarin tone 3 ("213") will be treated as one token "213"
                        data_constituents[i, pointer] = constituent_vocab['tone']
                        pointer += 1

                assert pointer == len(instance) + 1
                # if syllabify option selected, then create a matrix same size as data where the values are indices
                    # but need to align the phoneme and the constituent - 2 pointers approach
                    # assert that len of constituents == # phonemes; constituent could consist of more than 1 phoneme
        except Exception:
            continue

    return data_phonemes, data_constituents


def save_data(data_phonemes, data_constituents, lang, mode, ffolder):
    with open('%s/preprocess/data-%s-%s.npy' % (ffolder, lang, mode), 'wb') as f:
        np.save(f, data_phonemes)

    if data_constituents is not None:
        with open('%s/preprocess/data-syllabified-%s-%s.npy' % (ffolder, lang, mode), 'wb') as f:
            np.save(f, data_constituents)



def save_info(ffolder, languages, token_map, data_split, concepts_ids, IPA_to_concept):
    info = {
        'languages': languages,
        'token_map': token_map,
        'data_split': data_split,
        'concepts_ids': concepts_ids,
        'IPA_to_concept': IPA_to_concept,
    }
    with open('%s/preprocess/info.pckl' % ffolder, 'wb') as f:
        pickle.dump(info, f)


def load_info(args):
    with open('%s/preprocess/info.pckl' % args.ffolder, 'rb') as f:
        info = pickle.load(f)
    languages = info['languages']
    token_map = info['token_map']
    data_split = info['data_split']
    concept_ids = info['concepts_ids']

    return languages, token_map, data_split, concept_ids


def main(args):
    df = read_src_data(args.ffolder)

    languages = get_languages(df)
    train_df, val_df, test_df, data_split = separate_train(df)
    token_map = get_tokens(df)
    concepts_ids, IPA_to_concept = get_concept_ids(df)

    languages_df = separate_per_language(train_df, val_df, test_df, languages)

    process_languages(languages_df, token_map, args)
    save_info(args.ffolder, languages, token_map, data_split, concepts_ids, IPA_to_concept)


if __name__ == '__main__':
    np.random.seed(seed=12345)

    parser = argparse.ArgumentParser(description='Phoneme LM')
    parser.add_argument('--data', type=str, default='dutch_nodiacritics',
                    help='Dataset used. (default: dutch_nodiacritics)')
    parser.add_argument('--data-path', type=str, default='data',
                        help='Path where data is stored.')
    parser.add_argument('--syllabify', type=lambda x: (str(x).lower() == 'true'), help='whether or not to syllabify the phonetic transcriptions')
    args = parser.parse_args()

    print("Syllabify!" if args.syllabify else "Do not syllabify")

    args.ffolder = '%s/%s' % (args.data_path, args.data)  # Data folder
    assert args.data in ['dutch_nodiacritics', 'northeuralex', 'german', 'min'], 'this script should only be run with dutch_nodiacritics data'
    main(args)
