# wrapper for syllabiphon
from syllabiphon.syllabify import Syllabify
from collections import namedtuple
from panphon import FeatureTable
import re


class Syllable:
    def __init__(self, onset, nucleus, coda):
        self.ons = onset
        self.nucleus = nucleus
        self.coda = coda

    def __str__(self):
        return '(' + 'ons=' + self.ons + ',nuc=' + self.nucleus + ',cod=' + self.coda + ')'


    def __repr__(self):
        # https://stackoverflow.com/questions/12933964/printing-a-list-of-objects-of-user-defined-class
        # necessary for printing the actual contents instead of the memory address of Syllable objects when loading the pickle
        return str(self)


class SyllabifySerializer(Syllabify):
    def __init__(self):
        super(SyllabifySerializer, self).__init__()

    def serialize(self, syllables):
        if not syllables:
            return []
        # serialize as tuple of strings
        return [Syllable(syl.ons, syl.nuc, syl.cod) for syl in syllables]

    def pretty_print(self, syllables):
        if not syllables:
            return []
        # serialize as tuple of strings
        return [(syl.ons, syl.nuc, syl.cod) for syl in syllables]

    def deserialize(self):
        pass


class DutchSyllabify(SyllabifySerializer):
    def __init__(self):
        super(DutchSyllabify, self).__init__()

    def _to_grid(self, word):
        NASAL_SONORITY = 5

        Seg = namedtuple('Seg', ['ph', 'son'])
        segs = self.ft.ipa_segs(word)
        # set the sonority of [s] to 0 whenever it's in a cluster
        seg_list = []
        for idx, ph in enumerate(segs):
            sonority = self._sonority(ph)
            # if in a consonant cluster and at end or start of word
            # should remove nasal clusters?
            if ph == 's':
                #  #st
                #  ts#
                if len(segs) > 1 and ((idx == 0 and self._sonority(segs[idx + 1]) < NASAL_SONORITY) or \
                        (idx == len(segs) - 1 and self._sonority(segs[idx - 1]) < NASAL_SONORITY)):
                    sonority = 0
            seg_list.append(Seg(ph, sonority))
        return seg_list


class ChineseSyllabify(SyllabifySerializer):
    def __init__(self):
        super(ChineseSyllabify, self).__init__()
        self.ft = FeatureTable()

    def create_syllable(self, onset, nucleus, coda, tone):
        Syl = namedtuple('Syl', ['ons', 'nuc', 'cod', 'tone'])
        return Syl(onset, nucleus, coda, tone)

    def segment_syllable(self, syllable):
        """
        Syllabify the segments of a syllable (no tone) into onset, nucleus, and coda using the [cons] feature.
        Any [-cons] segment is in the nucleus (except for [ʔ], which we treat as a consonant)
        """
        onset, nucleus, coda = [], [], []
        #                           -1 => nucleus
        # in panphon, seg['cons'] =  0 => tone
        #                            1 => onset or coda
        for seg_str, seg in zip(self.ft.ipa_segs(syllable), self.ft.word_fts(syllable)):
            if seg['cons'] == 1 or seg_str == 'ʔ':
                if len(nucleus) == 0:
                    onset.append(seg_str)
                else:
                    coda.append(seg_str)
            elif seg['cons'] == -1:
                nucleus.append(seg_str)

        onset = ''.join(onset)
        nucleus = ''.join(nucleus)
        coda = ''.join(coda)

        return onset, nucleus, coda


    def syl_parse(self, word):
        """
        ex: ŋi31tʰœ51 -> [Syl('ŋ', 'i', '', '31'), Syl('tʰ', 'œ', '', '51')]
        """
        final_syllables = []
        syllables = re.findall('([^012345]+)([012345]+)', word)
        for syllable, tone in syllables:
            # do not pass in the tone
            onset, nucleus, coda = self.segment_syllable(syllable)
            syllable = self.create_syllable(onset, nucleus, coda, tone)
            final_syllables.append(syllable)

        return final_syllables
