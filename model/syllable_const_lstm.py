# adapted from Pimentel et al 2020 https://github.com/tpimentelms/phonotactic-complexity/blob/master/learn_layer/model/lstm.py

import copy
import torch.nn as nn


class SyllableConstituentLM(nn.Module):
    def __init__(self, phoneme_vocab_size, hidden_size, nlayers=1, dropout=0.1, embedding_size=None):
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size if embedding_size is not None else hidden_size
        self.dropout_p = dropout
        self.phoneme_vocab_size = phoneme_vocab_size

        # phoneme embedding
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(
            self.embedding_size, hidden_size, nlayers, dropout=(dropout if nlayers > 1 else 0), batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, phoneme_vocab_size)

        self._cuda = False
        self.best_state_dict = None

    def forward(self, x_phonemes, x_constituents, h_old=None):
        '''
        Input:
            x_phonemes: (B, S) input phonemes
            x_constituents: (B, S) input phonemes' constituents (supplied by the syllabifier)

        Where:
            B - batch size
            S - sequence length
        '''
        assert x_phonemes.size() == x_constituents.size()
        # separate phoneme embeddings and constituent embeddings
        #   enables phonemes in different constituent positions (e.g. /p/ in the onset and nucleus) to share information
        x_phon_emb = self.dropout(self.phoneme_embedding(x_phonemes))
        phon_c_t, phon_h_t = self.lstm(x_phon_emb, h_old)
        phon_c_t = self.dropout(phon_c_t).contiguous()
        phon_logits = self.out(phon_c_t)

        return phon_logits, phon_h_t, phon_logits, phon_h_t

    def initHidden(self, bsz=1):
        weight = next(self.parameters()).data
        return weight.new(self.nlayers, bsz, self.hidden_size).zero_(), \
            weight.new(self.nlayers, bsz, self.hidden_size).zero_()

    def cuda(self):
        super().cuda()
        self._cuda = True

    def cpu(self):
        super().cpu()
        self._cuda = False

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)
