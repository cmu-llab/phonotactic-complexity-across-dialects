with open('original.csv', 'r') as f_read:
    f_data = f_read.read()

with open('original_nodiacritics.csv', 'w') as f_write:
    # trill - use the newer symbol
    f_data = f_data.replace("ɼ", "r")
    # use newer IPA symbol
    f_data = f_data.replace("ɷ", "ʊ")
    # remove diacritic
    f_data = f_data.replace(" ̞", "")
    # remove diacritic
    f_data = f_data.replace("̍", "")
    # remove diacritic
    f_data = f_data.replace(" ̱", "")
    # rhotic vowel - should be tokenized with the preceding vowel
    f_data = f_data.replace(" ˞", "˞")
    # palatalization - should be tokenized as 1 phoneme
    f_data = f_data.replace(" ʲ", "ʲ")

    f_write.write(f_data)
