import pandas as pd
import numpy as np

# universal seed used by pandas
np.random.seed(12345)
new_csv = {}

# take 100 pronunciations at random and manually inspect their syllabifications
# randomly pick 100 dialects and randomly pick 1 word within the dialect
N_SAMPLES = 100
df = pd.read_csv("dutch_nodiacritics_syllabified.csv", index_col="dialect")
# pick 100 dialects
subset = df.sample(n=N_SAMPLES, axis=0)
# pick 1 word from each dialect
for _, dialect in subset.iterrows():
    sample = dialect.sample(n=1)
    while sample.values[0] == '[]':
        # print(sample.values[0])
        sample = dialect.sample(n=1)
    new_csv[sample.name] = sample.values[0]

with open('evaluate_syllabiphon.csv', 'w') as f:
    f.write('dialect,sample,wrong\n')
    for dialect, sample in new_csv.items():
        f.write(dialect)
        f.write(',')
        f.write(sample)
        f.write(',')
        f.write('\n')
