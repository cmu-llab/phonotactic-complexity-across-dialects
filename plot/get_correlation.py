import pandas as pd
import argparse
from scipy.stats import pearsonr


parser = argparse.ArgumentParser(description='Phoneme LM')
parser.add_argument('--data', type=str, default='dutch_nodiacritics',
                help='Dataset used. (default: dutch_nodiacritics) / min')
parser.add_argument('--run-mode', type=str, default='normal',
                    choices=['normal', 'non-syllabified', 'syllabified'],
                    help='Path where data is stored.')
args = parser.parse_args()


file = f'results/{args.data}/{args.run_mode}/compiled-results.tsv'
df = pd.read_csv(file, sep='\t')

print('Spearman', df.corr(method='spearman'))
print('Pearson', df.corr(method='pearson'))
print('Pearson (scipy)', pearsonr(df['avg_len'], df['lstm']))
