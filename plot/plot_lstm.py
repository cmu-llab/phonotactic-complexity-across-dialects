# source: Pimentel et al 2020

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
sys.path.append('./')
import argparse


aspect = {
    'size': 6.5,
    'font_scale': 2.5,
    'labels': False,
    'ratio': 1.625,
}


vals = ['lstm']

sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})



parser = argparse.ArgumentParser(description='Phoneme LM')
parser.add_argument('--data', type=str, default='dutch_nodiacritics',
                help='Dataset used. (default: dutch_nodiacritics) / northeuralex')
parser.add_argument('--run-mode', type=str, default='normal',
                    choices=['normal', 'non-syllabified'],
                    help='Path where data is stored.')
args = parser.parse_args()
df = pd.read_csv(f'results/{args.data}/{args.run_mode}/compiled-results_avg.tsv', delimiter='\t')

frames = []
for val in vals:
    df_new = df[['lang', 'avg_len']].copy()
    df_new['test_loss'] = df[val]
    df_new['Model'] = val
    df_new.reset_index(level=0, inplace=True)

    frames += [df_new]

data = pd.concat(frames)
data.loc[data['Model'] == 'lstm', 'Model'] = '$LSTM$'

fig = sns.lmplot(
    'avg_len', 'test_loss', data, hue='Model', palette='muted',
    height=aspect['size'], aspect=aspect['ratio'], legend_out=False, truncate=False, legend=False)

# plt.xlim([1, 6])
# plt.ylim([2.25, 3.5])
plt.xlabel('Average Length (# IPA tokens)')
plt.ylabel('Cross Entropy (bits per phoneme)')
fig.savefig('plot/lstm_complexity.pdf')
plt.show()
