# Phonotactic Complexity across Dialects

This repository contains the code for the LREC-Coling 2024 paper "Phonotactic Complexity across Dialects."

> Received wisdom in linguistic typology holds that if the structure of a language becomes more complex in one dimension, it will simplify in another, building on the assumption that all languages are equally complex (Joseph and Newmeyer 2012). 
We study this claim on a micro-level, using a tightly-controlled sample of Dutch dialects (across 366 collection sites) and Min dialects (across 60 sites), which enables a more fair comparison across varieties.
Even at the dialect level, we find empirical evidence for a tradeoff between word length and a computational measure of phonotactic complexity from a LSTM-based phone-level language model---a result previously documented only at the language level. A generalized additive model (GAM) shows that dialects with low phonotactic complexity concentrate around the capital regions, which we hypothesize to correspond to prior hypotheses that language varieties of greater or more diverse populations show reduced phonotactic complexity. 
We also experiment with incorporating the auxiliary task of predicting syllable constituency, but do not find an increase in the strength of the negative correlation observed.

We adapt Pimentel et al 2020's phonotactic language model (https://github.com/tpimentelms/phonotactic-complexity) and supporting code. 


### Data
* data/
  * parse.py - generate train,dev,test numpy arrays from data/*/orig.tsv
  * syllabifier.py - wrapper for syllabiphon, with modifications for Dutch and Sinitic

#### Dutch
* data/dutch_nodiacritics
  * orig.tsv - the data in Pimentel et al 2020's format, with Frisian and Belgium filtered out; this is NOT the original data with preprocess_dutch.py
  * original.csv - the original Dutch data  (Taeldeman and Goeman 1996)
  * original_nodiacritics.csv - the result of applying remove_diacritics.py to original.csv
  * evaluate_syllabiphon.{py,csv} - evaluating the accuracy of modified syllabiphon

In short, we obtained orig.tsv using:
```
cd data/dutch_nodiacritics
python remove_diacritics.py
python preprocess_dutch.py
```


* Preparing the training data
```
python data/parse.py --data dutch_nodiacritics --data-path data --syllabify true
python data/parse.py --data dutch_nodiacritics --data-path data --syllabify false
```

#### Min
* Obtaining the data
  * Make an account at https://zhongguoyuyan.cn/.
  * data/min/scrape_zhongguoyuwen.js
    * note: the format of the website may have changed since June 2023, when we obtained the data.
* Get the data in Pimentel's form and store as data/min/min_pimentel_format.csv
  * It should have 3 columns: 地區,Concept_ID,IPA
  * The IPA should have no spaces between syllables
* Preprocessing the data
```
cd data/min
python preprocess_min.py
```

* Preparing the training data
```
python data/parse.py --data min --data-path data --syllabify true
python data/parse.py --data min --data-path data --syllabify false
```


### Training and results

* results/
#### Dutch
```
python train.py --dataset dutch_nodiacritics --syllabify True
python train.py --dataset dutch_nodiacritics --syllabify False
python plot/compile_results.py --data dutch_nodiacritics --run-mode syllabified
python plot/compile_results.py --data dutch_nodiacritics --run-mode non-syllabified
python plot/get_correlation.py --data dutch_nodiacritics --run-mode syllabified
python plot/get_correlation.py --data dutch_nodiacritics --run-mode non-syllabified
```

#### Min
```
python train.py --dataset min --syllabify True
python train.py --dataset min --syllabify false
python plot/compile_results.py --data min --run-mode syllabified
python plot/compile_results.py --data min --run-mode non-syllabified
python plot/get_correlation.py --data min --run-mode syllabified
python plot/get_correlation.py --data min --run-mode non-syllabified
```


#### Analysis
#### GAM
* analysis/gam.R


### Citing our paper
Please cite as :

Ryan Soh-Eun Shim*, Kalvin Chang*, David R. Mortensen. 2024. Phonotactic Complexity across Dialects. In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-Coling 2024)*, Torino, Italy.

```
@InProceedings{phonotactic-complexity-across-dialects:2024,
  author = {Shim, Ryan Soh-Eun and Chang, Kalvin and Mortensen, David R.},
  title = {Phonotactic Complexity across Dialects},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
  year = {2024},
  month = {May},
  date = {20--25},
  location = {Torino, Italy},
}
```
