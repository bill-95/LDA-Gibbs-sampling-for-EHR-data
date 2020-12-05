import pandas as pd
import numpy as np
import seaborn as sns
from seaborn import heatmap
import matplotlib.pyplot as plt

alpha = 1 
beta = 0.001
K = 5
gibbs_iter = 100

icd_data = pd.read_csv('D_ICD_DIAGNOSES_DATA_TABLE.csv')
icd_diagnoses = pd.read_csv('DIAGNOSES_ICD_subset.csv')

# initialize icd words (w) corpus
icd_corpus = icd_diagnoses['ICD9_CODE']
icd_count = len(icd_corpus)
unique_icd = set(icd_corpus)
unique_icd_count = len(unique_icd)
# assign a new remapped index for each unique icd code for use with word-topic counter matrix
icd_index_map = {ele:i for i, ele, in enumerate(unique_icd)}
icd_diagnoses['ICD9_CODE_new_index'] = icd_corpus.map(icd_index_map)

# initialize subject document (d) collection
subj_ids = icd_diagnoses['SUBJECT_ID']
subj_count = len(subj_ids)
unique_subj = set(subj_ids)
unique_subj_count = len(unique_subj)
# assign a new remapped index for each unique subject for use with document-topic counter matrix
subj_index_map = {ele:i for i, ele, in enumerate(unique_subj)}
icd_diagnoses['SUBJECT_ID_new_index'] = subj_ids.map(subj_index_map)

# initially assign a random topic to every icd word in the corpus
# this vector is also for tracking topic of each icd during gibbs sampling
z = np.random.randint(0, K, icd_count)

# word-topic counter matrix (n_wk)
n_wk = np.zeros(shape=(unique_icd_count, K), dtype=int)
# initialize n_wk based on inital random topic assignments (z)
for w, k in zip(icd_diagnoses['ICD9_CODE_new_index'], z):
    n_wk[w, k] += 1

# document-topic counter matrix (n_dk)
n_dk = np.zeros(shape=(unique_subj_count, K), dtype=int)
# initialize n_dk based on inital random topic assignments (z)
for d, k in zip(icd_diagnoses['SUBJECT_ID_new_index'], z):
    n_dk[d, k] += 1

# topic counter vector (n_k)
# this vector is used for tracking the total icd words assigned to each topic k, to simplify the probability calulations
n_k = n_wk.sum(0)

# topic probabilities tracker
p_ks = np.zeros(shape=(K))

# sum over beta is constant in the probability calculations
beta_sum = beta*unique_icd_count

    
# # Gibbs sampling loop
print('Running Gibbs Sampling')
for i in range(gibbs_iter):
    print('Iteration', i)
    for j in range(len(z)):
        w = icd_diagnoses['ICD9_CODE_new_index'][j] # word
        d = icd_diagnoses['SUBJECT_ID_new_index'][j] # document
        k = z[j] # current topic
        # decrease counters for current word and topic
        n_wk[w, k] -= 1
        n_dk[d, k] -= 1 
        n_k[k] -= 1
        # calculate the posterior p(k_j|z(-j), w_j) for each topic k given current word and other word-topic assignments
        # update p_ks vector
        for k in range(K):
            p_k = (n_dk[d, k] + alpha) * (n_wk[w, k] + beta) / (n_k[k] + beta_sum)
            p_ks[k] = p_k
        # calculate normalize probabilities
        p_ks = p_ks / p_ks.sum()
        # sample new topic from multinomial distribution and update tracker and counters
        k_new = np.random.choice([0,1,2,3,4], p=p_ks)
        z[j] = k_new
        n_wk[w, k_new] += 1
        n_dk[d, k_new] += 1 
        n_k[k_new] += 1

# filter for the top 10 icd codes for each topic
n_wk_prob = n_wk/n_wk.sum(0) # first convert count matrix to probabilities of each word under each topic
top_10_idx = np.apply_along_axis(lambda x: np.argsort(x)[-10:][::-1], 0, n_wk_prob)
top_10_idx = top_10_idx.reshape(-1, order='F')
top_10_n_wks = n_wk_prob[top_10_idx]

# map icd codes to short titles to use as labels for heatmap
index_icd_map = {v:k for k, v in icd_index_map.items()} # reverse map from word index to original icd
icd_data['ICD9_CODE_cleaned'] = icd_data['ICD9_CODE'].apply(lambda x: x.lstrip('0')) # clean up leading 0's from ICD9 codes
icd_short_titles = []
for idx in top_10_idx:
    icd_code = index_icd_map[idx]
    short_title = icd_data[icd_data['ICD9_CODE_cleaned'] == str(icd_code)]['SHORT_TITLE'].values[0]
    icd_short_titles.append(short_title)

# plot heatmap of the top 10 icd codes in each topic
plt.figure(num=None, figsize=(8, 16), dpi=100, facecolor='w', edgecolor='k')
hm = heatmap(top_10_n_wks, cmap=sns.cm.rocket_r, yticklabels=icd_short_titles)