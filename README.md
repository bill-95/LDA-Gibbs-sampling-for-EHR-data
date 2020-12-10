# LDA Gibbs sampling for EHR data
Implemetation of Gibbs sampling algorithm for learning a Latent Dirichlet Allocation topic model on EHR (ICD-codes) data.
To run the code, clone repository, and run, `python lda_gibbs_sampling.py` .

You'll need to install the following packages:
- pandas - for loading the data
- numpy - for efficient computations
- seaborn - for visualizations
- matplotlib - for visualizations

The implementation is based on the following article.

> Darling, William M. "A theoretical and practical implementation tutorial on topic modeling and gibbs sampling." Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies. 2011.

The data used for this example can be obtained from: https://mimic.physionet.org/

> MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available at: http://www.nature.com/articles/sdata201635

## An example of learned topics from EHR ICD9-codes plotted as a heatmap
![Alt text](top_words_by_topic.png?raw=true "Top words by topic")
