# LDA Gibbs sampling for EHR data
Implemetation of Gibbs sampling algorithm for learning a Latent Dirichlet Allocation topic model on EHR (ICD-codes) data.
To run the code, clone repository, and run python lda_gibbs_sampling.py .

You'll need to the following packages:
- pandas - for loading the data
- numpy - for efficient computations
- seaborn - for visualizations
- matplotlib for visualizations

The implementation is based on the following article.

Darling, William M. "A theoretical and practical implementation tutorial on topic modeling and gibbs sampling." Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies. 2011.

## An example of learned topics from EHR ICD9-codes plotted as a heatmap
![Alt text](top_words_by_topic.png?raw=true "Top words by topic")
