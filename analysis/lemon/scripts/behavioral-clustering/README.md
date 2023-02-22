# LEMON Dataset Analysis for Sigrid Juselius 2022 Grant Application
## Data Cleaning and Behavioral clustering

*This README is an analysis log. Look to the bottom [**30 Nov 2022**](https://github.com/artem-ii/juselius22/tree/main/behavioral-clustering#clean-up-the-notebook-for-the-submission-of-the-proposal) for reproducing the clustering plot from the proposal*

---
### Clustering using Bayesian Gaussian Mixture model
file: LEMON-behavioral-clustering.ipynb

*Sat-Sun 29-30 Oct 2022*

#### What was done:

1. Dataset clustered using scikit-learn BayesianGaussianMixture
2. Dataset with bgm_clusters column saved to processed_data/clustered_behavioral_LEMON.csv
3. Feature importance tested using LogisticRegression processed_data/feature_importance_max(min)_logreg.csv


#### Comments:

##### BGM clustering step

Clustering seems to have found a pattern different from PCA, which is kind of expected and normal to me.  
Clusters differ by BMI (median), and are promising on that account at a first glance:


| bgm_cluster | BMI | Waist_cm |
|---|---|---|
| 0 | 24.609786 | 87.00 |
| 1 | 22.951595 | 84.00 |
| 2 | 24.676041 | 88.00 |
| 3 | 23.655986 | 78.30 |
| 4 | 22.459971 | 80.75 |
| 5 | 25.368990 | 84.50 |
| 6 | 23.468269 | 80.50 |
| 7 | 24.037299 | 80.50 |
| 8 | 24.811673 | 85.50 |
| 9 | 23.312965 | 78.80 |

However, the BGM algorithm is a variational inference algorithm. I provided 10 clusters and expected that it would only use a few. It would be seen as zero weights assigned to clusters. But I got the following result:

```python
array([0.07, 0.11, 0.09, 0.15, 0.08, 0.08, 0.13, 0.07, 0.11, 0.11])
```

To me that gives ambiguity as there are no zero-weight clusters (even when I added 20 instead of 10). Because of that I am thinking to try another algorithm like Gaussian Mixture or something simpler. The important step would also be to find optimal number of clusters, e.g. using an information criteria for Gaussian Mixture. Also introducing a BMI threshold and clustering only overweight subjects might be a good idea. PCA already shows behavioral differences between normal weight and overweight subjects. So with clustering I could focus only on overweight at this stage.  
In the proposal that's a good idea to emphasize the need for BIOBANK acsess as it has behavioral data as well together with wider BMI range.

###### To-do:
1. Try simpler clustering (e.g. simple Gaussian Mixture), explain cluster number mathematically
2. Split data and cluster only overweight (quite few for clustering?)

##### Cluster description/feature importance step (Logistic Regression)

I used multimodal logistic regression to assess which features drive which clusters. BMI or other anthropometric features were not among the top-1 ones. I constructed tables of features with maximal and minimal coefficients.  

Features of maximal coefficients for each cluster:
| Cluster | Feature with max coefficient |
|---|---|
| 0 | COPE_Acceptance |
| 1 | CVLT_1 |
| 2 | NEOFFI_Extraversion |
| 3 | TAP_WM_11 |
| 4 | FSoZu_Zuf |
| 5 | NYC-Q_lemon_21 |
| 6 | TeiQueSF_sociability |
| 7 | RWT_1 |
| 8 | TICS_WorkDiscontent |
| 9 | TAP_I_24 |

Features of minimal coefficients for each cluster:

| Cluster | Feature with min coefficient |
|---|---|
| 0 | MDBF_Day2_WM_Scale |
| 1 | MARS_Cognitive_Distraction |
| 2 | TICS_PressuereToPerform |
| 3 | COPE_Humor |
| 4 | CERQ_Acceptance |
| 5 | MDBF_Day2_GS_Scale |
| 6 | NYC-Q_lemon_14 |
| 7 | COPE_activeCoping |
| 8 | NYC-Q_lemon_18 |
| 9 | TAP_WM_11 |


###### To-do:
1. Read more on interpretation of clusters in multidimensional data
2. Read on logistic regression and maybe there are better algorithms for my aim?
3. What would be the best visualisation for cluster description? Maybe, plot simple 2D plots for parameters of interest and color with clusters? For example, plot BMI vs COPE_Acceptance
4. Read about all the tests/tasks which have high/low coefficients
5. Decide which clusters differ the most to compare neuroimaging. Representative instances? How to find them in Gaussian Mixture?

---
### Clustering on PCA-transformed data and cluster characterisation 
file: LEMON-behavioral-clustering.ipynb

*Mon-Thu 31-03 Nov 2022*

#### What was done:
1. PCA-transformed data clustered using Bayesian Gaussian Mixture model
2. Visualised scatters for several features that contribute to PCA
3. Identified features important for the clusters using logistic regression
4. Performed ANOVA Tukey HSD on BMI from 4 clusters
5. Visualised distributions of Age, Sex and Handedness across clusters  

#### Comments:

##### PCA based BGM clustering

Clustering looks good. The only weird thing was that it finds 5 clusters with every random seed tested except 42 and 43 (there it's 4 clusters).  

It is important to note that PCA with 2 components preserves only about 18% of variance. Performing PCA with 95% variance preservation threshold gives the need for 98 dimensions. Although this 2D PCA is probably still fine for visualisation, we may want to try different clustering, like BGM on full data performed above.  

```python
# In
pca.explained_variance_ratio_

# Out
array([0.10391384, 0.07820958])
```

Also, I have thought and read on the need of removing multicollinear variables for PCA analysis and concluded that it is probably not useful. Keeping everything as it is gives a more non-biased flavour. However, I should remove multicollinear variables in order to perform regression.  

Importantly, I have corrected the contributions of features by taking their absolute value into the account. Otherwise negative values are at the bottom which is not correct.  

###### To-do:
1. Create a consistent formatting theme for all the plots - read on matplotlib for that
2. Think more on multicollinearity - some features 

##### Important features for each cluster

Here I used logistic regression which I suppose is not appropriate. I believe linear model is needed here.  

And here the multicollinearity becomes important.

###### To-do:
1. Read on glm in statsmodels
2. Read on how to remove multicollinearity without using PCA (Or maybe just use PCA?)

##### ANOVA on BMI

That is clearly not optimal. I will need to infer this using a glm with proper modeling and corrections for multiple comparisons.

Also I think it's a good idea to include regressors of no interest (Age group, sex, handedness etc)

###### To-do:
1. Include regressors of no interest in the glm and test everything like that

---
### Remove all anthropometric values from PCA and define plotting style
files: LEMON-behavioral-clustering.ipynb	plotting_style.ipynb

*Sat-Sun 12-13 Nov 2022*
#### What was done:
1. Removed anthropometric variables from PCA data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
2. Created plotting_style.ipnb and plotted data 

---
### Clean up the notebook for the submission of the proposal
files: LEMON-behavioral-clustering.ipynb	processed_data/bgm_on_pca_nobmi.mdl	processed_data/components_nobmi.npy 

*Wed 30 Nov 2022*
#### What was done:
1. Legend with cluster sizes added to clustering plot
2. All analyses except PCA with no anthropometry and BGM clustering on it removed
3. Dumped BGM model and nobmi components array binary objects added

#### Comments:
Now repo should be suitable for clonning. To get a visualization of clusters from the proposal, use [`LEMON-behavioral-clustering.ipynb`](https://github.com/artem-ii/juselius22/blob/main/behavioral-clustering/LEMON-behavioral-clustering.ipynb) Jupyter Notebook. Use [Jupyter Lab](https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906) for better navigation with available outline. Set `SCRIPTS_PATH` to your local directory where the repo was cloned. Don't execute all the code, but just starting from **Plotting functions for behavioural clustering** section of the notebook. This should access all needed files from `processed_data` subdirectory and reproduce the plot.