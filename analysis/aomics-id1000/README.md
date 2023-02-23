# AOMICS ID1000 data analysis
## Lab Journal 

### Feb 22, 2023
@artem-ii

Created repo, created this lab journal


### Feb 23, 2023
@artem-ii

1. Set up spyder with conda. Installed miniconda instead of anaconda. Maybe the issue with space usage was related to me not executing `conda clear` regularly.
2. Performed basic PCA analysis. Can see heterogeneity. Seems that there may be a central cluster with overweight and lateral clusters with obesity.

Interestingly, we have "emotional" and "cognitive" PCs again (see report files in `processed-data` folder). In this dataset we explain about 50 percent of variance with 2 PCs when taking BMI >= 25 or >= 30

TODO:
1. Check if coloring works correctly of PCA scatters (probably already on normal PCA analysis)
2. Look at how age is distributed (although the range is nicely small)