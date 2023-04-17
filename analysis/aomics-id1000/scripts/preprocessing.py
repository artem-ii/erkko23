# %% Import packages
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as io
io.renderers.default='browser'

# %% Global variables
ROOT_DIR ='/Users/artemii/obrain_lab/projects/obesity-types/grant-proposals/erkko2023'
DATA_DIR = os.path.join("datasets","aomics-id1000")
PLOTS_DIR = os.path.join("analysis","aomics-id1000", "plots")
PROCESSED_DATA_DIR = os.path.join("analysis","aomics-id1000", "processed-data")

participants_file = os.path.join(ROOT_DIR, DATA_DIR, "aomics_ID1000_participants.tsv")
data = pd.read_csv(participants_file, sep="\t")


# %% Preprocessing
data.describe()


# %%% BMI and age distribution

# Seems fair
data['BMI'].hist(bins=50)

bmi = data['BMI']
age = data['age']
fig, ax = plt.subplots()

ax.scatter(bmi, age)
ax.set_xlabel('BMI')
ax.set_ylabel('Age')
ax.set_title('AOMIC ID1000 Participants')

plt.show()
plot_filename = "age-vs-bmi-scatter.png"
fig.savefig(os.path.join(ROOT_DIR, PLOTS_DIR, plot_filename), dpi=200)

# %%% Filtering
# There are participants with zero BMI, need to remove them

data_filtered = data[data['BMI'] != 0]

# Only psychometric numerical data for PCA

data_psy = data_filtered.iloc[:,6:20]

# Also there are some NAs, replace with medians

for col in data_psy.columns:
    median = data_psy[col].median()
    data_psy[col].fillna(median, inplace=True)


# %% PCA
# Preliminary analysis. I propose to make further steps in separate files

from sklearn.decomposition import PCA

# %%% Scale data

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(data_psy)
print(X.shape == data_psy.shape)

data_psy_transformed = pd.DataFrame(X, columns=data_psy.columns,
                                    index=data_psy.index)


# %%% Dim reduction and rough plot

def pca_and_scatter(df, filename, color, show_plot = False):
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)
    fig = px.scatter_matrix(components,
                            dimensions=range(2),
                            color=color,
                            color_continuous_scale=px.colors.sequential.Inferno,
                            width=1200,
                            height=700,
                            labels={'0': 'PC1', '1': 'PC2'})
    
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)',})
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    if show_plot:
        fig.show()
    fig.write_image(os.path.join(ROOT_DIR, PLOTS_DIR, filename), format='png')
    return pca

pca_and_scatter(data_psy_transformed,
                "pca-plot-all.png",
                color=data_filtered['BMI'])


# %%% Only use high BMI range with pca_and_scatter

def filter_by_bmi(bmi_threshold):
    data_psy_transformed_high_bmi = data_psy_transformed[
        data_filtered['BMI'] >= bmi_threshold]
    
    color_data = data_filtered[
        data_filtered['BMI'] >= bmi_threshold].loc[:, 'BMI']
    filename = "pca-plot-{0}-bmi-thrshld.png"
    pca = pca_and_scatter(data_psy_transformed_high_bmi,
                          filename.format(str(bmi_threshold)),
                          color=color_data)
    relevant_features = pd.DataFrame(abs(pca.components_),
                                     columns=data_psy_transformed.columns,
                                     index=['PC1', 'PC2'])
    relevant_features.idxmax(axis=1)
    relevant_features_pc1_sorted = relevant_features.transpose().sort_values(
        'PC1', ascending=False)
    relevant_features_pc2_sorted = relevant_features.transpose().sort_values(
        'PC2', ascending=False)
    report = [str("REPORT: BMI threshold: " + str(bmi_threshold) +
                  "   Shape of data: " +
                  str(data_psy_transformed_high_bmi.shape) +
                  "   " +
                  "Explained variance ratio: " +
                  str(pca.explained_variance_ratio_) +
                  "   " +
                  "Top PC1 features:" ),
              relevant_features_pc1_sorted.head(10),
              str("Top PC2 features:"),
              relevant_features_pc2_sorted.head(10)
              ]
    report_filename = os.path.join(ROOT_DIR,
                                   PROCESSED_DATA_DIR,
                                   filename.format(str(bmi_threshold)) +
                                   "report.txt"
                                   )
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(str(report))
    f.close()
    return [pca, report]


filter_by_bmi(30)

'''
Out[50]: 
['REPORT: BMI threshold: 30   Shape of data: (101, 14)
 Explained variance ratio: [0.25656869 0.19763296]
 Top PC1 features:',
                              PC1       PC2
 IST_intelligence_total  0.512050  0.105461
 IST_fluid               0.498996  0.099803
 IST_memory              0.470079  0.131450
 IST_crystallised        0.340373  0.046690
 NEO_O                   0.230577  0.087037
 NEO_C                   0.196551  0.044136
 NEO_E                   0.159390  0.422916
 BAS_fun                 0.115210  0.387343
 background_SES          0.084815  0.079465
 NEO_A                   0.073596  0.338722,
 'Top PC2 features:',
                              PC1       PC2
 BAS_drive               0.000631  0.534312
 NEO_E                   0.159390  0.422916
 BAS_fun                 0.115210  0.387343
 NEO_A                   0.073596  0.338722
 BIS                     0.062203  0.312542
 NEO_N                   0.023044  0.261731
 BAS_reward              0.067027  0.219821
 IST_memory              0.470079  0.131450
 IST_intelligence_total  0.512050  0.105461
 IST_fluid               0.498996  0.099803]
'''

filter_by_bmi(25)

'''
Out[51]: 
['REPORT: BMI threshold: 25   Shape of data: (312, 14)
 Explained variance ratio: [0.27188917 0.18023215]
 Top PC1 features:',
                              PC1       PC2
 IST_intelligence_total  0.511162  0.080655
 IST_fluid               0.481714  0.077459
 IST_crystallised        0.431578  0.049808
 IST_memory              0.412949  0.081534
 background_SES          0.220992  0.000210
 NEO_O                   0.219794  0.014739
 NEO_C                   0.130725  0.136115
 NEO_E                   0.103249  0.479424
 NEO_A                   0.102212  0.241979
 BAS_fun                 0.083956  0.352622,
 'Top PC2 features:',
                              PC1       PC2
 NEO_E                   0.103249  0.479424
 BAS_drive               0.002242  0.434810
 BIS                     0.063988  0.394708
 NEO_N                   0.019789  0.386770
 BAS_fun                 0.083956  0.352622
 NEO_A                   0.102212  0.241979
 BAS_reward              0.054732  0.228986
 NEO_C                   0.130725  0.136115
 IST_memory              0.412949  0.081534
 IST_intelligence_total  0.511162  0.080655]
'''

# Plots are saved in plots folder with corresponding names
# Reports are saved in processed data folder with corresponding names