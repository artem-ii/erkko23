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

from sklearn.decomposition import PCA

# %%% Scale data

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(data_psy)
print(X.shape == data_psy.shape)

data_psy_transformed = pd.DataFrame(X, columns=data_psy.columns,
                                    index=data_psy.index)


# %%% Dim reduction and rough plot

def pca_and_scatter(df, filename, color):
    
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
    fig.show()
    fig.write_image(os.path.join(ROOT_DIR, PLOTS_DIR, filename), format='png')
    return 0

pca_and_scatter(data_psy_transformed,
                "pca-plot-all.png",
                color=data_filtered['BMI'])
# %%% Only use high BMI range

def filter_by_bmi(bmi_threshold):
    data_psy_transformed_high_bmi = data_psy_transformed[
        data_filtered['BMI'] >= bmi_threshold]
    
    color_data = data_filtered[
        data_filtered['BMI'] >= bmi_threshold].loc[:, 'BMI']
    filename = "pca-plot-{0}-bmi-thrshld.png"
    pca_and_scatter(data_psy_transformed_high_bmi,
                    filename.format(str(bmi_threshold)),
                    color=color_data)
    return "Shape of data: " + str(data_psy_transformed_high_bmi.shape)


filter_by_bmi(30)

# Out[18]: (101, 14)


filter_by_bmi(25)

# Out[26]: (312, 14)

# Plots are saved in plots folder with corresponding names