"""
Created on Mon Apr 17 12:40:39 2023

@author: artemii
"""
# RUN PREPROCESSING.PY AND PLOTTING-STYLE.PY FIRST!
 
NEW_MODEL = True
SAVE_FILES = True

filter_30 = filter_by_bmi(30)
filter_25 = filter_by_bmi(25)

components_nobmi_30 = filter_30[0].fit_transform(data_psy_transformed)
components_nobmi_25 = filter_25[0].fit_transform(data_psy_transformed)


from sklearn.mixture import BayesianGaussianMixture
n_components = 20


def pca_and_bgm(df, filename, color, bmi_vector, show_plot = False):
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)
    bgm_on_pca_nobmi = BayesianGaussianMixture(n_components=n_components,
                                               n_init=20,
                                               covariance_type="full",
                                               weight_concentration_prior=10000,
                                               init_params="k-means++",
                                               random_state=44,
                                               tol=1e-3,
                                               max_iter=1000)



    if NEW_MODEL:
        bgm_on_pca_nobmi.fit(components)
        cluster_list = list(zip(range(n_components), bgm_on_pca_nobmi.weights_))
    plot_gaussian_mixture(bgm_on_pca_nobmi, components)
    print(cluster_list)
    bgm_cluster_on_pca_nobmi = bgm_on_pca_nobmi.predict(components)
    print(bgm_cluster_on_pca_nobmi)
    data_bgm_clustered_on_pca_nobmi = df.copy()
    data_bgm_clustered_on_pca_nobmi['bgm_pca_cluster'] = bgm_cluster_on_pca_nobmi
    data_bgm_clustered_on_pca_nobmi['BMI'] = bmi_vector
    data_bgm_clustered_on_pca_nobmi.to_csv("clustered_behavioral_AOMIC_pca_based_nobmi.csv")
    return components

def filter_by_bmi_bgm(bmi_threshold):
    data_psy_transformed_high_bmi = data_psy_transformed[
        data_filtered['BMI'] >= bmi_threshold]
    bmi_vector = data_filtered['BMI'][
        data_filtered['BMI'] >= bmi_threshold]
    color_data = data_filtered[
        data_filtered['BMI'] >= bmi_threshold].loc[:, 'BMI']
    filename = "pca-plot-{0}-bmi-thrshld.png"
    pca = pca_and_bgm(data_psy_transformed_high_bmi,
                      filename.format(str(bmi_threshold)),
                      color=color_data,
                      bmi_vector=bmi_vector)
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
    print(pca)
    return [pca, report]

# %%

clustering_filtered = filter_by_bmi_bgm(25)



# %%
color_data = data_filtered[data_filtered['BMI'] >= 25].loc[:, 'BMI']
