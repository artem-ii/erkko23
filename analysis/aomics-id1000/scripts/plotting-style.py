# Tweaking the code from the book to adjust the style

# %% Import modules

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_name('Helvetica')

# %% Parameters and functions

'''
Functions based on Géron 2019
'''
one_legend = True


color_list_subtle = ['steelblue',
                     'wheat',
                     'coral']

color_list_detailed = ['darkblue',
                       'skyblue',
                       'darkorange',
                       'orangered']

color_list_very_detailed = ['darkblue',
                            'skyblue',
                            'olivedrab',
                            'gold',
                            'orangered']

cmap_subtle = colors.LinearSegmentedColormap.from_list(
    'juselius22_subtle', color_list_subtle)
cmap_detailed = colors.LinearSegmentedColormap.from_list(
    'juselius22_detailed', color_list_detailed)
cmap_very_detailed = colors.LinearSegmentedColormap.from_list(
    'juselius22_very_detailed', color_list_very_detailed)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]

    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=60, linewidths=8,
                color=circle_color, zorder=10, alpha=0.5)
    x = 0
    y = 0
    centroids_text = centroids.copy() - 0.5
    for marker in [0, 1, 3, 8]:
        plt.text(centroids_text[x, 0], centroids_text[y, 1],
                 str(marker), fontsize=20,
                 color=cross_color,
                 zorder=11, alpha=1)
        x += 1
        y += 1
        
    if not one_legend:
        x = 0
        y = 0
        cluster_size_coordinates = np.array([[-10, -4.5],
                                             [9, -4.5],
                                             [-8, 10],
                                             [6, 10]])
        for cluster_size in [127, 34, 15, 52]:
            plt.text(cluster_size_coordinates[x, 0], cluster_size_coordinates[y, 1],
                     "N="+str(cluster_size), fontsize=15,
                     color=cross_color,
                     zorder=11, alpha=1)
            x += 1
            y += 1
    else:
        legend_coordinates = [12.5, -5]
'''
        plt.text(legend_coordinates[0], legend_coordinates[1],
                 "0: N = 127\n1: N = 34\n3: N = 15\n8: N = 52", fontsize=11,
                 color=cross_color,
                 zorder=11, alpha=1)
'''
        
def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #plt.contourf(xx, yy, Z,
    #             norm=LogNorm(vmin=1.0, vmax=30.0),
    #             levels=np.logspace(0, 2, 12))
    #plt.contour(xx, yy, Z,
    #            norm=LogNorm(vmin=1.0, vmax=30.0),
    #            levels=np.logspace(0, 2, 12),
    #            linewidths=1, colors='k')
    
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=[6, 7])
    ax.contour(xx, yy, Z,
               linewidths=1, colors='black', linestyles='dashed')
    
    ax.scatter(X[:, 0], X[:, 1],
               c=data_filtered['BMI'],
               alpha=1, s=25, cmap=cmap
              )
    plot_centroids(clusterer.means_, clusterer.weights_)
    
    ax.set_xlabel('Principal component 1', fontproperties=font, fontsize=18)
    if show_ylabels:
        ax.set_ylabel('Principal component 2', fontproperties=font, fontsize=18)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    '''
    colorbar = fig.colorbar(scatter, ticks=[],
                            location='bottom',
                            anchor=(0.97, 3.25),
                            shrink=0.2, aspect=5)
    colorbar.set_label('Waist, cm', loc='center', size=12)
    '''
    if SAVE_FILES:
        plt.savefig('plots/behavioural_clusters_cmap_detailed.svg')

cmap = cmap_detailed

