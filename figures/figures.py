import cartopy.crs as ccrs
import glob
import json
from matplotlib import cm
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import sys
import torch

from mpl_toolkits.basemap import Basemap
from chapydette import feature_generation

sys.path.append('..')
from . import figure_utils
from utils import utils

params = {'font.family': 'serif', 'font.serif': 'Times New Roman'}
plt.rcParams.update(params)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


def cruises_map(background_day, background_file, cruises, data_dir, save_dir=None, zoom=False):
    """
    Create a map with the locations of the cruises, with the temperature of the ocean in the background.

    :param background_day: Day of the year to use from the background file.
    :param background_file: File with the background temperatures. You can download it here:
                            https://psl.noaa.gov/repository/entry/show?max=12&entryid=synth%3Ae570c8f9-ec09-4e89-93b4-babd5651e7a9%3AL25vYWEub2lzc3QudjIuaGlnaHJlcw%3D%3D&ascending=true&orderby=name&showentryselectform=true
    :param cruises: List of cruises to plot on the map.
    :param data_dir: Directory where the physical data is stored.
    :param save_dir: Directory where the figure will be saved.
    :param zoom: Whether to zoom in on the northeast part of the Pacific Ocean.
    """
    # Load the cruise location data
    lons_cruises, lats_cruises, lons_scope16, lats_scope16 = [], [], [], []
    for cruise_num, cruise in enumerate(cruises):
        phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
        if cruise != 'SCOPE_16':
            lons_cruises.extend(list(phys_data['longitude']))
            lats_cruises.extend(list(phys_data['latitude']))
        else:
            lons_scope16.extend(list(phys_data['longitude']))
            lats_scope16.extend(list(phys_data['latitude']))

    lons_cruises, lats_cruises = np.array(lons_cruises), np.array(lats_cruises)
    lons_scope16, lats_scope16 = np.array(lons_scope16), np.array(lats_scope16)

    # Load the background temperatures
    bg_data = Dataset(background_file, mode='r')
    bg_lons, bg_lats = np.meshgrid(bg_data['lon'], bg_data['lat'])
    bg_mask = np.ma.masked_where(np.array(bg_data['sst']) < -100, bg_data['sst'])

    # Convert all longitudes to longitudes east
    lons_cruises[lons_cruises > 0] = lons_cruises[lons_cruises > 0] - 360
    bg_lons = bg_lons - 360

    # Create the map
    plt.clf()
    plt.figure(figsize=(20, 12))
    if not zoom:
        m = Basemap(projection='merc', llcrnrlat=-50, urcrnrlat=70, llcrnrlon=-260, urcrnrlon=-20, lat_ts=-5,
                    resolution='l')
    else:
        m = Basemap(projection='merc', llcrnrlat=10, urcrnrlat=65, llcrnrlon=-170, urcrnrlon=-120, lat_ts=-5,
                    resolution='l')
    m.drawcoastlines()
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color=plt.get_cmap('Pastel1').colors[5], lake_color='white', alpha=0.3)
    m.drawparallels(np.arange(-90, 90, 20), labels=[1, 1, 0, 1], fontsize=22)
    m.drawmeridians(np.arange(-180, 180, 20), labels=[0, 0, 0, 1], fontsize=22)

    # Add the background
    bg_x, bg_y = m(bg_lons, bg_lats)
    cm1 = m.pcolor(bg_x, bg_y, bg_mask[background_day], cmap=plt.get_cmap('Blues'), alpha=0.3, snap=True)

    # Plot the cruise tracks
    Xm_bio, Ym_bio = m(lons_cruises, lats_cruises)
    Xm_scope16, Ym_scope16 = m(lons_scope16, lats_scope16)
    m.scatter(Xm_bio, Ym_bio, s=2, color=plt.get_cmap('Accent').colors[-3])
    m.scatter(Xm_scope16, Ym_scope16, s=40, color='black')

    # Add the colorbar, update some plot parameters, and save the plot
    cbar = plt.colorbar(cm1, pad=0.081)
    cbar.set_alpha(0.3)
    cbar.ax.set_ylabel(r'Temperature ($^\circ$C)', fontsize=32)
    cbar.ax.tick_params(labelsize=22)

    plt.tick_params(axis='both', which='major', labelsize=32)

    if save_dir is not None:
        if not zoom:
            plt.savefig(os.path.join(save_dir, 'cruises_map.jpg'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_dir, 'cruises_map_zoom.jpg'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def stacked_bar_chart(cruise, data_dir, ncp, results_dir, arrow_cps_idxs=[], save_dir=None):
    """
    Create a plot with the histograms of the population labels over the course of the cruise and overlay the estimated
    change points.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncp: Number of change points to use in the plot.
    :param results_dir: Directory where the estimated change points are stored.
    :param arrow_cps_idxs: Indices of the change points above which to put an arrow on the outside of the plot.
    :param save_dir: Directory where the figure will be saved.
    :return: cum_dists_cps: The distance into the cruise at which each change point occurred.
    """
    # Load the data and the estimated change points
    bio_data = pd.read_parquet(os.path.join(data_dir, cruise + '_bio.parquet'))
    times = np.array(pd.Series(bio_data['date']).astype('category').cat.codes.values + 1)
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    cps_file = glob.glob1(os.path.join(results_dir, cruise), 'cps_bio*rule-of-thumb_*')
    if len(cps_file) > 1:
        print('WARNING: More than 1 results file for cruise', cruise)
    cps_bio = json.load(open(os.path.join(results_dir, cruise, cps_file[0]), 'r'))['cps_bio'][str(ncp)]
    dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])
    cum_dists_cps = np.cumsum([0] + [dists[cps_bio[0]]]
                              + [dists[cps_bio[i+1]]-dists[cps_bio[i]] for i in range(len(cps_bio)-1)]
                              + [dists[-1]-dists[cps_bio[-1]]])

    # Generate the histograms for each time point
    hists, labels = figure_utils.get_pop_hists(bio_data['pop'].tolist(), times,
                                               unique_pop=['picoeuk', 'synecho',  'prochloro'], normalize=True)

    # Plot the histograms, overlay the change points, and add arrows on top of the plot
    colors = ['#3eddf2', '#e542f4', '#D0D697']
    labels = ['Picoeukaryotes', 'Synechococcus', 'Prochlorococcus']

    plt.clf()
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    plt.fill_between(dists[:-1], 0, np.sum(hists[:, 0:1], axis=1)[:-1], color=colors[0], label=labels[0])
    plt.fill_between(dists[:-1], np.sum(hists[:, 0:1], axis=1)[:-1], np.sum(hists[:, 0:2], axis=1)[:-1],
                     color=colors[1], label=labels[1])
    plt.fill_between(dists[:-1], np.sum(hists[:, 0:2], axis=1)[:-1], 1, color=colors[2], label=labels[2])

    for dist in cum_dists_cps[1:-1]:
        plt.axvline(x=dist, ls='-', c='black')

    for idx in arrow_cps_idxs:
        ax.annotate('', xy=(dists[cps_bio[idx]], 1.0), xycoords='data', xytext=(dists[cps_bio[idx]], 1.075),
                    arrowprops=dict(facecolor='black', shrink=0.1, width=1.0, headwidth=7.0, headlength=7.0))

    # Edit the axis limits and labels, along with the legend and save the figure
    plt.xlabel('Alongtrack distance (km)', fontsize=20)
    plt.ylabel('Fraction of observed population', fontsize=20)
    plt.xlim(0, dists[-1])
    plt.ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=16)
    plt.gcf().subplots_adjust(bottom=0.15)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, cruise + '_stacked_bar_' + str(ncp) + '.svg'), format='svg')
    else:
        plt.show()
    plt.close()

    return cum_dists_cps


def cytograms(cruise, data_dir, idxs, x_label, y_label, x_var, y_var, save_dir=None):
    """
    Make four cytograms in a 2x2 grid of y_var vs. x_var for each time index in idxs.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param idxs: Indices of the times at which the cytograms should be plotted.
    :param x_label: x-axis label.
    :param y_label: y-axis label.
    :param x_var: Variable to plot on the x-axis. This should be a column name in the biological data.
    :param y_var: Variable to plot on the y-axis. This should be a column name in the biological data.
    :param save_dir: Directory where the figure will be saved.
    """
    # Load the data
    bio_data = pd.read_parquet(os.path.join(data_dir, cruise + '_bio.parquet'))
    bio_data['Label'] = bio_data['pop'].map({'picoeuk': 'Picoeukaryote',
                                             'prochloro': 'Prochlorococcus',
                                             'synecho': 'Synechococcus',
                                             'unknown': 'Unknown',
                                             })
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])
    times = np.array(pd.Series(bio_data['date']).astype('category').cat.codes.values + 1)

    # Make one plot for each time index in idxs
    colors = ['#E0E0E0', '#e542f4', '#D0D697', '#3eddf2']
    customPalette = sns.set_palette(sns.color_palette(colors))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.4, 6.4))

    for idx, ax in zip(idxs, [ax1, ax2, ax3, ax4]):
        plt.sca(ax)
        df = bio_data.loc[times == idx]
        df = df.sort_values(by=['Label'], ascending=False)
        df = df[df.Label != 'Bead']
        g = sns.scatterplot(x=x_var, y=y_var, data=df, hue='Label', s=15, palette=customPalette, ax=ax, legend=False)
        g.set(xlabel=None, ylabel=None)
        g.set(xscale='log', yscale='log')
        ax.set_title('%0.0f km' % np.round(dists[idx]), fontsize=16)
        ax.set_xlim((np.min(bio_data[x_var]) / 2, np.max(bio_data[x_var]) * 2))
        ax.set_ylim((np.min(bio_data[y_var]) / 2, np.max(bio_data[y_var]) * 2))
        ax.set_xticks([10**i for i in range(5)])
        ax.set_yticks([10**i for i in range(5)])
        plt.tick_params(axis='both', which='major', labelsize=16)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(x_label, fontsize=18, labelpad=20)
    plt.ylabel(y_label, fontsize=18, labelpad=20)

    # Create dummy Line2D objects for legend
    h1 = plt.Line2D([0], [0], marker='o', markersize=np.sqrt(20), color='b', linestyle='None')
    h2 = plt.Line2D([0], [0], marker='o', markersize=np.sqrt(20), color='r', linestyle='None')
    h3 = plt.Line2D([0], [0], marker='o', markersize=np.sqrt(20), color='o', linestyle='None')
    h4 = plt.Line2D([0], [0], marker='o', markersize=np.sqrt(20), color='g', linestyle='None')

    # Make and edit the legend and save the figure
    legend = fig.legend([h1, h2, h3, h4],
               labels=['Prochlorococcus', 'Synechococcus', 'Picoeukaryote', 'Unknown'],
               loc='lower center',
               bbox_to_anchor=(0.5, -0.018),
               borderaxespad=0.1,
               title=None,
               fontsize=16,
               ncol=2,
               frameon=False,
               )
    for i in range(len(legend.legendHandles)):
        legend.legendHandles[i]._sizes = [30]
    for marker, color in zip(legend.legendHandles, [colors[2], colors[1], colors[3], colors[0]]):
        marker.set_color(color)
    plt.setp(legend.get_title(), fontsize='18')

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.25)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, cruise + '_cytogram_' + y_var + '_vs_' + x_var + '_' + str(idxs[0]) +
                                 '.svg'), format='svg')
    else:
        plt.show()
    plt.close()


def histograms(cruise, data_dir, idxs, label, var, legend=True, save_dir=None):
    """
    Generate kernel density estimates for a specific variable at the specified time indices.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the biological data is stored.
    :param idxs: Indices of the point clouds (times) to use in the plot.
    :param label: Label corresponding to the variable to use in the plot.
    :param var: Variable to use in the plot.
    :param legend: Whether to include a legend (True) or not (False).
    :param save_dir: Directory where the figure will be saved.
    """
    # Load the data
    bio_data = pd.read_parquet(os.path.join(data_dir, cruise + '_bio.parquet'))
    bio_data['Label'] = bio_data['pop'].map({'picoeuk': 'Picoeukaryote',
                                             'prochloro': 'Prochlorococcus',
                                             'synecho': 'Synechococcus',
                                             'unknown': 'Unknown',
                                             })
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])
    times = np.array(pd.Series(bio_data['date']).astype('category').cat.codes.values + 1)

    # Add one density estimate for each time index in idxs
    colors = ['#01FEA7', '#52AD72', 'pink', 'palevioletred']
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    for i, idx in enumerate(idxs):
        df = bio_data.loc[times == idx]
        df = df.sort_values(by=['Label'], ascending=False)
        df = df[df.Label != 'Bead']
        df = df[[var]]
        ls = '-' if i < 2 else '--'
        sns.kdeplot(data=df, x=var, log_scale=True, color=colors[i], label='%0.0f km' % np.round(dists[idx]), ls=ls,
                    lw=3)
        plt.xlim((np.min(bio_data[var]), np.max(bio_data[var])))
        plt.xticks([10**i for i in range(5)])
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.xlabel(label, fontsize=30)
        plt.ylabel('Density (in log space)', fontsize=30)

    # Adjust the legend and save the plot
    if legend:
        plt.legend(frameon=False, fontsize=24)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, cruise + '_histogram_' + var + '_' + str(idxs[0]) + '.svg'), format='svg')
    else:
        plt.show()
    plt.close()


def mean_element_plots(cruise, data_dir, features_dir, idxs, x_label, y_label, x_var, y_var, grid_size=(100, 100),
                      percentiles=[], plot_centroids=False, projection_dim=128, vmaxes=None, save_dir=None):
    """
    Create a mean element figure for a given cruise. The figure will be of one of two forms. Either it will consist of
    (1) a plot of the mean element of a point cloud at four different indices, along with the difference in the first two
    and latter two mean elements; or (2) a plot of the mean element of a point cloud at a single index.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param features_dir: Directory where the biological features are stored.
    :param idxs: List of indices of the point clouds to use in the plots. This should be a list of lists.
    :param x_label: x-axis label.
    :param y_label: y-axis label.
    :param x_var: Variable to plot on the x-axis. This should be a column name in the biological data.
    :param y_var: Variable to plot on the y-axis. This should be a column name in the biological data.
    :param grid_size: Size of the grid to use when computing the mean elements.
    :param percentiles: List of percentiles of the third variable (that's not x_var or y_var) to use when computing the
                        mean elements.
    :param plot_centroids: Whether to overly the centroids from k-means that were used in the Nyström method (True) or
                           not (False).
    :param projection_dim: Dimension of the biological features
    :param vmaxes: List of largest color values to use in the plots
    :param save_dir: Directory where the figure will be saved.
    """

    # Load the original data and the parameters used in generating the features
    bio_data = pd.read_parquet(os.path.join(data_dir, cruise + '_bio.parquet'))
    features_path = os.path.join(features_dir, cruise + '_features_' + str(projection_dim) + '.pickle')
    cruise_features = pickle.load(open(features_path, 'rb'))
    scaler = cruise_features['scaler']
    centroids = cruise_features['centroids']
    bw = cruise_features['bandwidth']
    times = np.array(pd.Series(bio_data['date']).astype('category').cat.codes.values)
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])

    var_to_idx = {'fsc_small': 0, 'chl_small': 1, 'pe': 2}
    colindex_x, colindex_y = var_to_idx[x_var], var_to_idx[y_var]
    z_var = set(var_to_idx.keys()).difference({x_var, y_var})
    colindex_z = var_to_idx[list(z_var)[0]]

    # Generate the grid for the plot
    bio_data = np.asarray(bio_data[['fsc_small', 'chl_small', 'pe']])
    bio_data = np.log10(bio_data)
    bio_data = torch.from_numpy(scaler.transform(bio_data))

    xgrid = np.linspace(torch.min(bio_data[:, colindex_x]) - 0.1, torch.max(bio_data[:, colindex_x]) + 0.1,
                        num=grid_size[0])
    ygrid = np.linspace(torch.min(bio_data[:, colindex_y]) - 0.1, torch.max(bio_data[:, colindex_y]) + 0.1,
                        num=grid_size[1])

    # Generate the axes for the plot. The code assumes that either the figure is going to be 2x3 or 1x1.
    if len(idxs) == 6:
        fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(6.4*4, 6.4*2))
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    elif len(idxs) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6.4*1.1, 6.4))
        axes = [ax]
    else:
        raise NotImplementedError

    grids = []
    if len(idxs) == 6:
        ax_label_fontsize = 44
        ax_tick_fontsize = 32
        cbar_tick_fontsize = 24
        title_fontsize = 36
    else:
        ax_label_fontsize = 23
        ax_tick_fontsize = 20
        cbar_tick_fontsize = 16
        title_fontsize = 24

    for idx_num, (ax, idx, percentile, vmax) in enumerate(zip(axes, idxs, percentiles, vmaxes)):
        plt.sca(ax)

        if idx_num < 4:
            # Get the relevant subset of the data
            pct_z = np.percentile(bio_data[:, colindex_z], percentile)
            data_idxs = np.where(times == idx)[0]
            bio_data_subset = bio_data[data_idxs, :].to(DEVICE)

            # Compute the value of the mean element at every point in the grid (after standardizing the grid points)
            mean_element_grid = np.zeros(grid_size)
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    point = np.array([xgrid[i], ygrid[j], pct_z])
                    point = torch.from_numpy(scaler.transform(point.reshape(1, -1))).to(DEVICE)
                    rbf_values, _ = feature_generation.rbf_kernel(bio_data_subset, point, bandwidth=bw)
                    mean_element_grid[i, j] = torch.mean(rbf_values.cpu()).item()

            # Plot the mean element and add a colorbar
            pcol = ax.pcolor(xgrid, ygrid, mean_element_grid, linewidth=0, rasterized=True, vmax=vmax)
            pcol.set_edgecolor('face')
            cbar = fig.colorbar(pcol)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(cbar_tick_fontsize)

            # Add a title and possibly an xlabel and centroids, and adjust the tick labels
            if len(idxs) != 6:
                ax.set_xlabel(x_label, fontsize=ax_label_fontsize, labelpad=5)
            if plot_centroids:
                centroids = scaler.inverse_transform(centroids)
                ax.scatter(centroids[:, colindex_x], centroids[:, colindex_y], marker='x', c='m')
            ax.tick_params(axis='both', which='major', labelsize=ax_tick_fontsize)
            ax.set_title('%0.0f km, PE at %0.2f quantile' % (np.round(dists[idx]), percentile / 100),
                         fontsize=title_fontsize)
            grids.append(mean_element_grid)
        elif idx_num == 4:
            mp_bound = np.maximum(abs(np.min(grids[1]-grids[0])), abs(np.max(grids[1]-grids[0])))
            pcol = ax.pcolor(xgrid, ygrid, grids[1] - grids[0], linewidth=0, rasterized=True, cmap='coolwarm',
                             vmin=-mp_bound, vmax=mp_bound)
        elif idx_num == 5:
            mp_bound = np.maximum(abs(np.min(grids[3] - grids[2])), abs(np.max(grids[3] - grids[2])))
            pcol = ax.pcolor(xgrid, ygrid, grids[3] - grids[2], linewidth=0, rasterized=True, cmap='coolwarm',
                             vmin=-mp_bound, vmax=mp_bound)
        if idx_num in [4, 5]:
            pcol.set_edgecolor('face')
            cbar = fig.colorbar(pcol)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(cbar_tick_fontsize)
            ax.set_title('Difference, PE at %0.2f quantile' % (percentile/100), fontsize=title_fontsize)
            if plot_centroids:
                centroids = scaler.inverse_transform(centroids)
                ax.scatter(centroids[:, colindex_x], centroids[:, colindex_y], marker='x', c='m')
            ax.tick_params(axis='both', which='major', labelsize=ax_tick_fontsize)

    # Add a big axis in order to add the axis labels, adjust the axis labels and figure, and save the figure
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    if len(idxs) == 6:
        plt.xlabel(x_label, fontsize=ax_label_fontsize, labelpad=20)
        plt.ylabel(y_label, fontsize=ax_label_fontsize, labelpad=20)
    else:
        plt.ylabel(y_label, fontsize=ax_label_fontsize)

    fig.tight_layout()
    if len(idxs) == 6:
        plt.gcf().subplots_adjust(right=0.88)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, cruise + '_mean_element_' + y_var + '_vs_' + x_var + '_%05d' % idxs[0] +
                                 '_' + str(percentiles[0]) + '_' + str(plot_centroids) + '.svg'))
    else:
        plt.show()
    plt.close()


def scale_bar(ax, length, location=(0.5, 0.05), linewidth=3):
    """
    Add a scale bar to a map. Adapted from code from here:
    https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot

    :param ax:  The axes to draw the scalebar on.
    :param length: The length of the scalebar in km.
    :param location: The center of the scalebar in axis coordinates, i.e., 0.5 is the middle of the plot.
    :param linewidth: The thickness of the scalebar.
    """
    # Get the limits of the axis in lat long and generate the scalebar location
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    x0, x1, y0, y1 = ax.get_extent(tmc)
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Plot the scalebar
    ax.plot([sbx - length * 500, sbx + length * 500], [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    ax.text(sbx, 10*sby, str(length) + ' km', transform=tmc, horizontalalignment='center', verticalalignment='bottom')
    ax.text(x0+(x1-x0)*0.1, (y1 - y0) * (location[1] - 0.03), u'\u25B2\nN', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom', zorder=2)


def cruise_path_plot(lat, lon, extent=None, addl_lat=None, addl_lon=None, addl_lat2=None, addl_lon2=None,
                     colors='rainbow', figure_size=None, save_path=None):
    """
    Plot the path of one cruise.

    :param lat: Latitudes.
    :param lon: Longitudes.
    :param extent: Vector of (min longitude, max longitude, min latitude, max latitude) for the plot.
    :param addl_lat: Latitudes of additional points to mark with an x on the plot.
    :param addl_lon: Longitudes of additional points to mark with an x on the plot.
    :param addl_lat2: Latitudes of additional points to mark with a + on the plot.
    :param addl_lon2: Longitudes of additional points to mark with a + on the plot.
    :param colors: Color(s) of the cruise path. Current options are 'rainbow' (the points at the beginning of the cruise
                   are red and those at the end are violet), 'blue', and 'gray-blue' (the points in the first half of
                   the cruise are gray and in the second half are blue).
    :param figure_size: Size to make the figure.
    :param save_path: Where to save the plot. If None, the plot is displayed on the screen.
    """
    plt.clf()
    if figure_size is not None:
        fig = plt.figure(figsize=figure_size)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    # Plot the cruise path
    if colors == 'rainbow':
        cm_subsection = np.linspace(0, 1, len(lon))
        colors = np.array([cm.gist_rainbow(x) for x in cm_subsection])
        plt.scatter(lon, lat, color=colors, transform=ccrs.Geodetic(), s=10, alpha=0.01)
    elif colors == 'blue':
        plt.scatter(lon, lat, color='blue', transform=ccrs.Geodetic(), s=10)
    elif colors == 'gray-blue':
        lat, lon = np.array(lat), np.array(lon)
        turnaround_point = np.argmax(lat)
        plt.plot(lon[:turnaround_point], lat[:turnaround_point], color='silver', transform=ccrs.Geodetic(), lw=3, zorder=0)
        plt.plot(lon[turnaround_point:], lat[turnaround_point:], '--', color='dodgerblue', transform=ccrs.Geodetic(), lw=3, zorder=1)
    else:
        raise NotImplementedError

    # Mark additional points on the cruise path
    if addl_lat is not None and addl_lon is not None:
        plt.scatter(addl_lon, addl_lat, color='black', transform=ccrs.Geodetic(), s=100, marker='x', zorder=2)
    if addl_lat2 is not None and addl_lon2 is not None:
        plt.scatter(addl_lon2, addl_lat2, color='red', transform=ccrs.Geodetic(), s=100, marker='+', zorder=3)

    # Set the extent of the plot, add a scale bar, and save the figure
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    scale_bar(ax, length=100, location=(0.8, 0.025), linewidth=3)

    if save_path is not None:
        plt.savefig(save_path + '.svg', format='svg')
    else:
        plt.show()
    plt.close()


def temp_salinity_vs_latitude_plot(cruise, data_dir, ncp_bio, ncp_phys, results_dir, true_phys=[False, True],
                                   axis_font_size=16, bbox=(1.05, -0.1), cps=[], figure_size=None, legend_font_size=14,
                                   tick_labelsize=16, save_path=None):
    """
    Plot temperature and salinity vs. latitude in a four-paneled figure and overlay biological and physical change
    points if ncp_bio and ncp_phys > 0. Also overlay any change points in cps. The top two plots will be for the first
    half of the cruise whereas the bottom two plots will be for the second half of the cruise. The left two plots will
    be for one number of change points and the right two plots will be for another number of change points.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncp_bio: List of the number of biological change points to use in the plots. If None, use the estimated
                    number.
    :param ncp_phys: List of the number of physical change points to use in the plots. If None, use the estimated number
                     (if true_phys is False) or the true number (if true_phys is True).
    :param results_dir: Directory where the estimated change points are stored.
    :param true_phys: Whether the physical change points are the annotated change points.
    :param axis_font_size: Font size for the axis labels.
    :param bbox: Location of the legend.
    :param cps: Change-point locations to mark with black vertical lines (indices after extracting the subset).
    :param figure_size: Size to make the figure.
    :param legend_font_size: Font size for the legend.
    :param tick_labelsize: Size of the axis tick labels.
    :param save_path: Where to save the plot. If None, the plot is displayed on the screen.
    :return: Tuple consisting of:

        * first_cps_bio: Indices corresponding to the first set of biological change points.
        * first_cps_phys: Indices corresponding to the first set of physical change points.
        * dists: Cumulative distances traveled throughout the cruise.
    """
    # Load the location data and the estimated change points
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    if ncp_phys is None:
        ncp_phys = len(json.load(open(os.path.join(data_dir, cruise + '_annotated_phys_cps.json'), 'r')))
    dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])
    lats = np.array(phys_data[['latitude']])

    # Set up the axes and plot colors
    plt.clf()
    if figure_size is None:
        fig, axes = plt.subplots(2, 2, sharey=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=figure_size, sharey=True)

    temp_color = plt.get_cmap('Set3').colors[3]
    salin_color = plt.get_cmap('Set3').colors[4]

    for idx in range(len(axes[0])*len(axes[1])):
        ax1 = axes[idx // 2, idx % 2]
        # Obtain the relevant subset of data and change points for panel with index idx
        if idx < 2:
            subset = (0, np.argmax(lats))
        else:
            subset = (np.argmax(lats), len(lats))
        if true_phys[idx % 2]:
            cps_phys = json.load(open(os.path.join(data_dir, cruise + '_annotated_phys_cps.json'), 'r'))
        elif ncp_phys[idx % 2] > 0:
            cps_phys = json.load(open(os.path.join(results_dir, cruise, 'cps_phys.json'), 'r'))['cps_phys'][
                str(ncp_phys[idx % 2])]
        else:
            cps_phys = []

        if ncp_bio[idx % 2] is None:
            est_cp_results = pickle.load(open(os.path.join(results_dir, 'estimated_ncp.pickle'), 'rb'))
            ncp_bio[idx % 2] = int(est_cp_results.loc[est_cp_results['cruise'] == 'SCOPE_16']['n_est_cps_penalty'])
        cps_file = glob.glob1(os.path.join(results_dir, cruise), 'cps_bio*rule-of-thumb_*')
        if len(cps_file) > 1:
            print('WARNING: More than 1 results file for cruise', cruise)
        if ncp_bio[idx % 2] > 0:
            cps_bio = json.load(open(os.path.join(results_dir, cruise, cps_file[0]), 'r'))['cps_bio'][str(ncp_bio[idx % 2])]
        else:
            cps_bio = []

        if idx == 0:
            first_cps_bio = cps_bio
            first_cps_phys = cps_phys

        # Plot the temperature and salinity data vs. latitude
        ax1.plot(lats[subset[0]:subset[1]], np.array(phys_data['temp'])[subset[0]:subset[1]], color=temp_color, alpha=0.5)
        ax1.tick_params('y', colors=temp_color, grid_alpha=0.5)
        ax1.set_ylim(5, 30)

        ax2 = ax1.twinx()
        ax2.plot(lats[subset[0]:subset[1]], np.array(phys_data['salinity'])[subset[0]:subset[1]], color=salin_color)
        ax2.tick_params('y', colors=salin_color)
        ax1.set_xlim(min(lats), max(lats))
        ax2.set_xlim(min(lats), max(lats))
        ax2.set_ylim(29, 38)

        # Overlay the change points
        for cp in cps_phys:
            if subset[0] <= cp < subset[1]:
                if cp == cps_phys[0]:
                    if not true_phys:
                        ax1.axvline(x=lats[int(cp)], ls='-', c='saddlebrown', label='Estimated physical change point', lw=3)
                    else:
                        ax1.axvline(x=lats[int(cp)], ls='-', c='saddlebrown', label='Physical change point', lw=3)
                else:
                    ax1.axvline(x=lats[int(cp)], ls='-', c='saddlebrown', lw=3)

        for cp in cps_bio:
            if subset[0] <= cp < subset[1]:
                if cp == cps_bio[0]:
                    ax1.axvline(x=lats[int(cp)], ls='--', c='darkseagreen', label='Estimated biological change point', lw=3)
                else:
                    ax1.axvline(x=lats[int(cp)], ls='--', c='darkseagreen', lw=3)

        for cp in cps:
            ax1.axvline(x=lats[subset[0]:subset[1]][int(cp)], ls='--', c='k', lw=3)

        # Edit the axis parameters, labels, and titles
        ax1.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        ax1.tick_params(axis='both', which='minor', labelsize=tick_labelsize)
        ax2.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        ax2.tick_params(axis='both', which='minor', labelsize=tick_labelsize)
        if idx < 2:
            ax1.set_xlabel('Latitude on outbound leg', fontsize=axis_font_size)
        else:
            ax1.set_xlabel('Latitude on return leg', fontsize=axis_font_size)
        if idx == 0:
            plt.title('10 Biological Change Points', fontsize=axis_font_size)
        elif idx == 1:
            plt.title('65 Biological Change Points', fontsize=axis_font_size)

    # Add the legend
    single_ax = fig.add_subplot(111, frameon=False)
    if len(cps_bio) > 0 or len(cps_phys) > 0:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], loc='lower center', fontsize=legend_font_size,
                   bbox_to_anchor=bbox, ncol=2, frameon=False,
                   fancybox=False, shadow=False)

    # Add the axis labels, adjust the plot, and save the figure
    single_ax.set_ylabel('Temperature ($^\circ$C)', color=temp_color, fontsize=axis_font_size, labelpad=30, alpha=0.5)
    single_ax_mirror = single_ax.twinx()
    single_ax_mirror.set_ylabel('Salinity (PSU)', color=salin_color, fontsize=axis_font_size, labelpad=30)
    single_ax.get_xaxis().set_ticks([])
    single_ax.get_yaxis().set_ticks([])
    single_ax_mirror.get_xaxis().set_ticks([])
    single_ax_mirror.get_yaxis().set_ticks([])
    single_ax_mirror.set_frame_on(False)

    plt.gcf().subplots_adjust(left=0.3, bottom=0.3, right=0.7)
    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + '.svg')
    plt.close()

    return first_cps_bio, first_cps_phys, dists


def estimated_vs_annotated_ncp(results_dir, annotate=False, save_dir=None):
    """
    Plot the number of estimated vs. annotated number of change points for both estimation methods.

    :param results_dir: Directory where the results for the estimation of the number of change points in each cruise are
                       stored.
    :param annotate: Whether to annotate the points with the names of the cruises.
    :param save_dir: Directory where the figures will be saved.
    :return: est_cp_results: Dataframe containing the results from the estimation of the number of change points.
    """
    # Load the data and compute an upper bound for the limits of the plot
    est_cp_results = pickle.load(open(os.path.join(results_dir, 'estimated_ncp.pickle'), 'rb'))
    max_cp = max(130, max(np.max(est_cp_results['n_cp']), np.max(est_cp_results['n_est_cps_penalty']),
                             np.max(est_cp_results['n_est_cps_rule_thumb'])) + 2)
    # For each estimation method, plot the results and save the resultant figure
    for key in ['n_est_cps_penalty', 'n_est_cps_rule_thumb']:
        plt.clf()
        plt.figure(figsize=(8, 8))
        fig, ax = plt.subplots()
        ax.scatter(est_cp_results['n_cp'], est_cp_results[key], c='#1f77b4')
        if annotate:
            for i, txt in enumerate(range(len(est_cp_results))):
                ax.annotate(est_cp_results.iloc[i]['cruise'], (est_cp_results.iloc[i]['n_cp'],
                                                               est_cp_results.iloc[i][key]))
        plt.plot(range(0, max_cp), range(0, max_cp), c='red')
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('Number of annotated change points', fontsize=20)
        plt.ylabel('Number of estimated change points', fontsize=20)
        plt.xlim((0, max_cp))
        plt.ylim((0, max_cp))
        plt.xticks((0, 25, 50, 75, 100, 125))
        plt.yticks((0, 25, 50, 75, 100, 125))
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, key + '_' + str(annotate) + '.svg'))
        else:
            plt.show()
        plt.close()

    return est_cp_results


def normalized_distance_histogram(cruises, data_dir, results_dir, save_dir=None):
    """
    Plot a histogram of the distance from each estimated change point to the nearest annotated physical change point,
    normalized by the average distance between change points within each cruise. Overlay the histogram with a dashed
    line indicating the histogram one would obtain with uniformly-spaced change points in each cruise (with the number
    of change points being the estimated number of change points).

    :param cruises: Names of the cruises to use.
    :param data_dir: Directory where the physical data is stored.
    :param results_dir: Top-level directory where the change-point results are stored.
    :param save_dir: Directory where the figure will be saved.
    :return: Tuple containing:

        * all_dists_est_to_true: The distance from each estimated biological change point to the nearest annotated
                                 physical change point.
        * all_dists_unif_to_true: The distance from each change point one would obtain based on a uniform segmentation
                                  to the nearest annotated physical change point.
        * avg_dists_true: The average distance between change points in each cruise.
    """
    # Load the results from estimating the number of change points
    est_cp_results = pickle.load(open(os.path.join(results_dir, 'estimated_ncp.pickle'), 'rb'))

    # Obtain the estimated and annotated change points (in terms of distance) for each cruise, along with the average
    # distance between annotated change points in each cruise
    all_est_cps = []
    all_true_cps = []
    avg_dists_true = []
    all_uniform_dists = []
    for cruise in cruises:
        cruise_results_files = glob.glob1(os.path.join(results_dir, cruise), 'cps_bio*rule-of-thumb_*')
        if len(cruise_results_files) > 1:
            print('WARNING: More than 1 results file for cruise', cruise)
        bio_results = json.load(open(os.path.join(results_dir, cruise, cruise_results_files[0]), 'r'))
        phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))

        true_cps_idx = json.load(open(os.path.join(data_dir, cruise + '_annotated_phys_cps.json'), 'r'))
        est_cps_idx = bio_results['cps_bio'][
            str(int(est_cp_results.loc[est_cp_results['cruise'] == cruise]['n_est_cps_penalty']))]
        dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])

        all_est_cps.append([dists[i] for i in est_cps_idx])
        all_true_cps.append([dists[int(i)] for i in true_cps_idx])
        avg_dists_true.append([dists[-1] / (len(true_cps_idx)+1)] * len(est_cps_idx))
        all_uniform_dists.append(np.linspace(0, dists[-1], num=len(est_cps_idx)+2, endpoint=True)[1:-1])

    # Compute the normalized distances from estimated biological change points to the nearest annotated physical change
    # points
    all_dists_est_to_true = [figure_utils.compute_min_dists(all_est_cps[i], all_true_cps[i])
                             for i in range(len(all_est_cps))]
    all_dists_unif_to_true = [figure_utils.compute_min_dists(all_uniform_dists[i], all_true_cps[i])
                             for i in range(len(all_uniform_dists))]
    normalized_dists = np.concatenate(all_dists_est_to_true) / np.concatenate(avg_dists_true)
    normalized_uniform_dists = np.concatenate(all_dists_unif_to_true) / np.concatenate(avg_dists_true)

    # Make a histogram of the normalized distances
    plt.clf()
    fig, ax = plt.subplots()
    bins_max = max(np.max(normalized_dists), np.max(normalized_uniform_dists)) + 0.5
    plt.hist(normalized_dists, alpha=0.5, histtype='bar', ec='black', bins=np.arange(0, bins_max, 0.25),
             color='#1f77b4', weights=np.zeros_like(normalized_dists) + 1. / normalized_dists.size)
    plt.hist(normalized_uniform_dists, histtype='step', bins=np.arange(0, bins_max, 0.25), linestyle='dashed',
             color='r', weights=np.zeros_like(normalized_uniform_dists) + 1. / normalized_uniform_dists.size)
    plt.xlabel('Normalized distance', fontsize=20)
    plt.ylabel('Relative frequency of change points', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'histogram_normalized_distances.svg'))
    else:
        plt.show()

    return all_dists_est_to_true, all_dists_unif_to_true, avg_dists_true


def temp_salinity_vs_distance_plot(cruise, data_dir, ncp_bio, ncp_phys, results_dir, axis_font_size=16,
                                   bbox=(1.05, -0.1), cps=[], figure_size=None, legend_font_size=14,
                                   subset=(None, None), tick_labelsize=16, true_phys=False, save_path=None):
    """
    Plot temperature and salinity vs. distance traveled in a single figure and overlay biological and physical change
    points if ncp_bio and ncp_phys > 0. Also overlay any change points in cps.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncp_bio: Number of biological change points to use in the plot. If None, use the estimated number.
    :param ncp_phys: Number of physical change points to use in the plot. If None, use the estimated number (if
                     true_phys is False) or the true number (if true_phys is True).
    :param results_dir: Directory where the estimated change points are stored.
    :param axis_font_size: Font size for the axis labels.
    :param bbox: Location of the legend.
    :param cps: Change-point locations to mark with black vertical lines (indices after extracting the subset).
    :param figure_size: Size to make the figure.
    :param legend_font_size: Font size for the legend.
    :param subset: Subset of the data to plot.
    :param tick_labelsize: Size of the axis tick labels.
    :param true_phys: Whether the physical change points are the annotated change points.
    :param save_path: Where to save the plot. If None, the plot is displayed on the screen.
    :return: Tuple consisting of:

        * cps_bio: Indices corresponding to the biological change points.
        * cps_phys: Indices corresponding to the physical change points.
        * dists: Cumulative distances traveled throughout the cruise.
    """
    # Load the location data and the estimated change points
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    if ncp_bio is None:
        est_cp_results = pickle.load(open(os.path.join(results_dir, 'estimated_ncp.pickle'), 'rb'))
        ncp_bio = int(est_cp_results.loc[est_cp_results['cruise'] == 'SCOPE_16']['n_est_cps_penalty'])
    cps_file = glob.glob1(os.path.join(results_dir, cruise), 'cps_bio*rule-of-thumb_*')
    if len(cps_file) > 1:
        print('WARNING: More than 1 results file for cruise', cruise)
    if ncp_bio > 0:
        cps_bio = json.load(open(os.path.join(results_dir, cruise, cps_file[0]), 'r'))['cps_bio'][str(ncp_bio)]
    else:
        cps_bio = []
    if ncp_phys is None:
        ncp_phys = len(json.load(open(os.path.join(data_dir, cruise + '_annotated_phys_cps.json'), 'r')))
    if true_phys:
        cps_phys = json.load(open(os.path.join(data_dir, cruise + '_annotated_phys_cps.json'), 'r'))
    elif ncp_phys > 0:
        cps_phys = json.load(open(os.path.join(results_dir, cruise, 'cps_phys.json'), 'r'))['cps_phys'][str(ncp_phys)]
    else:
        cps_phys = []
    dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])

    # Plot the temperature and salinity vs. distance
    plt.clf()
    if figure_size is None:
        fig, ax1 = plt.subplots()
    else:
        fig, ax1 = plt.subplots(figsize=figure_size)

    temp_color = plt.get_cmap('Set3').colors[3]
    ax1.plot(dists[subset[0]:subset[1]], np.array(phys_data['temp'])[subset[0]:subset[1]], color=temp_color, alpha=0.5)
    ax1.set_xlabel('Alongtrack distance (km)', fontsize=axis_font_size)
    ax1.set_ylabel('Temperature ($^\circ$C)', color=temp_color, fontsize=axis_font_size, alpha=0.5)
    ax1.tick_params('y', colors=temp_color)

    ax2 = ax1.twinx()
    salin_color = plt.get_cmap('Set3').colors[4]
    ax2.plot(dists[subset[0]:subset[1]], np.array(phys_data['salinity'])[subset[0]:subset[1]], color=salin_color)
    ax2.set_ylabel('Salinity (PSU)', color=salin_color, fontsize=axis_font_size)
    ax2.tick_params('y', colors=salin_color)
    ax1.set_xlim(dists[subset[0]:subset[1]][0], dists[subset[0]:subset[1]][-1])
    ax2.set_xlim(dists[subset[0]:subset[1]][0], dists[subset[0]:subset[1]][-1])

    # Overlay the change points
    for cp in cps_phys:
        if cp == cps_phys[0]:
            if not true_phys:
                plt.axvline(x=dists[int(cp)], ls='-', c='saddlebrown', label='Estimated physical change point', lw=3)
            else:
                plt.axvline(x=dists[int(cp)], ls='-', c='saddlebrown', label='Ground-truth physical change point', lw=3)
        else:
            plt.axvline(x=dists[int(cp)], ls='-', c='saddlebrown', lw=3)

    for cp in cps_bio:
        if cp == cps_bio[0]:
            plt.axvline(x=dists[int(cp)], ls='--', c='darkseagreen', label='Estimated biological change point', lw=3)
        else:
            plt.axvline(x=dists[int(cp)], ls='--', c='darkseagreen', lw=3)

    for cp in cps:
        plt.axvline(x=dists[subset[0]:subset[1]][int(cp)], ls='--', c='k', lw=3)

    # Edit the axis parameters and legend and save the figure
    ax1.tick_params(axis='both', which='major', labelsize=tick_labelsize)
    ax1.tick_params(axis='both', which='minor', labelsize=tick_labelsize)
    ax2.tick_params(axis='both', which='major', labelsize=tick_labelsize)
    ax2.tick_params(axis='both', which='minor', labelsize=tick_labelsize)

    if len(cps_bio) > 0 or len(cps_phys) > 0:
        handles, labels = ax2.get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], loc='upper right', fontsize=legend_font_size,
                   bbox_to_anchor=bbox, ncol=2, frameon=False,
                   fancybox=False, shadow=False)

    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + '.svg', bbox_inches='tight')
    plt.close()

    return cps_bio, cps_phys, dists


def temp_vs_salinity_plot(cruise, data_dir, cps=None, subset=(None, None), save_path=None):
    """
    Create a plot of the temperature vs. salinity of the ocean in a specific subset of a cruise, with the input change
    points, cps, marked in black x's.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param cps: Indices of points to mark on the plot.
    :param subset: Subset of the data to plot.
    :param save_path: Where to save the plot. If None, the plot is displayed on the screen.
    """
    # Load the desired subset of the temperature and salinity data
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    temps = np.array(phys_data['temp'][subset[0]:subset[1]])
    salinities = np.array(phys_data['salinity'][subset[0]:subset[1]])

    # Plot the temperature vs. salinity and put an x at all of the elements of cps
    fig, ax = plt.subplots()
    cm_subsection = np.linspace(0, 1, len(temps))
    colors = np.array([cm.gist_rainbow(x) for x in cm_subsection])
    ax.scatter(salinities, temps, color=colors, s=10, picker=False)
    for cp in cps:
        ax.scatter(salinities[cp], temps[cp], color='black', marker='x', s=100, zorder=10)

    ax.set_xlabel('Salinity (PSU)', fontsize=20)
    ax.set_ylabel('Temperature ($^\circ$C)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + '.svg', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def stacked_bar_chart_robustness(cps_files, cps_labels, cruise, data_dir, ncp, reverse_legend=False, save_path=None):
    """
    Create a plot with the histograms of the population labels over the course of the cruise and overlay the estimated
    change points.

    :param cps_files: List of files containing the estimated sets of change points.
    :param cps_labels: Labels for the plot corresponding to the files containing the estimated sets of change points.
    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncp: Number of change points to use in the plot.
    :param reverse_legend: Whether or not to reverse the order in which entries appear in the legend.
    :param save_path: Where to save the plot. If None, the plot is displayed on the screen.
    :return: Tuple consisting of:

        * all_cps: A list of all lists of change points in the plot.
        * dists: Cumulative distances traveled along the cruise.
    """
    # Load the data and the estimated change points
    bio_data = pd.read_parquet(os.path.join(data_dir, cruise + '_bio.parquet'))
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])

    # Generate the histograms for each time point
    times = np.array(pd.Series(bio_data['date']).astype('category').cat.codes.values + 1)
    hists, labels = figure_utils.get_pop_hists(bio_data['pop'].tolist(), times,
                                               unique_pop=['picoeuk', 'synecho',  'prochloro'], normalize=True)

    # Plot the histograms, overlay the change points, and add arrows on top of the plot
    colors = ['#3eddf2', '#e542f4', '#D0D697']
    plt.clf()
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    alpha = 0.1
    plt.fill_between(dists[:-1], 0, np.sum(hists[:, 0:1], axis=1)[:-1], color=colors[0], alpha=alpha)
    plt.fill_between(dists[:-1], np.sum(hists[:, 0:1], axis=1)[:-1], np.sum(hists[:, 0:2], axis=1)[:-1],
                     color=colors[1], alpha=alpha)
    plt.fill_between(dists[:-1], np.sum(hists[:, 0:2], axis=1)[:-1], 1, color=colors[2], alpha=alpha)

    # Add the change points from each file
    colors = plt.cm.gist_rainbow_r(np.linspace(0, 1, len(cps_files)))
    markers = ['*', 'D', 'P', '>', 'p', '<', 's', '^', '8', 'v', 'o']
    all_cps = []
    for i, cps_file in enumerate(cps_files):
        cps = json.load(open(cps_file, 'r'))['cps_bio'][str(ncp)]
        ymin, ymax = ax.get_ylim()
        ax.scatter([dists[int(cp)] for cp in cps], [ymin + (i + 1) * (ymax - ymin) / (len(cps_files) + 2)] * len(cps),
                   label=cps_labels[i], s=40, color=colors[i], marker=markers[i])
        all_cps.append(cps)

    # Edit the axis labels and limits, along with the legend and save the figure
    plt.xlabel('Alongtrack distance (km)', fontsize=24)
    plt.ylabel('Fraction of observed population', fontsize=24)
    plt.xlim(0, dists[-1])
    plt.ylim(0, 1)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    handles, labels = ax.get_legend_handles_labels()
    if reverse_legend:
        ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18, frameon=False)
    else:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18, frameon=False)
    if 'RBF' not in labels:
        plt.gcf().subplots_adjust(bottom=0.15, right=0.8)
    else:
        plt.gcf().subplots_adjust(bottom=0.15, right=0.8)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

    return all_cps, dists


def variance_histogram(cruise, data_dir, ncp, results_dir_full, results_dir_subsample, subsample_of=10, save_dir=None):
    """
    Plot a histogram of the distances (in terms of indices divided by half of subsample_of) from each estimated
    change point in a set of subsampled cruises to the nearest change point estimated in the non-subsampled cruise.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncp: Number of change points in the full sample to base the plot on.
    :param results_dir_full: Directory where the change-point results on the full sample are stored.
    :param results_dir_subsample: Directory where the change-point results on the subsamples are stored.
    :param subsample_of: Number of subsamples previously generated.
    :param save_dir: Directory where the figure will be saved.
    :return: Tuple consisting of:

        * all_cps_subsamples_dists: Array with the estimated location of every change point in each subsample in terms
                                    of distance
        * hist: Output of the histogram plot
    """
    # Load the estimated change points in the full sample, collect the files corresponding to the subsamples, and
    # compute the distances traveled
    cps_file = glob.glob1(os.path.join(results_dir_full, cruise), 'cps_bio*rule-of-thumb_*')
    if len(cps_file) > 1:
        print('WARNING: More than 1 results file for cruise', cruise)
    cps_full_sample = json.load(open(os.path.join(results_dir_full, cruise, cps_file[0]), 'r'))['cps_bio'][str(ncp)]
    subsample_cps_files = glob.glob1(os.path.join(results_dir_subsample, cruise), '*_of_' + str(subsample_of) + '.json')
    subsample_cps_files = [os.path.join(results_dir_subsample, cruise, cps_file) for cps_file in subsample_cps_files]
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])

    # Collect all change points in the subsamples, in terms of both indices and distances traveled
    all_cps_subsamples = []
    all_cps_subsamples_dists = []
    for i, cps_file in enumerate(subsample_cps_files):
        cps = json.load(open(cps_file, 'r'))['cps_bio'][str(ncp)]
        cps = i + np.array(cps)*subsample_of
        all_cps_subsamples.append(cps)
        all_cps_subsamples_dists.append([dists[cp] for cp in cps])
    all_cps_subsamples = np.array(all_cps_subsamples)
    all_cps_subsamples_dists = np.array(all_cps_subsamples_dists)

    all_dists_est_to_true = figure_utils.compute_min_dists(all_cps_subsamples.flatten(), cps_full_sample)
    normalized_dists = all_dists_est_to_true/(subsample_of/2)

    # Make a histogram of the normalized distances
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins_max = np.max(normalized_dists) + 0.5
    hist = plt.hist(normalized_dists, alpha=0.5, histtype='bar', ec='black', bins=np.arange(0, bins_max, 1),
                    color='#1f77b4')
    plt.xlabel('Normalized distance from nearest change point in full sample', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'histogram_variance.svg'))
    else:
        plt.show()

    return all_cps_subsamples_dists, hist
