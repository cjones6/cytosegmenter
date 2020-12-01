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

from mpl_toolkits.basemap import Basemap

sys.path.append('..')
from . import figure_utils
from utils import utils

params = {'font.family': 'serif', 'font.serif': 'Times New Roman'}
plt.rcParams.update(params)


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
            for i in range(len(phys_data) - 1, -1, -1):
                # Remove the points from DeepDOM that shouldn't be there
                if cruise == 'DeepDOM' and -10 < phys_data['latitude'][i] < -2 and phys_data['longitude'][i] < -33:
                    phys_data = phys_data.drop(phys_data.index[i])
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
        m = Basemap(projection='merc', llcrnrlat=-50, urcrnrlat=70,
                    llcrnrlon=-260, urcrnrlon=-20, lat_ts=-5, resolution='l')
    else:
        m = Basemap(projection='merc', llcrnrlat=10, urcrnrlat=65,
                    llcrnrlon=-170, urcrnrlon=-120, lat_ts=-5, resolution='l')
    m.drawcoastlines()
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color=plt.get_cmap('Pastel1').colors[5], lake_color='white', alpha=0.3)
    m.drawparallels(np.arange(-90, 90, 20), labels=[1, 1, 0, 1], fontsize=22)
    m.drawmeridians(np.arange(-180, 180, 20), labels=[0, 0, 0, 1], fontsize=22)

    # Add the background
    bg_x, bg_y = m(bg_lons, bg_lats)
    cm1 = m.pcolormesh(bg_x, bg_y, bg_mask[background_day], cmap=plt.get_cmap('coolwarm'), alpha=0.2)

    # Plot the cruise tracks
    Xm_bio, Ym_bio = m(lons_cruises, lats_cruises)
    Xm_scope16, Ym_scope16 = m(lons_scope16, lats_scope16)
    m.scatter(Xm_bio, Ym_bio, s=2, color=plt.get_cmap('Accent').colors[-3])
    m.scatter(Xm_scope16, Ym_scope16, s=40, color='black')

    # Add the colorbar, update some plot parameters, and save the plot
    cbar = plt.colorbar(cm1, pad=0.081)
    cbar.ax.set_ylabel(r'Temperature ($^\circ$C)', fontsize=32)
    cbar.ax.tick_params(labelsize=22)

    plt.tick_params(axis='both', which='major', labelsize=32)

    if save_dir is not None:
        if not zoom:
            plt.savefig(os.path.join(save_dir, 'temperature_map.jpg'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_dir, 'temperature_map_zoom.jpg'), bbox_inches='tight')
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
    """
    # Load the data and the estimated change points
    bio_data = pd.read_parquet(os.path.join(data_dir, cruise + '_bio.parquet'))
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
    times = np.array(pd.Series(bio_data['date']).astype('category').cat.codes.values + 1)
    hists, labels = figure_utils.get_pop_hists(bio_data['pop'].tolist(), times,
                                               unique_pop=['picoeuk', 'synecho',  'prochloro'], normalize=True)

    # Plot the histograms, overlay the change points, and add arrows on top of the plot
    colors = ['#3eddf2', '#e542f4', '#00e68a']
    labels = ['Picoeukaryotes', 'Synechococcus', 'Prochlorococcus']

    plt.clf()
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)
    plt.fill_between(dists[:-1], 0, np.sum(hists[:, 0:1], axis=1)[:-1], color=colors[0], label=labels[0])
    plt.fill_between(dists[:-1], np.sum(hists[:, 0:1], axis=1)[:-1], np.sum(hists[:, 0:2], axis=1)[:-1],
                     color=colors[1], label=labels[1])
    plt.fill_between(dists[:-1], np.sum(hists[:, 0:2], axis=1)[:-1], 1, color=colors[2], label=labels[2])
    for dist in cum_dists_cps:
        plt.axvline(x=dist, ls='-', c='black')

    for idx in arrow_cps_idxs:
        ax.annotate('', xy=(dists[cps_bio[idx]], 1.0), xycoords='data', xytext=(dists[cps_bio[idx]], 1.075),
                    arrowprops=dict(facecolor='black', shrink=0.1, width=1.0, headwidth=7.0, headlength=7.0))

    # Edit the axis limits and labels, along with the legend and save the figure
    plt.xlabel('Alongtrack distance (km)', fontsize=20)
    plt.ylabel('Fraction of observed population', fontsize=20)
    plt.xlim(0, dists[-1])
    plt.ylim(0, 1)
    plt.xticks(cum_dists_cps, np.array(np.round(cum_dists_cps), dtype='int'), rotation=40, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=16)
    plt.gcf().subplots_adjust(bottom=0.15)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, cruise + '_stacked_bar.svg'), format='svg')
    else:
        plt.show()
    plt.close()


def cytogram(cruise, data_dir, idxs, x_label, y_label, x_var, y_var, save_dir=None):
    """
    Make one cytogram of y_variable vs. x_variable per time index in idxs.

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
    colors = list(reversed(['#1f77b4', '#d62728', '#2ca02c', '#FDAD42']))
    customPalette = sns.set_palette(sns.color_palette(colors))

    for idx in idxs:
        df = bio_data.loc[times == idx]
        df = df.sort_values(by=['Label'], ascending=False)
        df = df[df.Label != 'Bead']
        df[x_var] = np.log(df[x_var])
        df[y_var] = np.log(df[y_var])

        g = sns.pairplot(x_vars=[x_var], y_vars=[y_var], data=df, hue='Label', height=6, plot_kws={"s": 15},
                         palette=customPalette, aspect=0.75)
        g._legend.remove()
        g.add_legend(label_order=sorted(g._legend_data.keys()), fontsize=22)
        g._legend.get_title().set_fontsize('22')

        for lh in g._legend.legendHandles:
            lh.set_sizes([50])
        plt.xlabel(x_label, fontsize=24)
        plt.ylabel(y_label, fontsize=24)
        plt.xlim((-0.1, 8))
        if y_var == 'chl_small':
            plt.ylim((-0.1, 8.1))
        elif y_var == 'pe':
            plt.ylim((-0.25, 6.75))
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.title('Distance traveled: %4.2fkm, n=%d' % (dists[idx], len(df)), fontsize=18)

        plt.gcf().subplots_adjust(bottom=0.13, top=0.95)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, cruise + '_cytogram_' + y_var + '_vs_' + x_var + '_%05d' % idx + '.svg'),
                        format='svg')
        else:
            plt.show()
        plt.close()


def lat_lon_plot(cruise, data_dir, ncp, results_dir, save_dir=None):
    """
    Plot the estimated change points in a cruise overlaid on a plots of the latitude and longitude of the cruise as a
    function of the distance traveled.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncp: Number of change points to use in the plot.
    :param results_dir: Directory where the estimated change points are stored.
    :param save_dir: Directory where the figure will be saved.
    :return: Tuple consisting of:

        * cps_bio: Estimated change points.
        * dists: Distances traveled.
        * phys_data['latitude']: Latitude of the cruise at each time point.
        * phys_data['longitude']: Longitude of the cruise at each time point.
    """
    # Load the location data and the estimated change points
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    cps_file = glob.glob1(os.path.join(results_dir, cruise), 'cps_bio*rule-of-thumb_*')
    if len(cps_file) > 1:
        print('WARNING: More than 1 results file for cruise', cruise)
    cps_bio = json.load(open(os.path.join(results_dir, cruise, cps_file[0]), 'r'))['cps_bio'][str(ncp)]
    dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])

    # Plot the latitude and longitude of the cruise path as a function of the distance traveled
    plt.clf()
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(dists, phys_data['latitude'], label='_nolegend_', color='gray')
    axarr[1].plot(dists, phys_data['longitude'], label='_nolegend_', color='gray')

    # Overlay the change points
    for axis in [axarr[0], axarr[1]]:
        for cp in cps_bio:
            if cp == cps_bio[0]:
                axis.axvline(x=dists[int(cp)], ls='--', c='darkseagreen', label='Estimated biological change point')
            else:
                axis.axvline(x=dists[int(cp)], ls='--', c='darkseagreen')

    # Edit the axis labels and legend and save the figure
    axarr[0].set_ylabel('Latitude', fontsize=16)
    axarr[1].set_ylabel('Longitude', fontsize=16)
    plt.xlabel('Alongtrack distance (km)', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, cruise + '_lat_lon.svg'), format='svg')
    else:
        plt.show()
    plt.close()

    return cps_bio, dists, phys_data['latitude'], phys_data['longitude']


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
    ax1.plot(dists[subset[0]:subset[1]], np.array(phys_data['temp'])[subset[0]:subset[1]], color=temp_color)
    ax1.set_xlabel('Alongtrack distance (km)', fontsize=axis_font_size)
    ax1.set_ylabel('Temperature ($^\circ$C)', color=temp_color, fontsize=axis_font_size)
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


def estimated_vs_annotated_ncp(results_dir, annotate=False, save_dir=None):
    """
    Plot the number of estimated vs. annotated change points for both estimation methods.

    :param results_dir: Directory where the results for the estimation of the number of change points in each cruise are
                       stored.
    :param annotate: Whether to annotate the points with the names of the cruises.
    :param save_dir: Directory where the figures will be saved.
    :return: est_cp_results: Dataframe containing the results from the estimation of the number of change points.
    """
    est_cp_results = pickle.load(open(os.path.join(results_dir, 'estimated_ncp.pickle'), 'rb'))
    max_cp = max(130, max(np.max(est_cp_results['n_cp']), np.max(est_cp_results['n_est_cps_penalty']),
                             np.max(est_cp_results['n_est_cps_rule_thumb'])) + 2)
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
    colors = ['#3eddf2', '#e542f4', '#00e68a']
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
    marker = 's'
    all_cps = []
    for i, cps_file in enumerate(cps_files):
        cps = json.load(open(cps_file, 'r'))['cps_bio'][str(ncp)]
        ymin, ymax = ax.get_ylim()
        ax.scatter([dists[int(cp)] for cp in cps], [ymin + (i + 1) * (ymax - ymin) / (len(cps_files) + 2)] * len(cps),
                   label=cps_labels[i], s=40, color=colors[i], marker=marker)
        all_cps.append(cps)

    # Edit the axis limits and labels, along with the legend and save the figure
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
    if 'Gaussian RBF' not in labels:
        plt.gcf().subplots_adjust(bottom=0.15, right=0.8)
    else:
        plt.gcf().subplots_adjust(bottom=0.15, right=0.68)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

    return all_cps, dists


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


def cruise_path_plot(lat, lon, extent=None, addl_lat=None, addl_lon=None, colors='rainbow', figure_size=None,
                save_path=None):
    """
    Plot the path of one cruise.

    :param lat: Latitudes.
    :param lon: Longitudes.
    :param extent: Vector of (min longitude, max longitude, min latitude, max latitude) for the plot.
    :param addl_lat: Latitudes of additional points to mark with an x on the plot.
    :param addl_lon: Longitudes of additional points to mark with an x on the plot.
    :param colors: Colors of the points. Current options are 'rainbow' (the points at the beginning of the cruise are
                   red and those at the end are violet) and 'blue'.
    :param figure_size: Size to make the figure.
    :param save_path: Where to save the plot. If None, the plot is displayed on the screen.
    """
    plt.clf()
    if figure_size is not None:
        fig = plt.figure(figsize=figure_size)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    if colors == 'rainbow':
        cm_subsection = np.linspace(0, 1, len(lon))
        colors = np.array([cm.gist_rainbow(x) for x in cm_subsection])
        plt.scatter(lon, lat, color=colors, transform=ccrs.Geodetic(), s=10, alpha=0.01)
    elif colors == 'blue':
        plt.scatter(lon, lat, color='blue', transform=ccrs.Geodetic(), s=10)
    else:
        raise NotImplementedError

    if addl_lat is not None and addl_lon is not None:
        plt.scatter(addl_lon, addl_lat, color='black', transform=ccrs.Geodetic(), s=50, marker='x')

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    if save_path is not None:
        plt.savefig(save_path + '.svg', format='svg')
    else:
        plt.show()
    plt.close()
