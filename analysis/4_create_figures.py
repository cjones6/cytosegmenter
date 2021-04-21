import glob
import gsw
import json
import numpy as np
import os
import pandas as pd
import sys
import time

sys.path.append('..')
from figures import figures
from utils import utils


def figure1_cruises_map(background_day, background_file, cruises, data_dir, plot_dir=None):
    """
    Plot a map with the cruises overlaid.

    :param background_day: Numerical day of the year to use for the temperature data in the background.
    :param background_file: File with the background temperatures. You can download it here:
                            https://psl.noaa.gov/repository/entry/show?max=12&entryid=synth%3Ae570c8f9-ec09-4e89-93b4-babd5651e7a9%3AL25vYWEub2lzc3QudjIuaGlnaHJlcw%3D%3D&ascending=true&orderby=name&showentryselectform=true
    :param cruises: List of cruises to plot on the map.
    :param data_dir: Directory where the physical data is stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('Creating Figure 1')
    figures.cruises_map(background_day, background_file, cruises, data_dir, zoom=False, save_dir=plot_dir)


def table1(cruises, data_dir):
    """
    Generate the columns in Table 1 (except for the location column).

    :param cruises: List of cruises in the table.
    :param data_dir: Directory where the physical and biological data is stored.
    """
    print('\nCreating Table 1')
    official_name_dict = {'DeepDOM': 'KN210-04', 'KM1712': 'KM1712', 'KM1713': 'KM1713', 'MGL1704': 'MGL1704',
                          'SCOPE_2': 'KM1502', 'SCOPE_16': 'KOK1606', 'Thompson_1': 'TN248', 'Thompson_9': 'TN271',
                          'Thompson_12': 'TN292', 'Tokyo_1': 'Tokyo_1', 'Tokyo_2': 'Tokyo_2', 'Tokyo_3': 'Tokyo_3'}
    official_names = [official_name_dict[cruise] for cruise in cruises]
    sorted_idxs = sorted(range(len(official_names)), key=lambda k: official_names[k])

    print('Cruise \t Length (km) \t # point clouds \t # particles \t # changes')
    for idx in sorted_idxs:
        cruise, official_name = cruises[idx], official_names[idx]
        bio_data = pd.read_parquet(os.path.join(data_dir, cruise + '_bio.parquet'))
        phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
        dists = utils.compute_distances(phys_data['latitude'], phys_data['longitude'])
        n_cp = len(json.load(open(os.path.join(data_dir, cruise + '_annotated_phys_cps.json'), 'r')))
        print('%8s' % official_name_dict[cruise], '\t %5.0f' % dists[-1], '\t %5.0f' % len(phys_data),
              '\t %8.0f' % len(bio_data), '\t', n_cp)


def figure3_stacked_bar(cruise, data_dir, ncp, results_dir, plot_dir=None):
    """
    Create a stacked bar chart of the distribution of phytoplankton at each time point, with the estimated biological
    change points overlaid.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncp: Number of change points to use in the plot.
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 3')
    cum_dists_cps = figures.stacked_bar_chart(cruise, data_dir, ncp, results_dir, arrow_cps_idxs=[0, 1],
                                              save_dir=plot_dir)
    print('Estimated change points in Figure 2:', cum_dists_cps)


def figure4_supp_figures34_cytograms_histograms(cruise, data_dir, idx_sets, plot_dir=None):
    """
    Make cytograms (four plots with two cytograms each around each of two change points per plot and either chlorophyll
    or phycoerythrin on the y-axis) and univariate histograms at each of the input time points (one histogram per pair
    of change points).

    :param cruise: Cruise to use when making the cytograms and histograms.
    :param data_dir: Directory where the physical data is stored.
    :param idx_sets: Indices corresponding to the locations to be used in the plots. These should be a list of lists,
                     where each sub-list contains four elements.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 4 and Supplementary Figures 3 and 4')
    for idx_set in idx_sets:
        figures.cytograms(cruise, data_dir, idx_set, r'Light scatter (unitless)',
                          r'Chlorophyll (unitless)', 'fsc_small', 'chl_small', save_dir=plot_dir)
        figures.cytograms(cruise, data_dir, idx_set, r'Light scatter (unitless)',
                          r'Phycoerythrin (unitless)', 'fsc_small', 'pe', save_dir=plot_dir)
        figures.histograms(cruise, data_dir, idx_set, r'Light scatter (unitless)', 'fsc_small',
                           legend=True, save_dir=plot_dir)
        figures.histograms(cruise, data_dir, idx_set, r'Chlorophyll (unitless)', 'chl_small',
                           legend=False, save_dir=plot_dir)
        figures.histograms(cruise, data_dir, idx_set, r'Phycoerythrin (unitless)', 'pe', legend=False,
                           save_dir=plot_dir)


def figure5_lat_lon_map(cruise, data_dir, extent, ncp, results_dir, plot_dir=None):
    """
    Create a map with the ship's location throughout the cruise, with the estimated biological change points overlaid.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param extent: List with the longitudes and latitudes defining the extent of the map.
    :param ncp: Number of change points to use in the plot.
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 5')
    cps_file = glob.glob1(os.path.join(results_dir, cruise), 'cps_bio*rule-of-thumb_*')
    if len(cps_file) > 1:
        print('WARNING: More than 1 results file for cruise', cruise)
    cps_bio = json.load(open(os.path.join(results_dir, cruise, cps_file[0]), 'r'))['cps_bio'][str(ncp)]
    phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
    lats = phys_data['latitude']
    lons = phys_data['longitude']
    addl_lat = np.array(lats)[cps_bio]
    addl_lon = np.array(lons)[cps_bio]

    figures.cruise_path_plot(lats, lons, extent=extent, addl_lat=addl_lat[:5], addl_lon=addl_lon[:5],
                             addl_lat2=addl_lat[5:], addl_lon2=addl_lon[5:], colors='gray-blue',
                             figure_size=(9.6, 7.2), save_path=os.path.join(plot_dir, cruise + '_lat_lon_map'))

    dists = utils.compute_distances(lats, lons)
    lats_cps = [lats[cps_bio[i]] for i in range(len(cps_bio))]
    lons_cps = [lons[cps_bio[i]] for i in range(len(cps_bio))]
    dists_bio = np.inf*np.ones((len(lats_cps), len(lats_cps)))
    for i, locs in enumerate(zip(lats_cps, lons_cps)):
        for j, locs2 in enumerate(zip(lats_cps, lons_cps)):
            if i != j:
                dists_bio[i, j] = dists_bio[j, i] = gsw.distance([locs[1], locs2[1]], [locs[0], locs2[0]], 0)/1000.
    min_dists = np.min(dists_bio, axis=1)
    argmin_dists = np.argmin(dists_bio, axis=1)

    print('---------------------------')
    print('Distance traveled before reaching each change point:', [dists[cps_bio[i]] for i in range(len(cps_bio))])
    print('Distance of each change biological change point to the closest other biological change point:', min_dists)
    print('Indices corresponding to the closest change points:', argmin_dists)


def figure6_temp_salinity(cruise, data_dir, ncps, results_dir, plot_dir=None):
    """
    Create a plot with 2 horizontal panels and two vertical panels of the temperature and salinity of the ocean
    throughout the cruise, with one number of estimated and annotated physical change points overlaid on the left and
    another number on the right.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncps: Number of change points to use for the biological and physical data on each side of the plot. If None,
                 it uses the estimated number (for the biological data) or the annotated number (for the physical data).
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 6')
    if plot_dir is not None:
        save_path = os.path.join(plot_dir, cruise + '_temp_salinity_bio_vs_phys')
    else:
        save_path = None

    cps_bio, cps_phys, dists = figures.temp_salinity_vs_latitude_plot(cruise, data_dir, ncps, ncps, results_dir,
                                                                      true_phys=[False, True], axis_font_size=30,
                                                                      bbox=(0.5, -0.3), figure_size=(20, 10),
                                                                      legend_font_size=30, tick_labelsize=18,
                                                                      save_path=save_path)
    print('---------------------------')
    print('Number of estimated biological change points:', len(cps_bio))
    print('10 biological change points in terms of distance:')
    print([dists[int(cp)] for cp in cps_bio])
    print('10 physical change points in terms of distance:')
    print([dists[int(cp)] for cp in cps_phys])


def figure7a_supp_figure10_est_vs_true_ncp(results_dir, plot_dir=None):
    """
    Create plots of the estimated vs. annotated number of change points in the physical data with the penalty method
    and the rule-of-thumb method.

    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 7a and Supplementary Figure 10')
    est_cp_results = figures.estimated_vs_annotated_ncp(results_dir, save_dir=plot_dir)
    print('---------------------------')
    for key, name in zip(['n_est_cps_penalty', 'n_est_cps_rule_thumb'], ['penalty', 'rule-of-thumb']):
        print('Results for', name, 'approach:')
        print('Best parameter values:')
        if name == 'penalty':
            print(set(est_cp_results['alpha']))
        else:
            print(set(est_cp_results['nu']))
        print('Number of cruises for which the estimated number of change points is within a factor of 2 of the number '
              'of annotated change points:',
              np.sum((0.5 * est_cp_results[key].to_numpy() < est_cp_results['n_cp'].to_numpy())
                     & (est_cp_results['n_cp'].to_numpy() < 2 * est_cp_results[key].to_numpy())),
              'of', len(est_cp_results[key]))
        print('Correlation coefficient:',
              np.corrcoef(est_cp_results[key].to_numpy(), est_cp_results['n_cp'].to_numpy())[0, 1])
        print('Estimated number of change points in SCOPE 16:',
              est_cp_results.loc[est_cp_results['cruise'] == 'SCOPE_16'][key].item())


def figure7b_normalized_distances_hist(cruises, data_dir, results_dir, plot_dir=None):
    """
    Compute a histogram of the distance from each estimated change point to the nearest annotated physical change point,
    normalized by the average distance between annotated change points within each cruise.

    :param cruises: List of the names of cruises to use to make the histograms.
    :param data_dir: Directory where the physical data is stored.
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 7b')
    all_dists_est_to_true, all_dists_unif_to_true, avg_dists_true = figures.normalized_distance_histogram(cruises,
                                                                                                data_dir, results_dir,
                                                                                                save_dir=plot_dir)
    print('---------------------------')
    print('Total number of estimated change points:', len(np.concatenate(all_dists_est_to_true)))
    print('Fraction of estimated change points within x% of the average distance between true ones to true ones:')
    print('1%:', np.mean(np.concatenate(all_dists_est_to_true)/np.concatenate(avg_dists_true) < 0.01))
    print('5%:', np.mean(np.concatenate(all_dists_est_to_true)/np.concatenate(avg_dists_true) < 0.05))
    print('10%:', np.mean(np.concatenate(all_dists_est_to_true)/np.concatenate(avg_dists_true) < 0.1))
    print('25%:', np.mean(np.concatenate(all_dists_est_to_true)/np.concatenate(avg_dists_true) < 0.25))

    print('Fraction of uniform change points within x% of the average distance between true ones to true ones:')
    print('1%:', np.mean(np.concatenate(all_dists_unif_to_true)/np.concatenate(avg_dists_true) < 0.01))
    print('5%:', np.mean(np.concatenate(all_dists_unif_to_true)/np.concatenate(avg_dists_true) < 0.05))
    print('10%:', np.mean(np.concatenate(all_dists_unif_to_true)/np.concatenate(avg_dists_true) < 0.1))
    print('25%:', np.mean(np.concatenate(all_dists_unif_to_true)/np.concatenate(avg_dists_true) < 0.25))


def supp_figure1a_water_mixing(cruise, cps, data_dir, plot_dir=None, subset=(0, None)):
    """
    Create a plot of the temperature and salinity of the ocean vs. distance traveled, with the specified change points
    overlaid.

    :param cruise: Name of the cruise.
    :param cps: Change points to annotate on the plot (by index).
    :param data_dir: Directory where the physical data is stored.
    :param plot_dir: Directory where the figure will be saved.
    :param subset: Subset of the cruise to use (by index). It should be a tuple and will include the starting point but
                   not the endpoint.
    """
    print('\nCreating Supplementary Figure 1a')
    if plot_dir is not None:
        save_path = os.path.join(plot_dir, cruise + '_temp_salinity_vs_dist_mixing_event')
    else:
        save_path = None
    cps_bio, cps_phys, dists = figures.temp_salinity_vs_distance_plot(cruise, data_dir, 0, 0, results_dir,
                                                                      axis_font_size=20, cps=cps,
                                                                      legend_font_size=18, subset=subset,
                                                                      tick_labelsize=14, true_phys=False,
                                                                      save_path=save_path)
    print('---------------------------')
    print('Input change points in terms of indices:', [cp for cp in cps])
    print('Input change points in terms of distance:', [dists[cp] for cp in cps])


def supp_figure1b_water_mixing(cruise, cps, data_dir, plot_dir=None, subset=(0, None)):
    """
    Create a plot of the temperature vs. salinity of the ocean in a specific subset of a cruise, with the input change
    points, cps, marked in black x's.

    :param cruise: Name of the cruise.
    :param cps: Change points to annotate on the plot (by index).
    :param data_dir: Directory where the physical data is stored.
    :param plot_dir: Directory where the figure will be saved.
    :param subset: Subset of the cruise to use (by index). It should be a tuple and will include the starting point but
                   not the endpoint.
    """
    print('\nCreating Supplementary Figure 1b')
    if plot_dir is not None:
        save_path = os.path.join(plot_dir, cruise + '_temp_vs_salinity_mixing_event')
    else:
        save_path = None
    figures.temp_vs_salinity_plot(cruise, data_dir, cps=cps, subset=subset, save_path=save_path)


def supp_figures567_mean_element(cruise, data_dir, features_dir, idxs, pctiles, vmaxes, plot_dir=None):
    """
    Create three different mean element figures for a given cruise. The first two figures consist of a plot of the mean
    element of a point cloud at four different indices, along with the difference in the first two and latter two mean
    elements.  The last figure plots the mean element of a single point cloud with the centroids from k-means that were
    used in the Nyström approximation overlaid.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the biological data is stored.
    :param features_dir: Directory where the features are stored.
    :param idxs: Indices of the point clouds to be used in the plots. This should be a list of lists, where the first
                 two lists are of length four and the last is of length 1.
    :param pctiles: List of percentiles of the third variable (pe) to be used in the plots. This should be a list of
                    lists, where the first two lists are of length four and the last is of length 1.
    :param vmaxes: List of maximum color values to use in the plots. This should be a list of lists, where the first two
                   lists are of length four and the last is of length 1.
    :param plot_dir: Directory where the figures will be saved.
    """
    print('\nCreating Supplementary Figures 5, 6 and 7')
    figures.mean_element_plots(cruise, data_dir, features_dir, [*idxs[0], None, None],
                              r'Standardized light scatter (unitless)', r'Standardized chlorophyll (unitless)',
                              'fsc_small', 'chl_small', grid_size=(100, 100), percentiles=pctiles[0],
                              plot_centroids=False, projection_dim=128, vmaxes=[*vmaxes[0], None, None],
                              save_dir=plot_dir)
    figures.mean_element_plots(cruise, data_dir, features_dir, [*idxs[1], None, None],
                              r'Standardized light scatter (unitless)', r'Standardized chlorophyll (unitless)',
                              'fsc_small', 'chl_small', grid_size=(100, 100), percentiles=pctiles[1],
                              plot_centroids=False, projection_dim=128, vmaxes=[*vmaxes[1], None, None],
                              save_dir=plot_dir)
    figures.mean_element_plots(cruise, data_dir, features_dir, idxs[2], r'Standardized light scatter (unitless)',
                              r'Standardized chlorophyll (unitless)', 'fsc_small', 'chl_small', grid_size=(100, 100),
                              percentiles=pctiles[2], plot_centroids=True, projection_dim=128, vmaxes=vmaxes[2],
                              save_dir=plot_dir)


def supp_figure8_temp_salinity_robustness(cruise, ncp, results_dir, plot_dir=None):
    """
    Create plots of the temperature and salinity of the ocean throughout the cruise, with the estimated biological
    and physical change points overlaid when the number of change points is set to 10 and the parameter settings are
    varied.

    :param cruise: Name of the cruise.
    :param ncp: Number of change points to use in the plots.
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Supplementary Figure 8')
    baseline_kernel = 'Gaussian-Euclidean'
    baseline_bw = '_'.join(glob.glob1(os.path.join(results_dir, cruise), 'cps_bio*rule-of-thumb*')[0].split('_')[-3:-1])
    baseline_min_dist = 5
    baseline_nclusters = 128

    kernel_type_labels = ['Linear', 'RBF']
    min_dists = [1] + list(range(5, 55, 5))
    nsclusters = [2 ** i for i in range(2, 11)]

    # Robustness to the choice of bandwidth
    cps_files = [os.path.join(results_dir, cruise, 'cps_bio_' + str(baseline_nclusters) + '_' + baseline_kernel + '_'
                                 + baseline_bw + '_' + str(baseline_min_dist)) + '.json'] \
                + [os.path.join(results_dir, cruise, cps_file) for cps_file in
                    glob.glob1(os.path.join(results_dir, cruise), 'cps_bio_*Gaussian-Euclidean_0*.json')]
    sorted_idxs = np.argsort([float(cps_file.split('_')[-2]) for cps_file in cps_files])
    cps_files = [cps_files[idx] for idx in sorted_idxs]
    if plot_dir is not None:
        save_path = os.path.join(plot_dir, cruise + '_sensitivity_bandwidth.svg')
    else:
        save_path = None
    all_cps, dists = figures.stacked_bar_chart_robustness(cps_files, [cps_file.split('_')[-2] for cps_file in
                                                                      cps_files],
                                                          cruise, data_dir, ncp, reverse_legend=True,
                                                          save_path=save_path)
    print('---------------------------')
    print('Rule of thumb bandwidth:', baseline_bw.split('_')[-1])
    print('Estimated change points when varying the bandwidth:')
    print('Bandwidth \t Change points')
    for cps_file, cps in zip(cps_files, all_cps):
        print(float(cps_file.split('_')[-2]), '\t', np.array([np.round(dists[cp]) for cp in cps], dtype=int))

    # Robustness to the choice of kernel
    cps_files = [os.path.join(results_dir, cruise, 'cps_bio_' + str(baseline_nclusters) + '_' + 'Linear' + '_'
                                + 'rule-of-thumb_0' + '_' + str(baseline_min_dist)) + '.json'] \
                + [os.path.join(results_dir, cruise, 'cps_bio_' + str(baseline_nclusters) + '_' + baseline_kernel + '_'
                                 + baseline_bw + '_' + str(baseline_min_dist)) + '.json']
    if plot_dir is not None:
        save_path = os.path.join(plot_dir, cruise + '_sensitivity_kernel.svg')
    else:
        save_path = None
    all_cps, _ = figures.stacked_bar_chart_robustness(cps_files, kernel_type_labels, cruise, data_dir, ncp,
                                                   reverse_legend=True, save_path=save_path)
    print('Estimated change points when varying the kernel:')
    print('Kernel \t Change points')
    for kernel_type, cps in zip(kernel_type_labels, all_cps):
        print(kernel_type, '\t', np.array([np.round(dists[cp]) for cp in cps], dtype=int))

    # Robustness to the choice of the dimension of the embeddings
    cps_files = [os.path.join(results_dir, cruise,
                    glob.glob1(os.path.join(results_dir, cruise), 'cps_bio_' + str(nclusters) + '_*.json')[0])
                 for nclusters in nsclusters]
    if plot_dir is not None:
        save_path = os.path.join(plot_dir, cruise + '_sensitivity_embedding_dim.svg')
    else:
        save_path = None
    all_cps, _ = figures.stacked_bar_chart_robustness(cps_files, [str(nclusters) for nclusters in nsclusters], cruise,
                                                   data_dir, ncp, reverse_legend=True, save_path=save_path)
    print('Estimated change points when varying the embedding dimension:')
    print('Dimension \t Change points')
    for nclusters, cps in zip(nsclusters, all_cps):
        print(nclusters, '\t', np.array([np.round(dists[cp]) for cp in cps], dtype=int))

    # Robustness to the choice of the minimum distance between change points
    cps_files = [os.path.join(results_dir, cruise, 'cps_bio_' + str(baseline_nclusters) + '_' + baseline_kernel + '_'
                                 + baseline_bw + '_' + str(min_dist)) + '.json' for min_dist in min_dists]
    if plot_dir is not None:
        save_path = os.path.join(plot_dir, cruise + '_sensitivity_min_dist.svg')
    else:
        save_path = None
    all_cps, _ = figures.stacked_bar_chart_robustness(cps_files, [str(min_dist) for min_dist in min_dists], cruise,
                                                   data_dir, ncp, reverse_legend=True, save_path=save_path)
    print('Estimated change points when varying the minimum distance between change points:')
    print('Dimension \t Change points')
    for min_dist, cps in zip(min_dists, all_cps):
        print(min_dist, '\t', np.array([np.round(dists[cp]) for cp in cps], dtype=int))


def supp_figure9_variance(cruise, data_dir, ncp, results_dir, subsample_of=10, plot_dir=None):
    """
    Plot a histogram of the distances (in terms of indices divided by half of subsample_of) from each estimated
    change point in a set of subsampled cruises to the nearest change point estimated in the non-subsampled cruise.

    :param cruises: List of the names of cruises to use to make the histograms.
    :param data_dir: Directory where the physical data is stored.
    :param results_dir: Directory where the estimated change points are stored.
    :param subsample_of: Number of subsamples previously generated.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Supplementary Figure 9')
    all_cps, hist = figures.variance_histogram(cruise, data_dir, ncp, os.path.join(results_dir, '..'), results_dir,
                                         subsample_of=subsample_of, save_dir=plot_dir)

    print('Fraction of change points from subsamples in each histogram bin:', hist[0]/sum(hist[0]))
    print('Estimated change points when varying the subsample:')
    print('Subsample \t Change points')
    for i in range(len(all_cps)):
        print(i, '\t', np.array([np.round(cp) for cp in all_cps[i]], dtype=int))


def supp_figure11_histograms(cruise, data_dir, plot_dir=None):
    """
    Make univariate histograms at each of the input time points (one histogram per pair of time points).

    :param cruise: Cruise to use when making the histograms.
    :param data_dir: Directory where the physical data is stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Supplementary Figure 11')
    cps_file = glob.glob1(os.path.join(results_dir, cruise), 'cps_bio*rule-of-thumb_*')
    if len(cps_file) > 1:
        print('WARNING: More than 1 results file for cruise', cruise)
    cps_bio = json.load(open(os.path.join(results_dir, cruise, cps_file[0]), 'r'))['cps_bio'][str(10)]
    idx_set = [cps_bio[1] - 4, cps_bio[1] + 5]
    figures.histograms(cruise, data_dir, idx_set, r'Light scatter (unitless)', 'fsc_small', legend=True,
                       save_dir=plot_dir)
    figures.histograms(cruise, data_dir, idx_set, r'Chlorophyll (unitless)', 'chl_small', legend=False,
                       save_dir=plot_dir)
    figures.histograms(cruise, data_dir, idx_set, r'Phycoerythrin (unitless)', 'pe', legend=False, save_dir=plot_dir)


if __name__ == '__main__':
    t1 = time.time()
    # Directory where the cleaned data is stored
    data_dir = '../data/'
    # Directory where the features are stored
    features_dir = '../features/'
    # Directory where the change-point results are stored
    results_dir = '../results/'
    # Directory where the plots will be stored
    plot_dir = '../plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    # Background file and day of the year for the cruise map
    background_day = 116  # 117th day = April 26 = middle of scope 16 cruise
    background_file = r'../data/sst.day.mean.2016.v2.nc'
    # List of cruises to use
    cruises = ['DeepDOM', 'KM1712', 'KM1713', 'MGL1704', 'SCOPE_2', 'SCOPE_16', 'Thompson_1', 'Thompson_9',
               'Thompson_12', 'Tokyo_1', 'Tokyo_2', 'Tokyo_3']

    figure1_cruises_map(background_day, background_file, cruises, data_dir, plot_dir)
    table1(cruises, data_dir)
    figure3_stacked_bar('SCOPE_16', data_dir, 10, results_dir, plot_dir)
    figure4_supp_figures34_cytograms_histograms('SCOPE_16', data_dir, [[861, 874, 1964, 1976], [930, 942, 5780, 5791]],
                                                plot_dir)
    figure5_lat_lon_map('SCOPE_16', data_dir, [-160, -153, 15, 40], 10, results_dir, plot_dir)
    figure6_temp_salinity('SCOPE_16', data_dir, [10, None], results_dir, plot_dir)
    figure7a_supp_figure10_est_vs_true_ncp(results_dir, plot_dir)
    figure7b_normalized_distances_hist(cruises, data_dir, results_dir, plot_dir)

    supp_figure1a_water_mixing('SCOPE_16', [24], data_dir, plot_dir, subset=(2100, 2160))
    supp_figure1b_water_mixing('SCOPE_16', [24], data_dir, plot_dir, subset=(2100, 2160))
    supp_figures567_mean_element('SCOPE_16', data_dir, features_dir, [[861, 874, 1964, 1976], [930, 942, 5780, 5791],
                                                                      [1964]],
                                 [[95, 95, 50, 50, 95, 50], [95, 95, 50, 50, 95, 50], [50]],
                                 [[0.04, 0.04, 0.7, 0.7], [0.04, 0.04, 0.7, 0.7], [0.7]], plot_dir=plot_dir)
    supp_figure8_temp_salinity_robustness('SCOPE_16', 10, os.path.join(results_dir, 'sensitivity_analysis'), plot_dir)
    supp_figure9_variance('SCOPE_16', data_dir, 10, os.path.join(results_dir, 'variance_analysis'), subsample_of=10,
                          plot_dir=plot_dir)
    supp_figure11_histograms('KM1712', data_dir, plot_dir)

    t2 = time.time()
    print('Runtime:', t2-t1)
    # Runtime (Intel i9-7960X CPU @ 2.80GHz): 1m38s
