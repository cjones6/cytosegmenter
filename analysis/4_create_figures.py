import glob
import gsw
import numpy as np
import os
import sys
import time

sys.path.append('..')
from figures import figures


def figure1_cruises_map(background_day, background_file, cruises, data_dir, plot_dir=None):
    """
    Plot (1) a map with the cruises overlaid; and (2) a zoomed version with just the Northeast Pacific cruises.

    :param background_day: Numerical day of the year to use for the temperature data in the background.
    :param background_file: File with the background temperatures. You can download it here:
                            https://psl.noaa.gov/repository/entry/show?max=12&entryid=synth%3Ae570c8f9-ec09-4e89-93b4-babd5651e7a9%3AL25vYWEub2lzc3QudjIuaGlnaHJlcw%3D%3D&ascending=true&orderby=name&showentryselectform=true
    :param cruises: List of cruises to plot on the map.
    :param data_dir: Directory where the physical data is stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('Creating Figure 1')
    figures.cruises_map(background_day, background_file, cruises, data_dir, zoom=False, save_dir=plot_dir)
    figures.cruises_map(background_day, background_file, cruises, data_dir, zoom=True, save_dir=plot_dir)


def figure2_stacked_bar(cruise, data_dir, ncp, results_dir, plot_dir=None):
    """
    Create a stacked bar chart of the distribution of phytoplankton at each time point, with the estimated biological
    change points overlaid.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncp: Number of change points to use in the plot.
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 2')
    figures.stacked_bar_chart(cruise, data_dir, ncp, results_dir, arrow_cps_idxs=[0, 2], save_dir=plot_dir)


def figure3_cytograms(cruise, data_dir, plot_dir=None):
    """
    Make six cytograms, two around one change point and four around another. The first four will have log of chlorophyll
    vs. log of light scatter and the last two will have log of phycoerythrin vs. log of light scatter.

    :param cruise: Cruise to use when making the cytograms.
    :param data_dir: Directory where the physical data is stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 3')
    figures.cytogram(cruise, data_dir, [1952, 1964], r'log(Light scatter)', r'log(Chlorophyll)', 'fsc_small',
                     'chl_small', save_dir=plot_dir)
    figures.cytogram(cruise, data_dir, [2560, 2570], r'log(Light scatter)', r'log(Chlorophyll)', 'fsc_small',
                     'chl_small', save_dir=plot_dir)
    figures.cytogram(cruise, data_dir, [2560, 2570], r'log(Light scatter)', r'log(Phycoerythrin)', 'fsc_small',
                     'pe', save_dir=plot_dir)


def figure4_lat_lon(cruise, data_dir, ncp, results_dir, plot_dir=None):
    """
    Create a plot of the latitude and longitude throughout the cruise, with the estimated biological change points
    overlaid.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncp: Number of change points to use in the plot.
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 4')
    cps_bio, dists, lats, lons = figures.lat_lon_plot(cruise, data_dir, ncp, results_dir, save_dir=plot_dir)

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


def figure5_temp_salinity(cruise, data_dir, ncp, results_dir, plot_dir=None):
    """
    Create a plot of the temperature and salinity of the ocean throughout the cruise, with the estimated biological
    and physical change points overlaid when the number of change points is set to ncp for both the biological and
    physical data.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param ncp: Number of change points to use in the plot.
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 5')
    if plot_dir is not None:
        save_path = os.path.join(plot_dir, cruise + '_temp_salinity_est_vs_est10')
    else:
        save_path = None
    _, cps_phys, dists = figures.temp_salinity_vs_distance_plot(cruise, data_dir, ncp, ncp, results_dir,
                                                                axis_font_size=26, bbox=(1.05, -0.15),
                                                                figure_size=(10, 6), legend_font_size=18,
                                                                save_path=save_path)
    print('---------------------------')
    print('Physical change points in terms of distance:')
    print([dists[int(cp)] for cp in cps_phys])


def figure6a_est_vs_true_ncp(results_dir, plot_dir=None):
    """
    Create plots of the estimated vs. annotated number of change points in the physical data with the penalty method
    and the rule-of-thumb method.

    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 6a')
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


def figure6b_normalized_distances_hist(cruises, data_dir, results_dir, plot_dir=None):
    """
    Compute a histogram of the distance from each estimated change point to the nearest annotated physical change point,
    normalized by the average distance between annotated change points within each cruise.

    :param cruises: List of the names of cruises to use to make the histograms.
    :param data_dir: Directory where the physical data is stored.
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 6b')
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


def figure7_temp_salinity(cruise, data_dir, results_dir, plot_dir=None):
    """
    Create a plot of the temperature and salinity of the ocean throughout the cruise, with the estimated biological
    change points and annotated physical change points overlaid when the number of biological change points is estimated
    using the penalty method.

    :param cruise: Name of the cruise.
    :param data_dir: Directory where the physical data is stored.
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Figure 7')
    if plot_dir is not None:
        save_path = os.path.join(plot_dir, cruise + '_temp_salinity_est_vs_annotated')
    else:
        save_path = None
    cps_bio, cps_phys, dists = figures.temp_salinity_vs_distance_plot(cruise, data_dir, None, None, results_dir,
                                                                      axis_font_size=32, bbox=(0.91, -0.22),
                                                                      figure_size=(20, 5), legend_font_size=28,
                                                                      tick_labelsize=18, true_phys=True,
                                                                      save_path=save_path)
    print('---------------------------')
    print('Number of estimated biological change points:', len(cps_bio))


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


def supp_figure2_temp_salinity_robustness(cruise, ncp, results_dir, plot_dir=None):
    """
    Create plots of the temperature and salinity of the ocean throughout the cruise, with the estimated biological
    and physical change points overlaid when the number of change points is set to 10 and the parameter settings are
    varied.

    :param cruise: Name of the cruise.
    :param ncp: Number of change points to use in the plots.
    :param results_dir: Directory where the estimated change points are stored.
    :param plot_dir: Directory where the figure will be saved.
    """
    print('\nCreating Supplementary Figure 2')
    baseline_kernel = 'Gaussian-Euclidean'
    baseline_bw = '_'.join(glob.glob1(os.path.join(results_dir, cruise), 'cps_bio*rule-of-thumb*')[0].split('_')[-3:-1])
    baseline_min_dist = 5
    baseline_nclusters = 128

    kernel_type_labels = ['Linear', 'Gaussian RBF']
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


if __name__ == '__main__':
    t1 = time.time()
    # Directory where the cleaned data is stored
    data_dir = '../data/'
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
    figure2_stacked_bar('SCOPE_16', data_dir, 10, results_dir, plot_dir)
    figure3_cytograms('SCOPE_16', data_dir, plot_dir)
    figure4_lat_lon('SCOPE_16', data_dir, 10, results_dir, plot_dir)
    figure5_temp_salinity('SCOPE_16', data_dir, 10, results_dir, plot_dir)
    figure6a_est_vs_true_ncp(results_dir, plot_dir)
    figure6b_normalized_distances_hist(cruises, data_dir, results_dir, plot_dir)
    figure7_temp_salinity('SCOPE_16', data_dir, results_dir, plot_dir)
    supp_figure1a_water_mixing('SCOPE_16', [24], data_dir, plot_dir, subset=(2090, 2150))
    supp_figure1b_water_mixing('SCOPE_16', [24], data_dir, plot_dir, subset=(2090, 2150))
    supp_figure2_temp_salinity_robustness('SCOPE_16', 10, os.path.join(results_dir, 'sensitivity_analysis'), plot_dir)
    t2 = time.time()
    print('Runtime:', t2-t1)
    # Runtime (Intel i9-7960X CPU @ 2.80GHz): 44s
