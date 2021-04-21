import json
import numpy as np
import os
import pickle
import sklearn.metrics
import time

from chapydette import cp_estimation


def load_features(cruise, features_dir, projection_dim, subsample_num=1, subsample_of=1):
    """
    Load features for a cruise.

    :param cruise: Cruise to load features for.
    :param features_dir: Directory where the features are stored.
    :param projection_dim: Dimension of the features.
    :param subsample_num: Subsample number to load.
    :param subsample_of: Number of subsamples previously generated for this cruise.
    :return: Tuple consisting of:

        * cruise_features['bio_features']: The features for the biological data
        * cruise_features['phys_features']: The features for the physical data
    """
    if subsample_of == 1:
        features_path = os.path.join(features_dir, cruise + '_features_' + str(projection_dim) + '.pickle')
    else:
        features_path = os.path.join(features_dir, cruise + '_features_' + str(projection_dim) + '_subsample_' +
                                                    str(subsample_num+1) + '_of_' + str(subsample_of) + '.pickle')
    cruise_features = pickle.load(open(features_path, 'rb'))

    return cruise_features['bio_features'], cruise_features['phys_features']


def get_bw_range(features):
    """
    Get the rule-of-thumb bandwidth and a range of bandwidths on a log scale for the Gaussian RBF kernel.

    :param features: Features to use to obtain the bandwidths.
    :return: Tuple consisting of:

        * rule_of_thumb_bw: Computed rule-of-thumb bandwidth.
        * bws: List of bandwidths on a log scale.
    """
    dists = sklearn.metrics.pairwise.pairwise_distances(features).reshape(-1)
    rule_of_thumb_bw = np.median(dists)
    gammas = np.logspace(np.log(0.5/np.percentile(dists, 99)**2), np.log(0.5/np.percentile(dists, 1)**2), 10, base=np.e)
    bws = np.sqrt(1/(2*gammas))

    return rule_of_thumb_bw, bws


def est_cps(cruise, bio_features, phys_features, max_ncp=150, min_dists=[5], kernel_types=['Gaussian-Euclidean'],
            bw_method='rule-of-thumb', subsample_num=1, subsample_of=1, save_dir='../results/'):
    """
    Estimate the locations of change points in the input biological and physical features for a single cruise.

    :param cruise: Name of the cruise the features are from.
    :param bio_features: Features for the biological data.
    :param phys_features: Features for the physical data.
    :param max_ncp: Maximum number of change points in a sequence.
    :param min_dists: List of minimum acceptable distances between change points.
    :param kernel_types: List containing 'Gaussian-Euclidean' (Gaussian RBF kernel) and/or 'Linear'.
    :param bw_method: Method to use for obtaining the bandwidth(s). Either 'rule-of-thumb' or 'list'.
    :param subsample_num: Subsample number being used.
    :param subsample_of: Number of subsamples previously generated for this cruise.
    :param save_dir: Top-level directory where the results will be stored.
    """
    projection_dim = bio_features.shape[1]
    for min_dist in min_dists:
        # Perform change-point estimation on the physical data
        if not os.path.exists(os.path.join(save_dir, cruise)):
            os.makedirs(os.path.join(save_dir, cruise))

        if phys_features is not None:
            cps_phys, objs_phys = cp_estimation.mkcpe(X=phys_features,
                                                      n_cp=(1, min(max_ncp, int((len(phys_features)-1)/min_dist)-1)),
                                                      kernel_type='linear', min_dist=min_dist, return_obj=True)
            for key in cps_phys.keys():
                cps_phys[key] = cps_phys[key].flatten().tolist()
            save_path = os.path.join(save_dir, cruise, 'cps_phys.json')
            json.dump({'cps_phys': cps_phys, 'objs_phys': objs_phys}, open(save_path, 'w'))

        for kernel_type in kernel_types:
            # Get the bandwidth(s) (if applicable)
            if kernel_type != 'Linear':
                rot_bw, bws = get_bw_range(bio_features)
                all_bws = [rot_bw] if bw_method == 'rule-of-thumb' else bws
            else:
                all_bws = [0]
            for bw in all_bws:
                # Perform change-point estimation on the biological data
                cps_bio, objs_bio = cp_estimation.mkcpe(X=bio_features,
                                                        n_cp=(1, min(max_ncp, int((len(bio_features)-1)/min_dist)-1)),
                                                        kernel_type=kernel_type, bw=bw, min_dist=min_dist,
                                                        return_obj=True)
                for key in cps_bio.keys():
                    cps_bio[key] = cps_bio[key].flatten().tolist()

                bw_short = 'rule-of-thumb_' + str(np.round(bw, 3)) if bw_method == 'rule-of-thumb' else \
                            str(np.round(bw, 3))
                if subsample_of == 1:
                    save_path = os.path.join(save_dir, cruise, 'cps_bio_' + str(projection_dim) + '_' +
                                             kernel_type + '_' + str(bw_short) + '_' + str(min_dist) + '.json')
                else:
                    save_path = os.path.join(save_dir, cruise, 'cps_bio_' + str(projection_dim) + '_' +
                                             kernel_type + '_' + str(bw_short) + '_' + str(min_dist) + '_subsample_' +
                                             str(subsample_num+1) + '_of_' + str(subsample_of) + '.json')
                json.dump({'cps_bio': cps_bio, 'bw': bw, 'objs_bio': objs_bio}, open(save_path, 'w'))


def est_cps_all_cruises(cruises, features_dir, max_ncp=150, min_dist=5, projection_dim=128, save_dir='../results'):
    """
    Estimate the biological and physical change points for each cruise and for all desired parameter settings in order
    to make the plots.

    :param cruises: List of cruises to estimate change points for.
    :param features_dir: Directory where the features are stored.
    :param max_ncp: Maximum number of acceptable change points.
    :param min_dist: Minimum acceptable distance between change points.
    :param projection_dim: Dimension of the features.
    :param save_dir: Location where the estimated change points will be stored.
    """
    for cruise_num, cruise in enumerate(cruises):
        print('Estimating change points for', cruise, '- Cruise ', cruise_num+1, '/', len(cruises))
        bio_features, phys_features = load_features(cruise, features_dir, projection_dim)
        est_cps(cruise, bio_features, phys_features, max_ncp=max_ncp, min_dists=[min_dist], save_dir=save_dir)


def sensitivity_analysis(cruise, features_dir, max_ncp=150, min_dist=5, projection_dim=128,
                         save_dir='../results/sensitivity_analysis/'):
    """
    Estimate the biological change points for a given cruise when varying the parameter settings.

    :param cruise: Name of the cruise to perform the sensitivity analysis on.
    :param features_dir: Directory where the features are stored.
    :param max_ncp: Maximum number of acceptable change points.
    :param min_dist: Baseline minimum acceptable distance between change points.
    :param projection_dim: Baseline dimension of the features.
    :param save_dir: Location where the estimated change points will be stored.
    """
    print('Performing sensitivity analysis')
    bio_features, phys_features = load_features(cruise, features_dir, projection_dim)
    kernel_types = ['Linear']
    est_cps(cruise, bio_features, phys_features, max_ncp=max_ncp, min_dists=[min_dist], kernel_types=kernel_types,
            bw_method='rule-of-thumb', save_dir=save_dir)
    bw_method = 'list'
    est_cps(cruise, bio_features, phys_features, max_ncp=max_ncp, min_dists=[min_dist],
            kernel_types=['Gaussian-Euclidean'], bw_method=bw_method, save_dir=save_dir)
    min_dists = [1] + list(range(5, 55, 5))
    est_cps(cruise, bio_features, phys_features, max_ncp=max_ncp, min_dists=min_dists,
            kernel_types=['Gaussian-Euclidean'], bw_method='rule-of-thumb', save_dir=save_dir)
    for projection_dim in [2 ** i for i in range(2, 11)]:
        bio_features, phys_features = load_features(cruise, features_dir, projection_dim)
        est_cps(cruise, bio_features, phys_features, max_ncp=max_ncp, min_dists=[min_dist],
                kernel_types=['Gaussian-Euclidean'], bw_method='rule-of-thumb', save_dir=save_dir)


def variance_analysis(cruise, features_dir, max_ncp=150, min_dist=5, projection_dim=128, subsample_of=10,
                      save_dir='../results'):
    """
    Estimate the biological and physical change points for the given cruise after subsampling the data.

    :param cruise: Name of the cruise to perform the variance analysis on.
    :param features_dir: Directory where the features are stored.
    :param max_ncp: Maximum number of acceptable change points.
    :param min_dist: Minimum acceptable distance between change points.
    :param projection_dim: Dimension of the features.
    :param subsample_of: Number of subsamples previously generated.
    :param save_dir: Location where the estimated change points will be stored.
    """
    for subsample_num in range(subsample_of):
        print('Estimating change points for', cruise, '- subsample ', subsample_num+1, '/', subsample_of)
        bio_features, phys_features = load_features(cruise, features_dir, projection_dim, subsample_num=subsample_num,
                                                    subsample_of=subsample_of)
        est_cps(cruise, bio_features, phys_features, max_ncp=max_ncp, min_dists=[min_dist], subsample_num=subsample_num,
                subsample_of=subsample_of, save_dir=save_dir)


def simple_avg_comparison(cruises, features_dir, kernel_types=['Linear'], max_ncp=150, min_dist=5,
                          save_dir='../results'):
    """
    Estimate the biological change points for the given cruise when using features derived from simply averaging the
    data within each point cloud.

    :param cruises: List of cruises to estimate change points for.
    :param features_dir: Directory where the features are stored.
    :param kernel_types: List containing 'Gaussian-Euclidean' (Gaussian RBF kernel) and/or 'Linear'.
    :param max_ncp: Maximum number of acceptable change points.
    :param min_dist: Minimum acceptable distance between change points.
    :param save_dir: Location where the estimated change points will be stored.
    """
    for cruise in cruises:
        features_dict = pickle.load(open(os.path.join(features_dir, cruise + '_features_simple_avg.pickle'), 'rb'))
        bio_features = features_dict['bio_features']
        est_cps(cruise, bio_features, None, kernel_types=kernel_types, max_ncp=max_ncp, min_dists=[min_dist],
                save_dir=save_dir)


if __name__ == '__main__':
    t1 = time.time()
    # Directory where the features are stored
    features_dir = '../features/'
    # Directory where the output from the main analysis will be stored
    save_dir = '../results/'
    # Directory where the output from the sensitivity analysis will be stored
    save_dir_sensitivity_analysis = '../results/sensitivity_analysis/'
    # Directory where the output from the analysis of subsampled data will be stored
    save_dir_variance_analysis = '../results/variance_analysis/'
    # Directory where the output from the comparison with simply averaging the data for each point in a point cloud will
    # be stored
    save_dir_average_comparison = '../results/average_comparison/'
    # List of cruises to use
    cruises = ['DeepDOM', 'KM1712', 'KM1713', 'MGL1704', 'SCOPE_2', 'SCOPE_16', 'Thompson_1', 'Thompson_9',
               'Thompson_12', 'Tokyo_1', 'Tokyo_2', 'Tokyo_3']

    est_cps_all_cruises(cruises, features_dir, save_dir=save_dir)
    sensitivity_analysis('SCOPE_16', features_dir, save_dir=save_dir_sensitivity_analysis)
    variance_analysis('SCOPE_16', features_dir, save_dir=save_dir_variance_analysis, subsample_of=10)
    simple_avg_comparison(cruises, features_dir, max_ncp=150, min_dist=5, save_dir=save_dir_average_comparison)

    t2 = time.time()
    print('Runtime:', t2-t1)
    # Runtime (Intel i9-7960X CPU @ 2.80GHz): 5m39s
