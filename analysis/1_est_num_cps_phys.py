import json
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time

from chapydette import cp_estimation


def est_cps_objs(phys_data, max_cp, min_dist=5):
    """
    Estimate the locations of 0-max_cp change points in the physical data and return the corresponding objective values.

    :param phys_data: Physical data on which to estimate change points.
    :param max_cp: Largest number of change points to estimate.
    :param min_dist: Minimum allowable distance between change points.
    :return: objs_phys: Objective values when setting the number of change points to each of 0, 1, 2,..., max_cp (or the
                        maximum possible number of changes given that the minimum distance between change points is
                        min_dist).
    """
    phys_features = np.asarray(phys_data[['temp', 'salinity']])
    phys_features = StandardScaler().fit_transform(phys_features)
    cps_phys, objs_phys = cp_estimation.mkcpe(X=phys_features,
                                              n_cp=(0, min(max_cp, int((len(phys_features) - 1) / min_dist) - 1)),
                                              kernel_type='linear', min_dist=min_dist, return_obj=True)
    for key in objs_phys:
        objs_phys[key] = objs_phys[key]/len(phys_features)

    return objs_phys


def est_ncp_penalty(objs, n, alpha):
    """
    Estimate the number of change points using the penalty \alpha d/n(2log(n/d)+5) of Lebarbier (2005).

    :param objs: Dictionary of objective values for each number of change points.
    :param n: Length of the sequence.
    :param alpha: Value of the parameter alpha.
    :return: The estimated number of change points with the given value of alpha.
    """
    objs = np.array([objs[i] for i in range(0, len(objs))]).flatten()
    d = np.arange(1, len(objs)+1)

    return np.argmin(objs + alpha*d/n*(2*np.log(n/d)+5))


def obj_ratios(objs_phys):
    """
    Compute the ratio of successive objective values when going from one number of change points to the next.

    :param objs_phys: Dictionary of objective values for each number of change points.
    :return: ratios: Array with the ratios of successive objective values.
    """
    objs_phys = [objs_phys[i] for i in range(0, len(objs_phys))]
    objs_phys = np.array(objs_phys).flatten()
    ratios = objs_phys[1:]/objs_phys[:-1]

    return ratios


def est_ncp_rule_thumb(ratios, nu):
    """
    Estimate the number of change points using the rule of thumb of Harchaoui and Levy-Leduc (2007).

    :param ratios: Array with the ratios of successive objective values.
    :param nu: Parameter nu for the rule of thumb.
    :return: The estimated number of change points with the given value of nu.
    """
    return np.argwhere(ratios > 1-nu)[0].item()


def estimate_params_loo(cp_results, alphas, nus):
    """
    Estimate the values of alpha and nu in the methods for estimating the number of change points. Do this for each
    cruise by using the annotations from all cruises except that one.

    :param cp_results: Data frame with the cruise names, lengths, number of change points in the annotation files,
                       objective values from change-point estimation, and ratios of objective values for each cruise.
    :param alphas: Parameter values to try for the method of Lebarbier (2005).
    :param nus: Parameter values to try for the method of Harchaoui and Levy-Leduc (2007).
    :return: cp_results: Updated data frame with the estimated number of change points from both methods and the chosen
                         parameter values.
    """
    ncruises = len(cp_results)
    cp_results['n_est_cps_penalty'] = np.zeros(ncruises, dtype=int)
    cp_results['n_est_cps_rule_thumb'] = np.zeros(ncruises, dtype=int)
    for loo_idx in range(ncruises):
        print('Estimating parameters for', cp_results.iloc[loo_idx]['cruise'], '- Cruise ', loo_idx+1, '/', ncruises)
        errors_penalty = np.zeros(len(alphas))
        for alpha_num, alpha in enumerate(alphas):
            for i in range(ncruises):
                if i != loo_idx:
                    n_est_cps = est_ncp_penalty(cp_results.iloc[i]['objs'], cp_results.iloc[i]['n'], alpha)
                    errors_penalty[alpha_num] += np.abs(n_est_cps - cp_results.iloc[i]['n_cp'])
        best_alpha_idx = np.argmin(errors_penalty)

        errors_rule_thumb = np.zeros(len(nus))
        for nu_num, nu in enumerate(nus):
            for i in range(ncruises):
                if i != loo_idx:
                    n_est_cps = est_ncp_rule_thumb(cp_results.iloc[i]['ratios'], nu)
                    errors_rule_thumb[nu_num] += np.abs(n_est_cps - cp_results.iloc[i]['n_cp'])
        best_nu_idx = np.argmin(errors_rule_thumb)

        cp_results.at[loo_idx, 'n_est_cps_penalty'] = int(est_ncp_penalty(cp_results.iloc[loo_idx]['objs'],
                                                              cp_results.iloc[loo_idx]['n'],
                                                              alphas[best_alpha_idx]))
        cp_results.at[loo_idx, 'n_est_cps_rule_thumb'] = int(est_ncp_rule_thumb(cp_results.iloc[loo_idx]['ratios'],
                                                                          nus[best_nu_idx]))
        cp_results.at[loo_idx, 'alpha'] = alphas[best_alpha_idx]
        cp_results.at[loo_idx, 'nu'] = nus[best_nu_idx]

    return cp_results


def estimate_ncp(data_dir, cruises, max_cp, alphas, nus, output_dir, min_dist=5):
    """
    Annotate the change points in the physical data from the directory data_dir.

    :param data_dir: Directory containing the files with the cleaned physical data and annotated change points.
    :param cruises: List of cruises to use.
    :param max_cp: Maximum number of allowable change points.
    :param alphas: Parameter values to try for the method of Lebarbier (2005).
    :param nus: Parameter values to try for the method of Harchaoui and Levy-Leduc (2007).
    :param output_dir: Directory where the annotation results should be stored. The file will be called
                       estimated_ncp.pickle.
    :param min_dist: Minimum allowable distance between change points.
    """
    for cruise_num, cruise in enumerate(cruises):
        print('Estimating physical change points for', cruise, '- Cruise ', cruise_num+1, '/', len(cruises))
        phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))
        n = len(phys_data)
        n_cp = len(json.load(open(os.path.join(data_dir, cruise + '_annotated_phys_cps.json'), 'r')))
        objs_phys = est_cps_objs(phys_data, max_cp, min_dist=min_dist)
        all_ratios = obj_ratios(objs_phys)
        if cruise_num == 0:
            cp_results = pd.DataFrame({'cruise': cruise, 'n': n, 'n_cp': n_cp, 'ratios': [all_ratios],
                                       'objs': [objs_phys]})
        else:
            cp_results = cp_results.append({'cruise': cruise, 'n': n, 'n_cp': n_cp, 'ratios': all_ratios,
                                            'objs': objs_phys}, ignore_index=True)
    cp_results = estimate_params_loo(cp_results, alphas, nus)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cp_results.to_pickle(os.path.join(output_dir, 'estimated_ncp.pickle'))


if __name__ == '__main__':
    t1 = time.time()
    # Directory where the data is stored
    data_dir = '../data/'
    # Directory where the output will be stored
    output_dir = '../results/'
    # List of cruises to use
    cruises = ['DeepDOM', 'KM1712', 'KM1713', 'MGL1704', 'SCOPE_2', 'SCOPE_16', 'Thompson_1', 'Thompson_9',
               'Thompson_12', 'Tokyo_1', 'Tokyo_2', 'Tokyo_3']
    # Parameter values to consider when optimizing the criteria for the number of change points
    alphas = np.arange(0, 1, 0.01)
    nus = np.arange(0.01, 0.1, 0.001)
    # Maximum number of possible change points
    max_cp = 150

    estimate_ncp(data_dir, cruises, max_cp, alphas, nus, output_dir)
    t2 = time.time()
    print('Runtime:', t2-t1)
    # Runtime (Intel i9-7960X CPU @ 2.80GHz): 42s
