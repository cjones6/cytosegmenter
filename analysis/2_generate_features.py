import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import time

from chapydette import feature_generation


def generate_features(cruises, data_dir, features_dir, projection_dims=[128], subsample_every=1):
    """
    Generate features for the physical and biological data from each cruise.

    :param cruises: List of cruises to generate features for.
    :param data_dir: Directory where the (cleaned) biological and physical data is stored.
    :param features_dir: Directory where the features will be stored.
    :param projection_dims: List of dimensions to use for the projection for the biological data.
    :param subsample_every: How to subsample the data (if at all). If this value is n, then every nth point cloud is
                            kept.
    """
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    for cruise in cruises:
        print('Generating features for', cruise)
        # Load the data
        bio_data = pd.read_parquet(os.path.join(data_dir, cruise + '_bio.parquet'))
        times = np.array(pd.Series(bio_data['date']).astype('category').cat.codes.values + 1)
        bio_data = np.log10(np.asarray(bio_data[['fsc_small', 'chl_small', 'pe']]))

        phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))

        # Generate the features
        phys_features = np.asarray(phys_data[['salinity', 'temp']])
        phys_features = StandardScaler().fit_transform(phys_features)

        for subsample_num in range(subsample_every):
            print('(Sub)sample', subsample_num+1, 'of', subsample_every)
            subsample_idxs = np.where((times-1) % subsample_every == subsample_num)[0]
            bio_data_subsample = bio_data[subsample_idxs]
            times_subsample = times[subsample_idxs]

            for projection_dim in projection_dims:
                print('Dimension of projection:', projection_dim)
                if subsample_every == 1:
                    save_file = os.path.join(features_dir, cruise + '_features_' + str(projection_dim) + '.pickle')
                else:
                    save_file = os.path.join(features_dir, cruise + '_features_' + str(projection_dim) + '_subsample_' +
                                             str(subsample_num+1) + '_of_' + str(subsample_every) + '.pickle')
                bio_features, _, _, scaler, _, centroids, bandwidth = feature_generation.nystroem_features(
                                                                                          bio_data_subsample,
                                                                                          projection_dim,
                                                                                          window_length=1,
                                                                                          do_pca=False,
                                                                                          window_overlap=0,
                                                                                          times=times_subsample,
                                                                                          seed=0,
                                                                                          kmeans_iters=100,
                                                                                          standardize=True)
                pickle.dump({'bio_features': bio_features.astype('float64'), 'phys_features': phys_features,
                             'bandwidth': bandwidth, 'centroids': centroids, 'scaler': scaler}, open(save_file, 'wb'))


def generate_features_simple_avg(cruises, data_dir, features_dir):
    """
    Generate features for the biological data from each cruise by simply averaging the observations within each point
    cloud.

    :param cruises: List of cruises to generate features for.
    :param data_dir: Directory where the (cleaned) biological and physical data is stored.
    :param features_dir: Directory where the features will be stored.
    """
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    for cruise in cruises:
        print('Generating features based on a simple average for', cruise)
        # Load the data
        bio_data = pd.read_parquet(os.path.join(data_dir, cruise + '_bio.parquet'))
        times = np.array(pd.Series(bio_data['date']).astype('category').cat.codes.values + 1)
        bio_data = np.log10(np.asarray(bio_data[['fsc_small', 'chl_small', 'pe']]))
        scaler = StandardScaler().fit(bio_data)
        bio_data = scaler.transform(bio_data)
        bio_data = pd.DataFrame({'fsc_small': bio_data[:, 0], 'chl_small': bio_data[:, 1], 'pe': bio_data[:, 2],
                                 'time': times})

        # Generate the features
        bio_features = bio_data.groupby('time', as_index=False).mean()
        bio_features = bio_features[['fsc_small', 'chl_small', 'pe']]

        save_file = os.path.join(features_dir, cruise + '_features_simple_avg.pickle')
        pickle.dump({'bio_features': np.array(bio_features.astype('float64')), 'scaler': scaler}, open(save_file, 'wb'))


if __name__ == '__main__':
    t1 = time.time()
    # Directory where the data is stored
    data_dir = '../data/'
    # Directory where the output will be stored
    features_dir = '../features/'
    # List of cruises to use
    cruises = ['DeepDOM', 'KM1712', 'KM1713', 'MGL1704', 'SCOPE_2', 'SCOPE_16', 'Thompson_1', 'Thompson_9',
               'Thompson_12', 'Tokyo_1', 'Tokyo_2', 'Tokyo_3']

    generate_features(cruises, data_dir, features_dir, projection_dims=[128])
    generate_features(['SCOPE_16'], data_dir, features_dir, projection_dims=[2**i for i in range(1, 11)])
    generate_features(['SCOPE_16'], data_dir, features_dir, subsample_every=10)
    generate_features_simple_avg(cruises, data_dir, features_dir)
    t2 = time.time()
    print('Runtime:', t2-t1)
    # Runtimes:
    # CPU (Intel i9-7960X CPU @ 2.80GHz): ~14h
    # GPU (Titan Xp on same machine): 13m23s
