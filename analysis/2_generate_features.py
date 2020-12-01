import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import time

from chapydette import feature_generation


def generate_features(cruises, data_dir, features_dir, projection_dims=[128]):
    """
    Generate features for the physical and biological data from each cruise.

    :param cruises: List of cruises to generate features for.
    :param data_dir: Directory where the (cleaned) biological and physical data is stored.
    :param features_dir: Directory where the features will be stored.
    :param projection_dims: List of dimensions to use for the projection for the biological data.
    """
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    for cruise in cruises:
        print('Generating features for', cruise)
        # Load the data
        bio_data = pd.read_parquet(os.path.join(data_dir, cruise + '_bio.parquet'))
        times = np.array(pd.Series(bio_data['date']).astype('category').cat.codes.values + 1)
        bio_data = np.asarray(bio_data[['fsc_small', 'chl_small', 'pe']])

        phys_data = pd.read_parquet(os.path.join(data_dir, cruise + '_phys.parquet'))

        # Generate the features
        phys_features = np.asarray(phys_data[['salinity', 'temp']])
        phys_features = StandardScaler().fit_transform(phys_features)

        for projection_dim in projection_dims:
            print('Dimension of projection:', projection_dim)
            save_file = os.path.join(features_dir, cruise + '_features_' + str(projection_dim) + '.pickle')
            bio_features = feature_generation.nystroem_features(bio_data, projection_dim, window_length=1, do_pca=False,
                                                                window_overlap=0, times=times, seed=0,
                                                                kmeans_iters=100)[0].astype('float64')
            pickle.dump({'bio_features': bio_features, 'phys_features': phys_features}, open(save_file, 'wb'))


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
    t2 = time.time()
    print('Runtime:', t2-t1)
    # Runtimes:
    # CPU (Intel i9-7960X CPU @ 2.80GHz): 13h15m33s
    # GPU (Titan Xp on same machine): 12m10s
