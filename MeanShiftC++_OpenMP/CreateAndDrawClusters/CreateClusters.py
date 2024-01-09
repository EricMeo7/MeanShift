import os

import sklearn.datasets as datasets
import pandas as pd


def generate_dataset(points, n_features, centers, std, file_name, output_directory):
    data, labels = datasets.make_blobs(n_samples=points, n_features=n_features,
                                       centers=centers, cluster_std=std, shuffle
                                       =True, random_state=5000)
    pd.DataFrame(data).to_csv(
        os.path.join(output_directory, file_name), header=False, index=False
    )


def main():

    '''Datasets with an increasing number of points'''

    DATASETS_DIR = '../input/newCluster'

    # three dimensional datasets, five clusters

    for points in [100, 1000, 10000, 20000, 50000, 100000, 200000]:
        generate_dataset(points, n_features=3, centers=6, std=1, file_name=f'3D_data_{points}.csv', output_directory=DATASETS_DIR)


if __name__ == '__main__':
    main()