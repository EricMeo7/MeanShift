import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


EXECUTABLE = '../MeanShiftCorsinoviMeoni/cmake-build-releasepar/meanshift'


def main():
    output_file = "C:/Users/corsinovi/CLionProjects/MeanShiftCorsinoviMeoni/CreateAndDrawClusters/out/out_3D_data_100"

    data = np.genfromtxt('{}.csv'.format(output_file), delimiter=',')
    num_clusters = int(np.max(data[:, -1] + 1))
    clusters = np.ndarray(shape=num_clusters, dtype=np.ndarray)
    for i in range(0, num_clusters):
        clusters[i] = np.float32(
            [point[:-1] for point in data if point[-1] == i]
        )
    fig = plt.figure()
    if len(clusters[0][0]) == 2:
        # 2D plot
        for cluster in clusters:
            plt.scatter(cluster[:, 0], cluster[:, 1], s=3)
    else:
        # 3D plot
        ax = Axes3D(fig)
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_xlim3d([-10, 10])
        ax.set_ylim3d([-10, 10])
        ax.set_zlim3d([-10, 10])
        for cluster in clusters:
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=3)

    plt.show()


if __name__ == '__main__':
    main()
