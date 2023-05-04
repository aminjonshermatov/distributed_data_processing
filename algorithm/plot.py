import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def main():
    fig = plt.figure(figsize=(16, 8))
    spec = fig.add_gridspec(2, 4)

    # data[type][vertices]: [y]
    data = [
        [
            [326, 167, 119, 185],  # 240
            [2806, 1322, 941, 1074],  # 480
            [8863, 5111, 3431, 3619],  # 720
            [21377, 11165, 7158, 8053]  # 960
        ],
        [
            [240, 480, 720, 960, 1200, 1440, 1680, 1920, 2160, 2400, 2640, 2880],
            [5, 40, 162, 372, 727, 1263, 1987, 2962, 4208, 5772, 7729, 9981]
        ],
    ]

    for i in range(4):
        ax = fig.add_subplot(spec[0, i])
        ax.set_title(f'MPI N={240 * (i + 1)}')
        x = np.array([1, 2, 3, 4])
        y = np.array(data[0][i])

        X_Y_Spline = make_interp_spline(x, y)

        X_ = np.linspace(x.min(), x.max(), 500)
        Y_ = X_Y_Spline(X_)
        ax.plot(X_, Y_)
        ax.set_xlabel('M, threads')
        ax.set_ylabel('t, ms')

    ax = fig.add_subplot(spec[1, :])
    ax.set_title(f'CUDA')
    x = np.array(data[1][0])
    y = np.array(data[1][1])

    X_Y_Spline = make_interp_spline(x, y)

    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    ax.plot(X_, Y_)
    ax.plot(X_, Y_)
    ax.set_xlabel('N, vertices')
    ax.set_ylabel('t, ms')

    plt.show()


if __name__ == "__main__":
    main()
