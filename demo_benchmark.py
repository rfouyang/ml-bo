import numpy as np
from skopt import gp_minimize


def func_square_1d(x):

    # y = x^2    best 0
    x = np.array(x)
    return x.dot(x)


def func_square_3d(x):
    # y = (x-0)^2 + (x-1)^2 + (x-2)^2    best 0,1,2
    c = np.arange(3)
    x = np.array(x) - c
    return x.dot(x)


def callback(info):
    print(info['x'], info['fun'])


def demo_1d():
    res = gp_minimize(func_square_1d,
                      [(-5.0, 5.0)],
                      base_estimator='GP',
                      n_calls=20,
                      n_random_starts=5,
                      acq_func="LCB",
                      acq_optimizer="auto",
                      verbose=False,
                      callback=callback,
                      n_points=10000,
                      n_restarts_optimizer=5,
                      xi=0.1,
                      kappa=1.96,
                      noise="gaussian",
                      n_jobs=1)

    print(res['x'])


def demo_3d():
    dimensions = [
        (-5.0, 5.0),
        (-5.0, 5.0),
        (-5.0, 5.0)
    ]
    res = gp_minimize(func_square_3d,
                      dimensions,
                      base_estimator='GP',
                      n_calls=100,
                      n_random_starts=30,
                      acq_func="LCB",
                      acq_optimizer="auto",
                      verbose=False,
                      callback=callback,
                      n_points=10000,
                      n_restarts_optimizer=100,
                      xi=0.1,
                      kappa=1.96,
                      noise="gaussian",
                      n_jobs=1)

    print(res['x'])


def main():
    demo_1d()
    demo_3d()



if __name__=='__main__':
    main()