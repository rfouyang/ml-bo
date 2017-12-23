import numpy as np
from skopt import gp_minimize


def func_square(x):
    x = np.array(x)
    return (x.dot(x))


def callback(info):
    print(info['x'], info['fun'])


def main():
    res = gp_minimize(func_square,
                      [(-10.0, 10.0)],
                      base_estimator='GP',
                      n_calls=20,
                      n_random_starts=5,
                      acq_func="LCB",
                      acq_optimizer="auto",
                      verbose=False,
                      callback=callback,
                      n_points=10000,
                      n_restarts_optimizer=5,
                      xi=0.01,
                      kappa=1.96,
                      noise="gaussian",
                      n_jobs=1)

    print(res['x'])


if __name__=='__main__':
    main()