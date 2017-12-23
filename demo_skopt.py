import xgboost as xgb
import sklearn.datasets
from skopt import gp_minimize


def get_data_breast_cancer():
    info = sklearn.datasets.load_breast_cancer()
    xs = info['data']
    ys = info['target']
    return xs, ys

class DemoBO(object):
    def __init__(self):
        self.xs, self.ys = get_data_breast_cancer()

    def cross_validate_bo(self, theta):
        param = {
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            'nthread': 4,
            'booster': 'gbtree',
            'tree_method': 'exact',
            'silent': 1,
            'verbose_eval': False
        }

        param['eta'] = theta[0]
        param['min_child_weight'] = theta[1]
        param['cosample_bytree'] = theta[2]
        param['max_depth'] = theta[3]
        param['subsample'] = theta[4]
        param['gamma'] = theta[5]
        param['alpha'] = theta[6]

        dtrain = xgb.DMatrix(self.xs, label=self.ys)

        num_round = 200
        res = xgb.cv(param, dtrain, num_round, nfold=5, metrics={'auc'}, seed=0)
        score = res['test-auc-mean'].values[-1]

        return -score

    def callback(self, info):
        print(info['x'], info['fun'])

    def optimize_bo(self):
        dimensions = [
            (0.025, 0.5), # eta
            (1, 20), # min_child_weight
            (0.1, 1), # colsample_bytree
            (2, 15), # max_depth
            (0.5, 1.0), # subsample
            (0.0, 10.0), # gamma,
            (0.0, 10.0) # alpha
        ]

        res = gp_minimize(self.cross_validate_bo,
                          dimensions ,
                          base_estimator='GP',
                          n_calls=20,
                          n_random_starts=5,
                          acq_func="LCB",
                          acq_optimizer="auto",
                          verbose=False,
                          callback=self.callback,
                          n_points=10000,
                          n_restarts_optimizer=5,
                          xi=0.01,
                          kappa=1.96,
                          noise="gaussian",
                          n_jobs=1)

        print(res['space'])
        print(res['x'], res['fun'])
        return res['x']


def main():
    model = DemoBO()
    model.optimize_bo()


if __name__=='__main__':
    main()