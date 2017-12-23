import os
import numpy as np

import xgboost as xgb
import sklearn.datasets
import sklearn.model_selection
import sklearn.utils
import sklearn.metrics
from bayes_opt import BayesianOptimization


def get_data_iris():
    info = sklearn.datasets.load_iris()
    xs = info['data']
    ys = info['target']
    return xs, ys

class CreditModel(object):
    def __init__(self):
        self.xs, self.ys = get_data_iris()

    def cross_validate(self, param):
        weights = np.ones(len(self.ys))
        for i in range(len(self.ys)):
            if self.ys[i]==1:
                weights[i] *= 3

        dtrain = xgb.DMatrix(self.xs, label=self.ys, feature_names=self.f_name, weight=weights)

        num_round = 1000
        res = xgb.cv(param, dtrain, num_round, nfold=5, metrics={'auc'}, seed=0,
                     callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

        score = res['test-auc-mean'].values[-1]

        loss = 1 - score
        return {'loss': loss, 'status': hyperopt.STATUS_OK}

    def cross_validate_bo(self, eta, min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha):
        param = {
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            'nthread': 4,
            'booster': 'gbtree',
            'tree_method': 'exact',
            'silent': 1,
            'verbose_eval': True
        }
        param['eta'] = eta
        param['min_child_weight'] = int(min_child_weight)
        param['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
        param['max_depth'] = int(max_depth)
        param['subsample'] = max(min(subsample, 1), 0)
        param['gamma'] = max(gamma, 0)
        param['alpha'] = max(alpha, 0)

        weights = np.ones(len(self.ys))
        for i in range(len(self.ys)):
            if self.ys[i] == 1:
                weights[i] *= 3

        dtrain = xgb.DMatrix(self.xs, label=self.ys, feature_names=self.f_name, weight=weights)

        num_round = 1000
        res = xgb.cv(param, dtrain, num_round, nfold=5, metrics={'auc'}, seed=0,
                     callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

        score = res['test-auc-mean'].values[-1]

        return score

    def optimize_bo(self):
        space = {
            'eta': (0.025, 0.5),
            'min_child_weight': (1, 20),
            'colsample_bytree': (0.1, 1),
            'max_depth': (2, 15),
            'subsample': (0.5, 1),
            'gamma': (0, 10),
            'alpha': (0, 10)
        }

        obj = lambda eta, min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha: \
            self.cross_validate_bo(eta, min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha)

        opt = BayesianOptimization(obj, space)
        opt.maximize(init_points=10, n_iter=100)

        max_val = opt.res['max']['max_val']
        max_params = opt.res['max']['max_params']
        print(max_val)
        print(max_params)

        param_fp = os.path.join(settings.BASE_DIR, 'data', 'bo_opt.json')
        io_helper.save_json(param_fp, max_params)

        return max_params