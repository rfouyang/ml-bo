import numpy as np
import operator
import matplotlib.pyplot as plt
import GPy


def branin(x, param):
    # x1: [-5, 10], x2:[0, 15]
    y = (x[1]-5.1/(4*np.pi**2)*x[0]**2+5*x[0]/np.pi-6)**2
    y += 10*(1-1/(8*np.pi))*np.cos(x[0])+10
    return -y


def schw(x, param):
    # [-500,500]
    n = 2
    s = -x[0]*np.sin(np.sqrt(np.abs(x[0])))-x[1]*np.sin(np.sqrt(np.abs(x[1])))
    y = 418.9829*n+s
    return y


def sphere(x, param):
    # [-5,5]
    y = -x.dot(x)
    return y


def func_arg(x, param):
    # [-5,5]
    y = param['a']*x[0]*x[0] + x[1]*x[1]
    return y


class BO:
    def __init__(self):
        self.lst_Xt = []; self.lst_Yt = []
        self.Xt = None; self.Yt = None
        self.Xp = None; self.Yp = None; self.Vp = None; self.Up = None

        self.theta = dict(); self.gp = None
        self.x_best = None; self.x_next = None
        self.t = 1

    def config(self, dim_x, lx, ux, grid, ker_type, fn, param):
        self.theta = dict()

        self.theta['D'] = dim_x
        self.theta['ker_type'] = ker_type

        self.theta['func'] = fn
        self.theta['param'] = param
        self.theta['lx'] = lx; self.theta['ux'] = ux; self.theta['grid'] = grid

    def get_Xp_grid(self):
        D = self.theta['D']
        lx = self.theta['lx']; ux = self.theta['ux']; grid = self.theta['grid']

        x_set = []
        for d in range(0, D):
            x_d = np.linspace(lx[d],ux[d],grid[d])
            x_set.append(x_d)

        x_d1 = x_set[0]

        ans = []
        for i in range(0, len(x_d1)):
            z = [x_d1[i]]
            ans.append(z)

        for d in range(1, D):
            x_d1 = ans
            x_d2 = x_set[d]
            ans = []
            for i in range(0, len(x_d1)):
                for j in range(0, len(x_d2)):
                    z = x_d1[i] + [x_d2[j]]
                    ans.append(z)

        self.Xp = np.atleast_2d(ans)

    def narrow_Xp_grid(self):
        D = self.theta['D']
        lx = self.theta['lx']; ux = self.theta['ux']; grid = self.theta['grid']

        span = (ux - lx)/4.0
        lx = self.x_best - span; ux = self.x_best + span
        self.theta['lx'] = lx; self.theta['ux'] = ux

        x_set = []
        for d in range(0, D):
            x_d = np.linspace(lx[d],ux[d],grid[d])
            x_set.append(x_d)

        x_d1 = x_set[0]

        ans = []
        for i in range(0, len(x_d1)):
            z = [x_d1[i]]
            ans.append(z)

        for d in range(1, D):
            x_d1 = ans
            x_d2 = x_set[d]
            ans = []
            for i in range(0, len(x_d1)):
                for j in range(0, len(x_d2)):
                    z = x_d1[i] + [x_d2[j]]
                    ans.append(z)

        self.Xp = np.atleast_2d(ans)

    def init_Xt_grid(self, N):
        D = self.theta['D']
        lx = self.theta['lx']; ux = self.theta['ux']

        func = self.theta['func']; param = self.theta['param']

        x_set = []
        for d in range(0, D):
            x_d = np.linspace(lx[d], ux[d], N)
            x_set.append(x_d)

        x_d1 = x_set[0]

        ans = []
        for i in range(0, len(x_d1)):
            z = [x_d1[i]]
            ans.append(z)


        for d in range(1, D):
            x_d1 = ans
            x_d2 = x_set[d]
            ans = []
            for i in range(0, len(x_d1)):
                for j in range(0, len(x_d2)):
                    z = x_d1[i] + [x_d2[j]]
                    ans.append(z)

        self.lst_Xt = []; self.lst_Yt = []
        for i in range(0, len(ans)):
            x = np.array(ans[i])
            self.lst_Xt.append(x); self.lst_Yt.append(func(x, param))

    def add_Xt(self, x, y):
        self.lst_Xt.append(x)
        self.lst_Yt.append(y)

    def distance(self, x, y):
        z = x-y
        return np.sqrt(z.dot(z))

    def is_close(self, x, eps):
        dist = [self.distance(x, y) for y in self.lst_Xt]
        idx, v = min(enumerate(dist), key=operator.itemgetter(1))
        return True if v < eps else False

    def plot2D(self, plot_freq, is_demo):
        func = self.theta['func']; param = self.theta['param']
        grid = self.theta['grid']
        if is_demo is True:
            Yd = np.atleast_2d([func(x, param) for x in self.Xp]).T

        if self.Yp is None:
            plt.figure(1)
            if is_demo is True:
                plt.imshow(Yd.reshape(grid[0], grid[1]))
            plt.title('true function')
            plt.show()
        else:
            f, ax = plt.subplots(2, 2)
            if is_demo is True:
                ax[0][0].imshow(Yd.reshape(grid[0], grid[1]))
            ax[0][1].imshow(self.Vp.reshape(grid[0], grid[1]))
            ax[1][0].imshow(self.Yp.reshape(grid[0], grid[1]))
            ax[1][1].imshow(self.Up.reshape(grid[0], grid[1]))

            lx = self.theta['lx']
            ux = self.theta['ux']
            grid = self.theta['grid']

            for i in range(0, len(self.lst_Xt)):
                x = self.lst_Xt[i]
                r = int(np.floor(1.0 * grid[0] * (x[0] - lx[0]) / (ux[0] - lx[0])))
                c = int(np.floor(1.0 * grid[1] * (x[1] - lx[1]) / (ux[1] - lx[1])))
                ax[0][0].scatter(x=c, y=r, c='r', s=40, marker='x')
                ax[0][1].scatter(x=c, y=r, c='r', s=40, marker='x')

            x = self.lst_Xt[-1]
            r = int(np.floor(1.0 * grid[0] * (x[0] - lx[0]) / (ux[0] - lx[0])))
            c = int(np.floor(1.0 * grid[1] * (x[1] - lx[1]) / (ux[1] - lx[1])))
            ax[0][0].scatter(x=c, y=r, c='g', s=40, marker='o')
            ax[0][1].scatter(x=c, y=r, c='g', s=40, marker='o')

            if self.x_next is not None:
                x = self.x_next
                r = int(np.floor(1.0 * grid[0] * (x[0] - lx[0]) / (ux[0] - lx[0])))
                c = int(np.floor(1.0 * grid[1] * (x[1] - lx[1]) / (ux[1] - lx[1])))
                ax[1][1].scatter(x=c, y=r, c='g', s=40, marker='o')

            if self.x_best is not None:
                x = self.x_best
                r = int(np.floor(1.0 * grid[0] * (x[0] - lx[0]) / (ux[0] - lx[0])))
                c = int(np.floor(1.0 * grid[1] * (x[1] - lx[1]) / (ux[1] - lx[1])))
                ax[1][0].scatter(x=c, y=r, c='r', s=40, marker='o')

            ax[0][0].set_title('true function')
            ax[1][0].set_title('predicted function')
            ax[0][1].set_title('predicted uncertainty')
            ax[1][1].set_title('acquisition function')

            if (self.t % plot_freq) == 0:
                plt.savefig('fig/' + str(self.t) + '.png')

    def plot1D(self, plot_freq, is_demo):
        func = self.theta['func']; param = self.theta['param']
        grid = self.theta['grid']
        if is_demo is True:
            Yd = [func(x, param) for x in self.Xp]

        if self.Yp is None:
            plt.figure()
            if is_demo is True:
                plt.plot(Yd)
            plt.title('true function')
            plt.show()
        else:
            f, ax = plt.subplots(2, 2)
            if is_demo is True:
                ax[0][0].plot(Yd)
            ax[0][1].plot(self.Vp)
            ax[1][0].plot(self.Yp)
            ax[1][1].plot(self.Up)

            lx = self.theta['lx']
            ux = self.theta['ux']
            grid = self.theta['grid']

            for i in range(0, len(self.lst_Xt)):
                x = self.lst_Xt[i]
                r = int(np.floor(1.0 * grid[0] * (x[0] - lx[0]) / (ux[0] - lx[0])))
                ax[0][0].axvline(r, color='r')
                ax[0][1].axvline(r, color='r')

            x = self.lst_Xt[-1]
            r = int(np.floor(1.0 * grid[0] * (x[0] - lx[0]) / (ux[0] - lx[0])))
            ax[0][0].axvline(r, color='g')
            ax[0][1].axvline(r, color='g')

            if self.x_next is not None:
                x = self.x_next
                r = int(np.floor(1.0 * grid[0] * (x[0] - lx[0]) / (ux[0] - lx[0])))
                ax[1][1].axvline(r, color='g')

            if self.x_best is not None:
                x = self.x_best
                r = int(np.floor(1.0 * grid[0] * (x[0] - lx[0]) / (ux[0] - lx[0])))
                ax[1][0].axvline(r, color='r')

            ax[0][0].set_title('true function')
            ax[1][0].set_title('predicted function')
            ax[0][1].set_title('predicted uncertainty')
            ax[1][1].set_title('acquisition function')

            if (self.t % plot_freq) == 0:
                plt.savefig('fig/' + str(self.t) + '.png')

    def plot(self, plot_freq, is_demo):
        D = self.theta['D']
        if D == 1:
            self.plot1D(plot_freq, is_demo)
        elif D == 2:
            self.plot2D(plot_freq, is_demo)
        else:
            print('can only plot 1D or 2D.')

    def gp_init(self):
        lx = self.theta['lx']; ux = self.theta['ux']
        grid = self.theta['grid']; D = self.theta['D']
        ker_type = self.theta['ker_type']

        self.Xt = np.atleast_2d(self.lst_Xt)
        self.Yt = np.atleast_2d(self.lst_Yt).T

        mf = GPy.core.Mapping(D, 1)
        mf.f = lambda x:np.mean(self.lst_Yt)
        mf.update_gradients = lambda a, b: None

        len = sum(ux-lx)/D
        sig = (np.max(self.lst_Yt) - np.min(self.lst_Yt))/10.0

        kr = GPy.kern.RBF(input_dim=D, ARD=True)
        km32 = GPy.kern.Matern32(input_dim=D, ARD=True)
        km52 = GPy.kern.Matern52(input_dim=D, ARD=True)
        kb = GPy.kern.Bias(input_dim=D)
        kw = GPy.kern.White(input_dim=D)

        if ker_type == 'RBF':
            k = kr + kb + kw
            self.gp = GPy.models.GPRegression(self.Xt, self.Yt, k, mean_function=mf)
            self.gp.sum.rbf.lengthscale = len*np.ones(D)/20
            self.gp.sum.rbf.variance = sig
            self.gp.Gaussian_noise.variance = sig/100.0
        elif ker_type == 'Matern32':
            k = km32 + kb + kw
            self.gp = GPy.models.GPRegression(self.Xt, self.Yt, k, mean_function=mf)
            self.gp.sum.Mat32.lengthscale = len*np.ones(D)/20
            self.gp.sum.Mat32.variance = sig
            self.gp.Gaussian_noise.variance = sig/100.0
        elif ker_type == 'Matern52':
            k = km52 + kb + kw
            self.gp = GPy.models.GPRegression(self.Xt, self.Yt, k, mean_function=mf)
            self.gp.sum.Mat52.lengthscale = len * np.ones(D)/20
            self.gp.sum.Mat52.variance = sig
            self.gp.Gaussian_noise.variance = sig/100.0

        self.gp['.*var'].constrain_bounded(sig / 1000, 1000 * sig)
        self.gp['.*noise'].constrain_bounded(sig / 10000, sig / 10)
        self.gp['.*lengthscale'].constrain_bounded(len / (60.0), 2 * len)
        self.gp.Gaussian_noise.variance.constrain_bounded(sig / 10000, sig / 10)

    def gp_train(self):
        lx = self.theta['lx']; ux = self.theta['ux']
        grid = self.theta['grid']; D = self.theta['D']
        ker_type = self.theta['ker_type']

        self.Xt = np.atleast_2d(self.lst_Xt)
        self.Yt = np.atleast_2d(self.lst_Yt).T

        mf = GPy.core.Mapping(D, 1)
        mf.f = lambda x: np.mean(self.lst_Yt)
        mf.update_gradients = lambda a, b: None

        len = sum(ux - lx) / D
        sig = (np.max(self.lst_Yt) - np.min(self.lst_Yt)) / 10.0

        kr = GPy.kern.RBF(input_dim=D, ARD=True)
        km32 = GPy.kern.Matern32(input_dim=D, ARD=True)
        km52 = GPy.kern.Matern52(input_dim=D, ARD=True)
        kb = GPy.kern.Bias(input_dim=D)
        kw = GPy.kern.White(input_dim=D)
        k = None

        if ker_type == 'RBF':
            k = kr + kb + kw
            self.gp = GPy.models.GPRegression(self.Xt, self.Yt, k, mean_function=mf)
            self.gp.sum.rbf.lengthscale = len * np.ones(D) / 20
            self.gp.sum.rbf.variance = sig
            self.gp.Gaussian_noise.variance = sig / 100.0
        elif ker_type == 'Matern32':
            k = km32 + kb + kw
            self.gp = GPy.models.GPRegression(self.Xt, self.Yt, k, mean_function=mf)
            self.gp.sum.Mat32.lengthscale = len * np.ones(D) / 20
            self.gp.sum.Mat32.variance = sig
            self.gp.Gaussian_noise.variance = sig / 100.0
        elif ker_type == 'Matern52':
            k = km52 + kb + kw
            self.gp = GPy.models.GPRegression(self.Xt, self.Yt, k, mean_function=mf)
            self.gp.sum.Mat52.lengthscale = len * np.ones(D) / 20
            self.gp.sum.Mat52.variance = sig
            self.gp.Gaussian_noise.variance = sig / 100.0

        self.gp['.*var'].constrain_bounded(sig / 1000, 1000 * sig)
        self.gp['.*noise'].constrain_bounded(sig / 10000, sig / 10)
        self.gp['.*lengthscale'].constrain_bounded(len / (60.0), 2 * len)
        self.gp.Gaussian_noise.variance.constrain_bounded(sig / 10000, sig / 10)

        self.gp.randomize()
        self.gp.optimize('scg', xtol=1e-6, ftol=1e-6, max_iters=50, messages=False)
        self.gp.optimize_restarts(robust=True, num_restarts=10, max_iters=50)

    def gp_pred(self):
        self.Yp, self.Vp = self.gp.predict(self.Xp)

    def gp_ucb(self):
        D = self.theta['D']
        delta = 0.1
        v = 1.0*D
        tau = 2.0 * np.log(self.t ** (2.0 + D / 2.0) * (np.pi ** 2) / (3.0 * delta))
        beta = v * tau

        UCB = [beta * np.sqrt(np.finfo(np.float32).eps+self.Vp[i][0]) + self.Yp[i][0] for i in range(len(self.Xp))]
        self.Up = np.atleast_2d(UCB).T

    def gp_next(self, eps):
        lx = self.theta['lx']; ux = self.theta['ux']
        grid = self.theta['grid']; D = self.theta['D']
        func = self.theta['func']; param = self.theta['param']

        y_max = self.Yp[0][0]
        idx = 0
        for i in range(0, len(self.Xp)):
            if y_max < self.Yp[i][0]:
                y_max = self.Yp[i][0]
                idx = i
        x = self.Xp[idx]
        self.x_best = x

        ucb_max = 0
        for i in range(0, len(self.Up)):
            x = self.Xp[i]
            if self.is_close(x, eps/(self.t*self.t)) == False:
                ucb_max = self.Up[i]
                break

        idx = 0
        for i in range(0, len(self.Up)):
            x = self.Xp[i]
            if ucb_max < self.Up[i] and self.is_close(x, eps/(self.t*self.t)) == False:
                ucb_max = self.Up[i][0]
                idx = i
        x = self.Xp[idx]
        y = func(x, param)

        self.x_next = x

        return x, y

    def gp_obs(self):
        self.Xt = np.atleast_2d(self.lst_Xt)
        self.Yt = np.atleast_2d(self.lst_Yt).T
        self.gp.set_XY(self.Xt, self.Yt)


def optimize(dim_x, lx, ux, grid, fn, param, **kwargs):
    T = kwargs['T'] if 'T' in kwargs.keys() else 50
    N = kwargs['N'] if 'N' in kwargs.keys() else 3
    eps = kwargs['eps'] if 'eps' in kwargs.keys() else 0.01
    ker_type = kwargs['ker_type'] if 'ker_type' in kwargs.keys() else 'RBF'
    train_start = kwargs['train_start'] if 'train_start' in kwargs.keys() else 5
    train_freq = kwargs['train_freq'] if 'train_freq' in kwargs.keys() else 20
    narrow_start = kwargs['narrow_start'] if 'narrow_start' in kwargs.keys() else T
    narrow_freq = kwargs['narrow_freq'] if 'narrow_freq' in kwargs.keys() else 200
    is_plot = kwargs['is_plot'] if 'is_plot' in kwargs.keys() else False
    plot_freq = kwargs['plot_freq'] if 'plot_freq' in kwargs.keys() else 1
    is_demo = kwargs['is_demo'] if 'is_demo' in kwargs.keys() else False


    bo = BO()
    bo.config(dim_x, lx, ux, grid, ker_type, fn, param)
    bo.get_Xp_grid()
    bo.init_Xt_grid(N)

    bo.gp_init()
    while bo.t <= T:
        bo.gp_obs(); bo.gp_pred(); bo.gp_ucb()
        x, y = bo.gp_next(eps)
        bo.add_Xt(x, y)
        bo.t += 1

        if is_plot is True:
            bo.plot(plot_freq, is_demo)

        if bo.t > train_start and bo.t % train_freq == 0:
            bo.gp_train()

        if bo.t > narrow_start and bo.t % narrow_freq == 0:
            bo.narrow_Xp_grid()

        print('\n\n-----\niteration = ', bo.t, 'x_best = ', bo.x_best, '\n-----\n\n')

    return bo.x_best


def help():
    print('Bayesian Optimization\n')
    print('Instruction:')
    print('import numpy as np')
    print('import BO')
    print('BO.optimize(dim_x, lx, ux, grid, fn, param, **kwargs)')
    print('Example: BO.optimize(2, np.array([-5,0]), np.array([10,15]), np.array([100,100]), BO.branin, None)')
    print('\n')
    print('Parameters:')
    print('dim_x:\tfunction input dimension, type: int')
    print('lx:\tfunction input lower bound, type: np.array')
    print('ux:\tfunction input upper bound, type: np.array')
    print('grid:\tfunction input discretization number, type: np.array')
    print('fn:\tthe target function need to be optimized, type: function')
    print('param:\tthe other parameters in the target function except the input, type: dict')
    print('\n')
    print('Other Options:')
    print('N: the initial sample size of BO, value: [3, inf]')
    print('T: the maximum iteration of BO, value: [1, inf]')
    print('eps: the minimum distance between two samples, value: (0, inf]')
    print('ker_type: the kernel for Gaussian process used in BO, value: {\'RBF\', \'Matern32\', \'Matern52\'}')
    print('train_start: the time starting to train Gaussian process, value: [1, inf]')
    print('train_freq: the frequency to train Gaussian process, value: [1, inf]')
    print('narrow_start: the time starting to narrow down the searching space, value: [1, inf]')
    print('narrow_freq: the frequency to narrow down the searching space, value: [1, inf]')
    print('is_plot: plot the procedure or not, value: {True,False}')
    print('plot_freq: plotting frequency, value: [1, inf]')
    print('is_demo: plot the true function or not, value: {True,False}')


def demo_1D():
    optimize(1, np.array([-5]), np.array([5]), np.array([100]), sphere, None,
             T = 30, ker_type = 'RBF', is_demo=True, is_plot=True)


def demo_2D():
    optimize(2, np.array([-5, -5]), np.array([5, 5]), np.array([100, 100]), sphere, None,
             T = 30, ker_type = 'Matern32', is_demo=True, is_plot=True, plot_freq = 5)


def main():
    demo_2D()

if __name__ == '__main__':
    main()
