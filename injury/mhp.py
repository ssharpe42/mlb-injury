import copy

import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b


class SimpleMultivariateHawkesProcess:
    _ndim = None  # no. of dimensions
    _lambda = None  # ndim x 1 vector
    _alpha = None  # ndim x ndim matrix
    _beta = None  # ndim x ndim matrix
    _observation_window = None
    _EPS = 10 ** -9  # numbers below this are set to 0

    def __init__(
        self,
        ndim=None,
        lam=None,
        alpha=None,
        beta=None,
        observation_window=None,
    ):
        self._ndim = ndim
        self._lambda = lam
        self._alpha = alpha
        self._beta = beta
        self._observation_window = observation_window

    def fit(
        self,
        ndim,
        alpha,
        data_epochs,
        data_dims,
        observation_window,
        niter=1000,
    ):
        print("Fitting...")
        print("\tPrecomputing A(i), B(i)...")
        # A, B = self._par_precompute(ndim, alpha, data_epochs, data_dims, observation_window)
        A, B = self._precompute(
            ndim, alpha, data_epochs, data_dims, observation_window
        )

        print("\tMinimizing negative loglikelihood...")
        alpha_ravel = alpha.ravel()
        x0 = np.random.uniform(
            low=10 ** -5.0,
            high=alpha_ravel[0] - self._EPS,
            size=(ndim * (ndim + 1)),
        )
        bounds = [(self._EPS, None) for i in range(ndim)] + [
            (self._EPS, None) for i in range(ndim * ndim)
        ]
        # [(self._EPS, None) for i in range(ndim * ndim)]

        x, f, d = fmin_l_bfgs_b(
            self._seq_negloglikelihood,
            x0=x0,
            args=(ndim, alpha, observation_window, A, B),
            factr=10.0 ** 7.0,
            iprint=99,
            pgtol=10 ** -5,
            approx_grad=False,
            callback=None,
            disp=99,
            bounds=bounds,
            maxiter=niter,
            maxfun=niter,
        )

        return x, f, d

    @staticmethod
    def _precompute(ndim, alpha, data_epochs, data_dims, observation_window):
        T0, T = observation_window
        nsamples = len(data_epochs)

        A = []
        B = [
            np.zeros(shape=(ndim, ndim), dtype=np.float64)
            for k in range(nsamples)
        ]
        for k in range(nsamples):
            # memoize locations of timestamps for each dimension for this sample (2s for 18K samples)
            epochs_per_dimension = []
            for i in range(ndim):
                idx_i = data_dims[k] == i
                epochs_i = data_epochs[k][idx_i]
                epochs_per_dimension.append(epochs_i)

            A_k = [
                np.zeros(
                    shape=(ndim, len(epochs_per_dimension[m])),
                    dtype=np.float64,
                )
                for m in range(ndim)
            ]
            # B_mn = np.zeros(shape=(ndim, ndim), dtype=np.float64)
            for m in range(ndim):
                epochs_m = epochs_per_dimension[m]
                if len(epochs_m) == 0:
                    continue  # no transactions on dimension m
                # A_m = np.zeros(shape=(ndim, len(epochs_m)), dtype=np.float64)
                for n in range(ndim):
                    epochs_n = epochs_per_dimension[n]
                    if len(epochs_n) == 0:  # no transactions on dimension n
                        continue
                    len_n = len(epochs_n)
                    A_mn = A_k[m][
                        n
                    ]  # influence of dimension n on each timestamp of dimension m

                    j = 0
                    if m != n:  # mutual excitation

                        while j < len_n and epochs_n[j] < epochs_m[0]:
                            j += 1

                        # A_mn[0] = np.sum(np.exp(-alpha[m,n] * (epochs_m[0] - epochs_n[epochs_n < epochs_m[0]]) / 1000.0))
                        A_mn[0] = np.sum(
                            np.exp(
                                -alpha[m, n]
                                * (epochs_m[0] - epochs_n[:j])
                                / 1000.0
                            )
                        )

                        for i in range(1, len(epochs_m)):
                            prev_j = j
                            while j < len_n and epochs_n[j] < epochs_m[i]:
                                j += 1
                            A_mn[i] = (
                                np.exp(
                                    -alpha[m, n]
                                    * (epochs_m[i] - epochs_m[i - 1])
                                    / 1000.0
                                )
                                * A_mn[i - 1]
                            )
                            # A_mn[i] += np.sum(np.exp(-alpha[m,n] *
                            #                         (epochs_m[i] - epochs_n[(epochs_n >= epochs_m[i-1]) &
                            #                                                 (epochs_n < epochs_m[i])]) / 1000.0))
                            A_mn[i] += np.sum(
                                np.exp(
                                    -alpha[m, n]
                                    * (epochs_m[i] - epochs_n[prev_j:j])
                                    / 1000.0
                                )
                            )

                        B[k][m, n] = np.sum(
                            1 - np.exp(-alpha[m, n] * (T - epochs_n) / 1000.0)
                        )
                    else:  # dim_m = dim_n, self-excitation
                        for i in range(1, len(epochs_m)):
                            A_mn[i] = np.exp(
                                -alpha[m, n]
                                * (epochs_m[i] - epochs_m[i - 1])
                                / 1000.0
                            ) * (1 + A_mn[i - 1])
                        B[k][m, n] = np.sum(
                            1 - np.exp(-alpha[m, n] * (T - epochs_m) / 1000.0)
                        )

                # A_k.append(A_m)

            A.append(A_k)
            # B[k] = B_mn

        return A, B

    def _seq_negloglikelihood(
        self, params, ndim, alpha, observation_window, A, B
    ):
        nsamples = len(A)
        nll = 0.0
        grad = np.zeros(
            shape=((ndim + 1) * ndim), dtype=np.float64
        )  # first row is grad of lambda
        for k in range(nsamples):
            sample_nll, sample_grad = self._negloglikelihood(
                params, ndim, alpha, observation_window, A[k], B[k]
            )
            nll += sample_nll
            grad += sample_grad
        nll /= nsamples
        grad /= nsamples
        return nll, grad

    @staticmethod
    def _negloglikelihood(params, ndim, alpha, observation_window, A, B):
        lam = params[:ndim]  # first ndim elements
        beta = params[ndim:].reshape((ndim, ndim))  # next ndim * ndim elements
        T0, T = observation_window
        T_length = (T - T0) / 1000.0  # scale down

        loglikelihood = 0.0
        grad_lambda = np.zeros(shape=(ndim), dtype=np.float64)
        grad_beta = np.zeros(shape=(ndim, ndim), dtype=np.float64)

        for m in range(ndim):
            Z = lam[m] + np.sum(beta[m, :][:, None] * A[m], axis=0)
            loglikelihood += (
                np.sum(np.log(Z))
                - np.sum(beta[m, :] * B[m, :] / alpha[m, :])
                - lam[m] * T_length
            )
            grad_lambda[m] = np.sum(1.0 / Z) - T_length
            grad_beta[m] = np.sum(A[m] / Z, axis=1) - B[m, :] / alpha[m, :]

            # print np.sum(np.log(Z)) - np.sum(beta[m,:] * B[m,:] / alpha[m,:]) - lam[m] * T_length \
            #   - len(A[m][0]) * np.log(1000) # compare with Nan
            # print 1000.0 * np.sum(1.0/Z) - T_length * 1000.0 # compare with Nan
            # print 1000.0 * grad_beta[m] # compare with nan

        gradient = np.concatenate((grad_lambda, grad_beta.ravel()))
        return -loglikelihood, -gradient

    def check_gradient(
        self,
        ndim,
        alpha,
        data_epochs,
        data_dims,
        observation_window,
        delta=10 ** -9,
        niter=50,
    ):
        print("Checking gradient...")
        print("\tPrecomputing A, B...")
        A, B = self._precompute(
            ndim, alpha, data_epochs, data_dims, observation_window
        )

        nsamples = len(data_epochs)
        for i in range(niter):
            row_idx = np.random.randint(0, nsamples)
            a = A[row_idx]
            b = B[row_idx]
            x0 = np.array(
                [
                    np.random.uniform(10 ** -3, 10 ** 3),
                    np.random.uniform(10 ** -3, 10 ** 3),  # lambda
                    np.random.uniform(10 ** -3, 10 ** 3),
                    np.random.uniform(10 ** -3, 10 ** 3),  # beta
                    np.random.uniform(10 ** -3, 10 ** 3),
                    np.random.uniform(10 ** -3, 10 ** 3),
                ]
            )  # beta

            nll, grad = self._negloglikelihood(
                x0, ndim, alpha, observation_window, a, b
            )
            param_idx = np.random.randint(len(x0))
            x0[param_idx] += delta
            new_nll, new_grad = self._negloglikelihood(
                x0, ndim, alpha, observation_window, a, b
            )
            num_grad = (new_nll - nll) / delta
            if abs((grad[param_idx] - num_grad) / num_grad) > 0.01:
                print("Gradient checking failed for parameter:", param_idx)
                print("Analytical gradient:", grad[param_idx])
                print("Numerical gradient:", num_grad)
        print("\tGradient checking complete.")

    def simulate(self, t0, prev_t, prev_d, horizon_duration):
        simulated_iats = []
        simulated_dims = []
        prev_t_copy = copy.copy(list(prev_t))
        prev_d_copy = copy.copy(list(prev_d))
        prev_t_array = np.array(prev_t_copy, dtype=np.float64)
        prev_d_array = np.array(prev_d_copy, dtype=np.int)
        cumulative_iat = 0.0
        while cumulative_iat < horizon_duration:
            upper_bound = np.sum(
                self.intensity(
                    t0 + cumulative_iat,
                    (prev_t_array, prev_d_array),
                    include_t0=True,
                )
            )
            s = 1000.0 * np.random.exponential(
                1.0 / upper_bound
            )  # params are in 1/1000s units

            cumulative_iat += s  # move ahead in time (may accept or reject)
            if cumulative_iat >= horizon_duration:
                break

            lambda_t_dim = self.intensity(
                t0 + cumulative_iat, (prev_t_array, prev_d_array)
            )
            lambda_t = np.sum(lambda_t_dim)

            u = np.random.uniform(0.0, 1.0)
            if u <= lambda_t / upper_bound:  # accept
                prev_t_copy.append(t0 + cumulative_iat)
                prev_t_array = np.array(prev_t_copy, dtype=np.float64)
                simulated_iats.append(cumulative_iat)

                cumprob = np.cumsum(lambda_t_dim / lambda_t)
                u2 = np.random.uniform(0.0, 1.0)
                dim = np.searchsorted(cumprob, u2)

                prev_d_copy.append(dim)
                prev_d_array = np.array(prev_d_copy, dtype=np.int)
                simulated_dims.append(dim)

        return simulated_iats, simulated_dims

    def predict_dist(self, t0, prev_t, prev_d, niter=1000):
        # step = 3600.0 - (t0 % 3600)
        simulated_iats = np.zeros(shape=(niter), dtype=np.float64)
        simulated_dims = np.zeros(shape=(niter), dtype=np.int)
        for i in range(niter):
            cumulative_iat = 0.0
            while True:
                upper_bound = np.sum(
                    self.intensity(
                        t0 + cumulative_iat, (prev_t, prev_d), include_t0=True
                    )
                )
                s = 1000.0 * np.random.exponential(
                    1.0 / upper_bound
                )  # params are in 1/1000s units

                # print 'Candidate next IAT:', t + s
                # if s >= step: # upper bound no longer valid
                #    t += step
                #    step = 3600.0 # one hour in seconds
                #    #print '\tLarger than step, moved to:', t
                #    continue

                cumulative_iat += (
                    s  # move ahead in time (may accept or reject)
                )
                # step -= s
                lambda_t_dim = self.intensity(
                    t0 + cumulative_iat, (prev_t, prev_d)
                )
                lambda_t = np.sum(lambda_t_dim)
                u = np.random.uniform(0.0, 1.0)

                if u <= lambda_t / upper_bound:  # accept
                    simulated_iats[i] = cumulative_iat
                    if self._ndim > 1:
                        simulated_dims[i] = np.random.choice(
                            range(self._ndim), p=lambda_t_dim / lambda_t
                        )
                    break

        return simulated_iats, simulated_dims

    def intensity(self, t0, prev_e, include_t0=False):
        prev_t, prev_d = prev_e
        if not include_t0:
            prev_t_ = prev_t[prev_t < t0]
            prev_d_ = prev_d[prev_t < t0]
        else:
            prev_t_ = prev_t[prev_t <= t0]
            prev_d_ = prev_d[prev_t <= t0]
        t0_minus_ti = (t0 - prev_t_) / 1000.0
        return self._lambda + np.sum(
            self._beta[:, prev_d_]
            * np.exp(-self._alpha[:, prev_d_] * t0_minus_ti),
            axis=1,
        )

    def intensity_integral(self, start_t, end_t, data_epochs, data_dims):
        # memoize locations of timestamps for each dimension for this sample (2s for 18K samples)
        epochs_per_dimension = []
        for i in range(self._ndim):
            idx_i = data_dims == i
            epochs_i = data_epochs[idx_i]
            epochs_per_dimension.append(epochs_i)

        integral_value = 0.0

        for m in range(self._ndim):
            integral_value += self._lambda[m] * (end_t - start_t) / 1000.0
            for n in range(self._ndim):
                beta_mn = self._beta[m, n]
                alpha_mn = self._alpha[m, n]
                epochs_n = epochs_per_dimension[n]

                epochs_n_ = epochs_n[epochs_n < start_t]
                term1 = np.sum(
                    np.exp(-alpha_mn * (start_t - epochs_n_) / 1000.0)
                    - np.exp(-alpha_mn * (end_t - epochs_n_) / 1000.0)
                )

                epochs_n_ = epochs_n[
                    (epochs_n >= start_t) & (epochs_n < end_t)
                ]
                term2 = np.sum(
                    1 - np.exp(-alpha_mn * (end_t - epochs_n_) / 1000.0)
                )

                integral_value += beta_mn / alpha_mn * (term1 + term2)

        return integral_value
