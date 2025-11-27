import numpy as np
# from DTAFNSModels import generate_path
from DTAFNS_model_forward_measure import DTAFNS_forward_measure
from DTAFNS_zero_coupon_price_multipaths import DTAFNS_close
# import matplotlib.pyplot as plt
import time


def swap_swaption_simulation(param_kappa_Q, param_theta_Q, param_rho, param_sigma, param_lambda, p_initial_factors,
                             param_T_alpha, param_T_beta, param_fixed_rate, p_notional_value, stepsize=1 / 12,
                             npaths=100000,
                             swaptype='payer', seed=None):
    # times_pricing = list(range(0, param_T_alpha + 1))
    times_swaps = list(range(param_T_alpha, param_T_beta + 1))

    # factorpath, _ =  generate_path(param_kappa_P, param_theta_P, param_rho, param_sigma, p_initial_factors,
    # param_T_alpha, npaths )
    dtafns_fm = DTAFNS_forward_measure(Xt=p_initial_factors, kappa=param_kappa_Q, theta=param_theta_Q,
                                       sigma=param_sigma,
                                       rho=param_rho, T=param_T_alpha * stepsize,
                                       t=0, delta_t=stepsize, lam=param_lambda, seed=seed)
    factorpath = dtafns_fm.generate_sample(sizes=npaths)

    DTAFNS_price = np.zeros((npaths, len(times_swaps)))
    DTAFNS_price_sr = np.zeros(len(times_swaps))

    for j, TT in enumerate(times_swaps):
        dtafns = DTAFNS_close(Xt=factorpath, kappa=param_kappa_Q, theta=param_theta_Q, sigma=param_sigma,
                              rho=param_rho, T=TT * stepsize, t=param_T_alpha * stepsize, delta_t=stepsize,
                              lam=param_lambda)
        DTAFNS_price[:, j] = dtafns.price_zero_coupon()
        dtafns_sr = DTAFNS_close(Xt=p_initial_factors.reshape(1, -1), kappa=param_kappa_Q, theta=param_theta_Q,
                                 sigma=param_sigma, rho=param_rho, T=TT * stepsize, t=0, delta_t=stepsize,
                                 lam=param_lambda)
        DTAFNS_price_sr[j] = dtafns_sr.price_zero_coupon()

    swap_rate = (DTAFNS_price_sr[0] - DTAFNS_price_sr[-1]) / (stepsize * np.sum(DTAFNS_price_sr[1:]))

    # dtafns2 = DTAFNS_close(Xt=p_initial_factors.reshape(1, -1), kappa=param_kappa_Q, theta=param_theta_Q,
    #                        sigma=param_sigma,
    #                        rho=param_rho, T=param_T_alpha * stepsize, t=0, delta_t=stepsize,
    #                        lam=param_lambda)
    DTAFNS_price_0 = DTAFNS_price_sr[0]  # dtafns2.price_zero_coupon()
    payoff = 0
    if swaptype == 'payer':
        payoff = DTAFNS_price[:, 0] - DTAFNS_price[:, -1] - param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:],
                                                                                                 axis=1)
        payoff[payoff < 0] = 0
    elif swaptype == 'receiver':
        payoff = param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:], axis=1) - (
                DTAFNS_price[:, 0] - DTAFNS_price[:, -1])
        payoff[payoff < 0] = 0

    swaption_price = DTAFNS_price_0 * (p_notional_value * payoff).mean()
    return swaption_price, swap_rate, DTAFNS_price


def swaprate_swaption_pricing(param_kappa, param_theta, param_rho, param_sigma, param_lambda, p_initial_factors,
                              factorpath_Talpha,
                              param_T_alpha, param_T_beta, param_fixed_rate, p_notional_value, stepsize=1 / 12,
                              npaths=100000,
                              swaptype='payer', seed=None):
    # times_pricing = list(range(0, param_T_alpha + 1))
    times_swaps = list(range(param_T_alpha, param_T_beta + 1))
    # dtafns_fm = DTAFNS_forward_measure(Xt=p_initial_factors, kappa=param_kappa, theta=param_theta, sigma=param_sigma,
    #                                    rho=param_rho, T=param_T_alpha * stepsize,
    #                                    t=0, delta_t=stepsize, lam=param_lambda, seed=seed)
    # factorpath = dtafns_fm.generate_sample(sizes=npaths)

    DTAFNS_price = np.zeros((npaths, len(times_swaps)))
    DTAFNS_price_sr = np.zeros(len(times_swaps))

    for j, TT in enumerate(times_swaps):
        dtafns = DTAFNS_close(Xt=factorpath_Talpha, kappa=param_kappa, theta=param_theta, sigma=param_sigma,
                              rho=param_rho, T=TT * stepsize, t=param_T_alpha * stepsize, delta_t=stepsize,
                              lam=param_lambda)
        DTAFNS_price[:, j] = dtafns.price_zero_coupon()
        dtafns_sr = DTAFNS_close(Xt=p_initial_factors.reshape(1, -1), kappa=param_kappa, theta=param_theta,
                                 sigma=param_sigma, rho=param_rho, T=TT * stepsize, t=0, delta_t=stepsize,
                                 lam=param_lambda)
        DTAFNS_price_sr[j] = dtafns_sr.price_zero_coupon()

    swap_rate = (DTAFNS_price_sr[0] - DTAFNS_price_sr[-1]) / (stepsize * np.sum(DTAFNS_price_sr[1:]))

    # dtafns2 = DTAFNS_close(Xt=p_initial_factors.reshape(1, -1), kappa=param_kappa, theta=param_theta,
    # sigma=param_sigma, rho=param_rho, T=param_T_alpha * stepsize, t=0, delta_t=stepsize, lam=param_lambda)
    DTAFNS_price_0 = DTAFNS_price_sr[0]
    payoff = 0
    if swaptype == 'payer':
        payoff = DTAFNS_price[:, 0] - DTAFNS_price[:, -1] - param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:],
                                                                                                 axis=1)
        payoff[payoff < 0] = 0
    elif swaptype == 'receiver':
        payoff = param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:], axis=1) - (
                DTAFNS_price[:, 0] - DTAFNS_price[:, -1])
        payoff[payoff < 0] = 0

    swaption_price = DTAFNS_price_0 * (p_notional_value * payoff).mean()
    return swaption_price, swap_rate, DTAFNS_price


def swaprate_swaption_pricing_for_grad_test(param_kappa, param_theta, param_rho, param_sigma, param_lambda,
                                            p_initial_factors, factorpath_Talpha,
                                            param_T_alpha, param_T_beta, param_fixed_rate, p_notional_value,
                                            stepsize=1 / 12,
                                            npaths=100000,
                                            swaptype='payer', seed=None, t_pricing=0):
    # times_pricing = list(range(0, param_T_alpha + 1))
    times_swaps = list(range(param_T_alpha, param_T_beta + 1))
    # dtafns_fm = DTAFNS_forward_measure(Xt=p_initial_factors, kappa=param_kappa, theta=param_theta, sigma=param_sigma,
    #                                    rho=param_rho, T=param_T_alpha * stepsize,
    #                                    t=0, delta_t=stepsize, lam=param_lambda, seed=seed)
    # factorpath = dtafns_fm.generate_sample(sizes=npaths)

    DTAFNS_price = np.zeros((npaths, len(times_swaps)))
    DTAFNS_price_sr = np.zeros(len(times_swaps))

    for j, TT in enumerate(times_swaps):
        dtafns = DTAFNS_close(Xt=factorpath_Talpha, kappa=param_kappa, theta=param_theta, sigma=param_sigma,
                              rho=param_rho, T=TT * stepsize, t=param_T_alpha * stepsize, delta_t=stepsize,
                              lam=param_lambda)
        DTAFNS_price[:, j] = dtafns.price_zero_coupon()
        dtafns_sr = DTAFNS_close(Xt=p_initial_factors.reshape(1, -1), kappa=param_kappa, theta=param_theta,
                                 sigma=param_sigma, rho=param_rho, T=TT * stepsize, t=t_pricing * stepsize,
                                 delta_t=stepsize,
                                 lam=param_lambda)
        DTAFNS_price_sr[j] = dtafns_sr.price_zero_coupon()

    swap_rate = (DTAFNS_price_sr[0] - DTAFNS_price_sr[-1]) / (stepsize * np.sum(DTAFNS_price_sr[1:]))

    # dtafns2 = DTAFNS_close(Xt=p_initial_factors.reshape(1, -1), kappa=param_kappa, theta=param_theta,
    # sigma=param_sigma, rho=param_rho, T=param_T_alpha * stepsize, t=t_pricing * stepsize, delta_t=stepsize,
    # lam=param_lambda)
    DTAFNS_price_0 = DTAFNS_price_sr[0]
    payoff = 0
    if swaptype == 'payer':
        payoff = DTAFNS_price[:, 0] - DTAFNS_price[:, -1] - param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:],
                                                                                                 axis=1)
        payoff[payoff < 0] = 0
    elif swaptype == 'receiver':
        payoff = param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:], axis=1) - (
                DTAFNS_price[:, 0] - DTAFNS_price[:, -1])
        payoff[payoff < 0] = 0

    swaption_price = DTAFNS_price_0 * (p_notional_value * payoff).mean()
    return swaption_price, swap_rate, DTAFNS_price


if __name__ == '__main__':
    theseed = 2022
    dt = 1 / 12
    num_paths = 200000
    Times = np.linspace(1, 30, 30)
    lambda_param = 0.02332395
    theta_p = np.array([0, 0.03014341, 0.05050011])
    theta_q = np.array([0, 0.06326598, 0.07656804])
    rho = np.array([[1.0, -0.6303387, -0.4097114], [-0.6303387, 1.0, 0.2993069], [-0.4097114, 0.2993069, 1.0]])
    sigma_diag_vec = np.array([0.002680604, 0.004542753, 0.00701344])
    gamma = np.array([2.7923, 1.2016, 1.7167])
    kappa_p = np.array([[0.007484917, 0.0, 0.0], [0.0, 0.02878259, -0.02332395], [0, 0, 0.03536366]])
    kappa_q = np.array([[0, 0, 0], [0, 0.02332395, -0.02332395], [0, 0, 0.02332395]])
    # initial_factors = np.array([0.04911763, 0.03911763, 0.02911763])
    initial_factors = np.array([-0.03123863, 0.03842199, 0.06882331])
    isPmeasure = False

    if isPmeasure:
        kappa = kappa_p
        theta = theta_p
    else:
        kappa = kappa_q
        theta = theta_q

    sigma = np.diag(sigma_diag_vec)

    # Swaption Specification
    T_alpha = 60  # in month
    T_beta = T_alpha + 120  # in month

    fixed_rate = 0.02
    notional_value = 1.0

    start_time = time.time()

    swaption_price1, swap_rate1, zero_coupon_price1 = swap_swaption_simulation(
        kappa_q, theta_q, rho, sigma, lambda_param,
        initial_factors, T_alpha, T_beta,
        fixed_rate,
        notional_value, stepsize=dt,
        npaths=num_paths,
        swaptype='payer', seed=theseed)

    end_time = time.time()

    time_execute = end_time - start_time
    print(time_execute)
    print(swaption_price1, swap_rate1)

    pass
    # plt.plot(swap_price.T)
    # plt.show()
