import numpy as np
import pickle
from DTAFNSModels import generate_path
from DTAFNS_swaption_pricing_forward import swap_swaption_simulation
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# === Configuration ===
np.random.seed(2022)
dt = 1 / 12
num_paths = 500000  # 🟩 this was 100000
N = 20000
delta = 0.01
T_alpha = 60
tenor_length = 120
T_beta = T_alpha + tenor_length

lambda_param = 0.02332395
theta_p = np.array([0, 0.03014341, 0.05050011])
theta_q = np.array([0, 0.06326598, 0.07656804])
rho = np.array([
    [1.0, -0.6303387, -0.4097114],
    [-0.6303387, 1.0, 0.2993069],
    [-0.4097114, 0.2993069, 1.0]
])
sigma_diag_vec = np.array([0.002680604, 0.004542753, 0.00701344])
sigma = np.diag(sigma_diag_vec)

kappa_p = np.array([
    [0.007484917, 0.0, 0.0],
    [0.0, 0.02878259, -0.02332395],
    [0, 0, 0.03536366]
])
kappa_q = np.array([
    [0, 0, 0],
    [0, 0.02332395, -0.02332395],
    [0, 0, 0.02332395]
])

initial_factors = np.array([-0.03123863, 0.03842199, 0.06882331])
fixed_rate = 0.025083
notional_value = 1.0

# === Generate base factors and Talphas ===
xpath, _ = generate_path(kappa_p, theta_p, rho, sigma, initial_factors, T_beta, N, theseed=None)
samples = xpath.reshape(N * (T_beta + 1), 3)
sample_indices = np.random.choice(samples.shape[0], size=N, replace=False)
base_factors = samples[sample_indices]

Talphas = np.random.randint(0, T_alpha + 2, size=N)  # 🟩 This was Talphas = np.random.randint(1, T_alpha + 1, size=N)

# === Sensitivity bumping: original + plus/minus delta on each factor ===
factors_list = [base_factors]
Talpha_list = [Talphas]

for col in range(3):
    for sign in [1, -1]:
        bumped = base_factors.copy()
        bumped[:, col] += sign * delta
        factors_list.append(bumped)
        Talpha_list.append(Talphas)

factors_combined = np.vstack(factors_list)
Talphas_combined = np.concatenate(Talpha_list)


# === Pricing ===

def price_swaption(factor, talpha):
    return swap_swaption_simulation(
        kappa_q, theta_q, rho, sigma, lambda_param,
        factor, talpha, talpha + tenor_length,
        fixed_rate, notional_value, stepsize=dt, npaths=num_paths,
        swaptype='payer', seed=2022
    )[0]


if __name__ == '__main__':
    from multiprocessing import Pool

    print("Pricing swaptions in parallel...")
    with Pool() as pool:
        swaption_price = np.array(
            list(pool.starmap(price_swaption, zip(factors_combined, Talphas_combined)))
        )

    data = {
        'factors': factors_combined,
        'Talphas': Talphas_combined,
        'swaption_price': swaption_price
    }
    path_data = os.path.join(ROOT,"..", "data", "samples_to_trainNN_al60_be180_k25083_forgrad.pkl")
    with open(path_data, 'wb') as f:
        pickle.dump(data, f)

    print("Samples Saved.")
