import numpy as np
import matplotlib.pyplot as plt
from src.aft import compute_AFT_solution
from src.fourier_conversion import (convert_cossin_to_comexp,
                                    convert_comexp_to_cossin)
from src.nn_inference import evaluate_Duffing_nn_H3
from src.error_metrics import compute_error_metrics


###############################################################################
# Performance on test data
###############################################################################
test_data_id = '2026-02-18_14-04-47'
test_data = np.load('data/duffing_test_data_H3_N64_' + test_data_id + '.npz')
q_test = test_data['q_coeffs']
fnl_test_aft = test_data['fnl_coeffs']

nn_id = '2026-02-18_13-29-30'
fnl_test_nn = []
for i in range(len(q_test)):
    fnl_test_nn.append(evaluate_Duffing_nn_H3(nn_id, q_test[i, :]))
fnl_test_nn = np.array(fnl_test_nn)
global_metrics_test, individual_metrics_test = \
    compute_error_metrics(fnl_test_aft, fnl_test_nn)
plt.bar(global_metrics_test.keys(),
        [global_metrics_test[k] for k in global_metrics_test.keys()])
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.title(f'Global error metrics on test data for dataset {nn_id}')
plt.tight_layout()
plt.show()

global_metrics_test, individual_metrics_test = \
    compute_error_metrics(fnl_test_aft, fnl_test_nn, normalize=True)
plt.bar(global_metrics_test.keys(),
        [global_metrics_test[k] for k in global_metrics_test.keys()])
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.title(f'Global error metrics on test data for dataset {nn_id}')
plt.tight_layout()
plt.show()

###############################################################################
# Performance on FRC trajectory
###############################################################################
q_frc_full = np.loadtxt('data/nn_input_Duffing.txt', delimiter=',')
q_rel = np.concatenate([q_frc_full[:, 1:3], q_frc_full[:, 5:7]], axis=1)

H = 3
N = 2**6
gamma = 0.1
fnl_rel_aft = np.empty((0, 4))
fnl_rel_nn = np.empty((0, 4))
for i in range(np.shape(q_frc_full)[0]):
    q_ce = convert_cossin_to_comexp(q_frc_full[i, :7])
    fnl_ce = compute_AFT_solution(N, H, q_ce, gamma)
    fnl_cs = convert_comexp_to_cossin(fnl_ce, H)
    fnl_rel_aft = np.vstack([fnl_rel_aft, fnl_cs[[1, 2, 5, 6]]])

    fnl_cs_NN = evaluate_Duffing_nn_H3(nn_id, q_rel[i])
    fnl_rel_nn = np.vstack([fnl_rel_nn, fnl_cs_NN])

global_metrics, individual_metrics = \
    compute_error_metrics(fnl_rel_aft, fnl_rel_nn)
plt.bar(global_metrics.keys(),
        [global_metrics[k] for k in global_metrics.keys()])
plt.xticks(rotation=45, ha='right')
plt.show()

colors = ['#1D3557', '#008b9a', '#f19699', '#e63946']
q_labels = ['A1', 'B1', 'A3', 'B3']
fnl_labels = ['A1', 'B1', 'A3', 'B3']

# NN prediction vs. AFT ground-truth over test samples and FRC inputs
fig, ax = plt.subplots(1, 4, figsize=(15, 4))
for i in range(4):
    ax[i].plot(fnl_test_aft[:, i], fnl_test_nn[:, i], 'o', label=fnl_labels[i],
               color='#A8DADC')
    ax[i].plot(fnl_rel_aft[:, i], fnl_rel_nn[:, i], 'x',
               label=f'{fnl_labels[i]} (rel)', color='#E63946')
    ax[i].set_xlabel(f'{fnl_labels[i]} true')
    ax[i].set_ylabel(f'{fnl_labels[i]} predicted')
    ax[i].legend()
plt.suptitle(f'ground truth vs. prediction for dataset {nn_id}')
plt.tight_layout()


# AFT vs. NN over FRC iterations
fig, ax = plt.subplots(4, 1, figsize=(5, 8))
for i in range(4):
    ax[0].plot(q_rel[:, i], label=q_labels[i], color=colors[i])
    ax[1].plot(fnl_rel_aft[:, i], label=fnl_labels[i],
               color=colors[i])
    ax[2].plot(fnl_rel_nn[:, i], label=f'{fnl_labels[i]}_NN', color=colors[i])
    ax[3].plot(fnl_rel_aft[:, i] - fnl_rel_nn[:, i],
               label=f'{fnl_labels[i]} error', color=colors[i])
ax[0].legend()
ax[0].set_title('Input coefficients over iterations')
ax[1].legend()
ax[1].set_title('AFT output coefficients over iterations')
ax[2].legend()
ax[2].set_title('NN output coefficients over iterations')
ax[3].legend()
ax[3].set_title('Error between AFT and NN outputs over iterations')
plt.tight_layout()

plt.show()
