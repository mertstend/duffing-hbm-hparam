import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# --------------------- Analytical Solution for Comparison --------------------
zeta = 0.05     # damping ratio
gamma = 0.1     # nonlinearity coefficient
P = 0.18        # forcing amplitude
a_min = .1      # minimum amplitude for analytical solution
a_max = 3       # maximum amplitude for analytical solution
a_ana = np.linspace(a_min, a_max, 50)  # amplitudes for analytical solution
# For each amplitude, determine the two associated excitation frequencies.
# Only the real-valued solutions exist.
Om_ana = np.zeros((len(a_ana), 2), dtype=complex)
for i, a in enumerate(a_ana):
    prefactor = 1 - zeta**2 / 2 + 3 * gamma * a**2 / 4
    inner = P**2 / a**2 + zeta**4 / 4 - zeta**2 - \
        3 * zeta**2 * gamma * a**2 / 4
    Om_ana[i, 0] = np.sqrt(prefactor + np.sqrt(inner))
    Om_ana[i, 1] = np.sqrt(prefactor - np.sqrt(inner))
valid_ana = (np.isreal(Om_ana[:, 0])) & (np.isreal(Om_ana[:, 1]))

# ---------------------- Load Reference and Test Results ----------------------
ref = pd.read_csv('./results/Duffing_reference_results.csv', header=None)
test = pd.read_csv('./results/Duffing_testing_results.csv', header=None)

diff = ref - test
max_diff = diff.abs().max().max()

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
tol = 1e-6
if max_diff < tol:
    print(GREEN + "Test passed: Results are within the tolerance limit.")
    print('Maximum difference between reference and test results: ' +
          str(max_diff) + RESET)
else:
    print(RED + "Test failed: Results exceed the tolerance limit.")
    print('Maximum difference between reference and test results: ' +
          str(max_diff) + RESET)

colorlist = ['#1D3557', '#E63946', '#00b695', "#457B9D"]
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(Om_ana[valid_ana, 0].real, a_ana[valid_ana], '-', color='#A8DADC',
        label='Analytical (H=1)')
ax.plot(Om_ana[valid_ana, 1].real, a_ana[valid_ana], '-', color='#A8DADC')
ax.plot(ref.iloc[0].to_numpy(), ref.iloc[1].to_numpy(), linestyle='-',
        color=colorlist[0], label='AFT (H=3)')
ax.plot(test.iloc[0].to_numpy(), test.iloc[1].to_numpy(), linestyle='',
        marker='.', color=colorlist[1], label='Neural Network (H=3)')
ax.set_xlim(0.4, 1.7)
ax.set_ylim(0, 3.0)
ax.set_xlabel(r'Excitation Frequency $\Omega$')
ax.set_ylabel(r'Amplitude $a$')
ax.legend(loc='upper left')
ax.grid()

axins = inset_axes(ax,
                   width="30%",
                   height="30%",
                   loc='center left',
                   bbox_to_anchor=(.71, .0, 1.1, 1.2),
                   bbox_transform=ax.transAxes,
                   borderpad=0)
axins.plot(Om_ana[valid_ana, 0].real, a_ana[valid_ana], '-', color='#A8DADC')
axins.plot(Om_ana[valid_ana, 1].real, a_ana[valid_ana], '-', color='#A8DADC')
axins.plot(ref.iloc[0].to_numpy(), ref.iloc[1].to_numpy(),
           linestyle='-', color=colorlist[0])
axins.plot(test.iloc[0].to_numpy(), test.iloc[1].to_numpy(),
           linestyle='', marker='.', color=colorlist[1])
axins.set_xlim(1.24, 1.285)
axins.set_ylim(2.7, 2.95)
axins.grid()
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")

plt.savefig('./figures/duffing_analytical_aft_nn.svg', bbox_inches='tight')
plt.show()
