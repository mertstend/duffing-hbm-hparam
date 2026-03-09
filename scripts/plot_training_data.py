import numpy as np
from src.plotting import (training_data_distribution_violinplot,
                          training_data_distribution_histogram,
                          frc_and_training_inputs_3d_scatter_plot)


data_load_date = '2026-02-18_14-04-47'
data = np.load('data/duffing_train_data_H3_N64_'+data_load_date+'.npz')
q_data = data['q_coeffs']
fnl_data = data['fnl_coeffs']

training_data_distribution_violinplot(q_data, fnl_data)

training_data_distribution_histogram(q_data, fnl_data)

q_rel = np.loadtxt('data/nn_input_Duffing.txt', delimiter=',')

frc_and_training_inputs_3d_scatter_plot(q_rel, q_data,
                                        figure_name='duffing_training_samples')
