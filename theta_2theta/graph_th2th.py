# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pylab as plt
import os
from tabulate import tabulate
# %load_ext autoreload
# %autoreload 2
import peakfit as pf

# # 2theta-omega scan
#
# - export data to '.csv' file

from two_theta_values import list_peak, w_K_alpha1_Cu


def annotates_peak(peaks, color='red', I_level=1e3, symbol=''):
    """Add line at peak position on the current figure"""
    for p in peaks:
        angle = p['2𝜃 (deg)']

        if angle > np.max(twth):
            break

        plt.axvline(x=angle, linewidth=1,
                    color=color, alpha=0.7)

        linename = p['hkl'].replace('(', '').replace(')', '')
        text_y_position = I_level+300*np.gcd.reduce(p['hkl_tuple'])
        plt.text(angle, text_y_position, linename,
                     rotation=0, color=color);


# +
filename = 'scan_th2th_ARB_145deg_10s.csv'

output_dir = 'output_'+filename.replace('.csv', '')

# Load data
data_directory = 'data/ARB_Erlangen/'
filepath = os.path.join(data_directory, filename)
data = np.genfromtxt(filepath,
                     skip_header=35,
                     delimiter=',')

twth = data[:, 0]
I = data[:, 1]
# -

# !mkdir {output_dir}

# ## Graph

# +
plt.figure(figsize=(14, 5))
plt.xlabel('two theta [deg]')
plt.ylabel('Intensity [cp]')

plt.semilogy(twth, I, 'k')
plt.xlim([twth.min(), twth.max()])
plt.title(filename)

# Add peak annotation:
material_symbol = 'Cu'
peaks = list_peak(material_symbol, w_K_alpha1_Cu, twoth_max=np.max(twth))
annotates_peak(peaks, color='darkorange', I_level=3e3, symbol=material_symbol)

material_symbol = 'Nb'
peaks = list_peak(material_symbol, w_K_alpha1_Cu, twoth_max=np.max(twth))
annotates_peak(peaks, color='gray', I_level=1e3, symbol=material_symbol)

plt.savefig(os.path.join(output_dir, 'th2th.svg'))
# -

peaks = list_peak('Cu', w_K_alpha1_Cu, twoth_max=np.max(twth))
peaks += list_peak('Nb', w_K_alpha1_Cu, twoth_max=np.max(twth))

print(tabulate(peaks, headers='keys'))


# ## Peak parameters estimation (fit)

def fit_peaks(peak_list):
    # !!  mutable 

    for peak in peak_list:
        peak_center = peak['2𝜃 (deg)']
        window_size = 6 # in twth unit (deg.)

        mask = np.logical_and(twth > peak_center - window_size/2,
                              twth < peak_center + window_size/2)

        x, y = twth[mask], I[mask]
        try:
            results, fit = pf.peakfit(x, y,
                                      pf.PseudoVoigt()
                                      )#pf.Gauss())
            peak['fit_param'] = results[0]
            peak['fit_function'] = (x, fit(x))
            
            pf.plot_results(x, y, results, fit,
                            save_path=output_dir,
                            save_name=peak['hkl'])

        except RuntimeError:
            print('fit error for', peak['hkl'])
            pf.plot_results(x, y,
                            save_path=output_dir,
                            save_name=peak['hkl'])



fit_peaks(peaks)

# +
plt.figure(figsize=(14, 5))
plt.xlabel('two theta [deg]')
plt.ylabel('Intensity [cp]')

plt.plot(twth, I, 'k', alpha=0.3)
plt.xlim([twth.min(), twth.max()])
plt.title(filename);

for peak in peaks:
    if not 'fit_function' in peak:
        continue
    x, y = peak['fit_function']
    plt.plot(x, y, '-r', linewidth=1)

plt.savefig(os.path.join(output_dir, 'th2th_with_fit.svg'))

# +
lmbda = w_K_alpha1_Cu  # wavelength, Angström
def DebyeScherrer(params):
    K = 0.9  #  dimensionless shape factor
    theta_rad = params['x0']/2 *np.pi/180
    beta_rad = params['fwhm'] *np.pi/180
    return K*lmbda/(beta_rad*np.cos(theta_rad))

def peak_summary(peaks):
    data_table = [{'hkl':p['hkl'],
                   'x0(deg)':p['fit_param']['x0'],
                   'ampl(cp)':p['fit_param']['amplitude'],
                   'fwhm(deg)':p['fit_param']['fwhm'],
                   'fwhm_std':p['fit_param']['fwhm_std'],
                   'DebyeScherrer(Å)':DebyeScherrer(p['fit_param'])
                  } if 'fit_param' in p
                  else {'hkl':p['hkl']} 
                  for p in peaks]
    return tabulate(data_table, headers='keys')
    


# -

print(peak_summary(peaks))

output_summary = os.path.join(output_dir, "peaks_summary.txt")
with open(output_summary, "w") as f:
    text = filename + '\n'
    text += peak_summary(peaks)
    f.write(text)


