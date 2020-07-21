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
#     display_name: py3 venv
#     language: python
#     name: py3
# ---

# +
# list two-theta angle values 
# for a given materials and wavelength

# note: do not work yet for non-cubic material
# note: export to  .py file using jupytext light-script pairing
# -

import numpy as np
from scipy import constants
import tabulate as tab

# +
# Latttice parameter in Angström
# and crystal structure

materials = {
    'Cu':   (3.615, 'fcc'),
    'Nb':   (3.3063, 'bcc'),
    'Si':   (5.431194, 'fcc'),
    'LaB6': (4.15682, 'fcc'), # https://www-s.nist.gov/srmors/certificates/660c.pdf
    'Al':   (4.046, 'fcc'),
    'Ag':   (4.079, 'fcc')
}
# source:
# https://en.wikipedia.org/wiki/Lattice_constant
# -

def keV_to_Angstrom(E_keV):
    """
    # E = hc/lambda
    # lambda = hc/E
    """
    E_J = 1e3 * E_keV * constants.e
    lambda_m = constants.c * constants.h / E_J
    return lambda_m * 1e10


# +
def twoTheta_Bragg_law(distance, wavelength):
    u = wavelength/2/distance
    if np.abs(u) <= 1:
        return 2* np.arcsin(u) * 180/np.pi
    else:
        return np.NaN
    
def d_hkl_cubic(a, h, k, l):
    return a/np.sqrt( h**2 + k**2 + l**2 )


# +
def all_hkl(n_max=7):
    """Generate all possible hlk triplet"""
    all_hkl = []
    for l in range(1, n_max):  #  <-- indice max.
        for k in range(l+1):
            for h in range(k+1):
                all_hkl.append(sorted([h, k, l], reverse=True))

    all_hkl = sorted(all_hkl, key=lambda x: sum(i**2 for i in x))
    return all_hkl

# Condition d'existences:
def existence_FCC(h, k, l):
    # même parité
    return h % 2 == k % 2 and k % 2 == l % 2

def existence_BCC(h, k, l):
    # somme paire
    return (h+k+l) % 2 == 0

def existence_ZincBlende(h, k, l):
    # même parité
    meme_parite = (h % 2 == k % 2 and k % 2 == l % 2)
    motif = ((h+k+l-2) % 4 != 0)
    return meme_parite and motif


# -

hkl_list_per_struct = {
    'fcc':[ hkl for hkl in all_hkl(n_max=6) if existence_FCC(*hkl) ],
    'bcc':[ hkl for hkl in all_hkl(n_max=6) if existence_BCC(*hkl) ],
}


def list_peak(mat_symbol, wavelength, twoth_max=360):
    mat = materials[mat_symbol]
    data = []
    for hkl in hkl_list_per_struct[mat[1]]:
        d = d_hkl_cubic(mat[0], *hkl)
        two_theta = twoTheta_Bragg_law(d, wavelength)
        
        if np.isnan( two_theta ):
            break
        if two_theta > twoth_max:
            continue
            
        peak_info = {'hkl':f"({''.join(str(u) for u in hkl)})" + mat_symbol,
                    '2𝜃 (deg)':two_theta,
                    'd (A)':d,
                    'hkl_tuple':hkl}
        data.append(peak_info)
    return data


def peak_table(mat_symbol, wavelength):
    material = materials[mat_symbol]
    hkl_list = hkl_list_per_struct[material[-1].lower()]
    a = material[0] # if cubic...
    tab_data = list_peak(mat_symbol, wavelength)

    text = f'{mat_symbol}, {material[-1].lower()}, a={material[0]:.3f}A, w={wavelength:.3f}A \n'
    return text + tab.tabulate(tab_data, headers="keys")


w_K_alpha1_Cu = 1.540598  # Angström
w_K_alpha2_Cu = 1.544426  # Angström

E_keV = 20# 18.777 # keV
w = keV_to_Angstrom(E_keV)
#print('wavelength:', w, 'Angstöm')

table = peak_table('Nb', w_K_alpha1_Cu)
#print(table)


