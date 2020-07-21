# -*- coding: utf-8 -*-

import os, re
import numpy as np
import matplotlib.pylab as plt


import cristallo as cr

# # Figure de pôle
#
# - export (convert) data as a single .csv file

# ## Load the data

# ====================
#  Pole Figure Graph
# ====================

def polar_axis(fig=None, figsize=(6, 6)):
    """Create polar axis

    - uses stereographic projection

    Parameters
    ----------
    fig : optional
        matplotlib figure object, by default None
    figsize : tuple, optional
        size of teh figure, by default (6, 6)

    Returns
    -------
    fig, ax
    """

    if not fig:
        fig = plt.figure(figsize=figsize);

    # Define polar axes
    ax = fig.add_subplot(111, projection='polar');
    ax.set_rmin(0)
    ax.set_rorigin(-0.01)
    ax.set_ylim((0, 2))

    psi_ticks_deg = np.array( [15, 30, 45, 60, 75] )
    psi_ticks_stereo = 2*np.tan(psi_ticks_deg*np.pi/180 /2)
    ax.set_yticks(psi_ticks_stereo)
    psi_ticks_label = [f'{str(int(v))}°' for v in psi_ticks_deg]
    ax.set_yticklabels(psi_ticks_label, alpha=0.4, color='black')
    ax.grid(True, alpha=0.4, color='black')
    return fig, ax


def plot_direction(ax, phi_deg, psi_deg, 
                   color='black', marker='d', markersize=3,
                   label=None, label_position='right', text_bold=False):
    """Plot a point for the corresponding direction"""
    phi_rad = phi_deg*np.pi/180 + 0.001
    psi_stereo = 2*np.tan(psi_deg*np.pi/180 /2)  #  Stereographic projection
    psi_stereo_annotate = psi_stereo if psi_stereo > 0.1 else 0.1   # Bug... ?

    if label:
        va = 'center' if label_position == 'right' else 'baseline'
        ha = 'center' if label_position == 'center' else 'left'
        weight = 'bold 'if text_bold else 'normal'

        ax.annotate(label, (phi_rad, psi_stereo_annotate),
                    textcoords='offset points', xytext=(0, 5),
                    rotation=0, alpha=0.9, color=color, family='sans-serif',
                    horizontalalignment=ha, va=va, weight=weight)

    ax.plot(phi_rad, psi_stereo, marker, color=color, markersize=markersize)


def hkl_tuple_to_str(hkl):
    return ''.join(str(u) for u in hkl)


def list_eq_directions(hkl_figure, phi0, n):
    """[summary]

    upper hemisphere only

    Parameters
    ----------
    hkl_figure : [type]
        [description]
    phi0 : [type]
        [description]
    n : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    eq_directions = []
    for hkl_eq in cr.equivalent_directions(hkl_figure):
        phi, psi = cr.phi_psi_angles(hkl_eq, phi0, n)
        if psi <= 90:
            d = {'hkl': hkl_tuple_to_str(hkl_eq),
                 'phi': phi,
                 'psi': psi}
            eq_directions.append(d)

    return eq_directions



# ==================
#  Import csv data
# ==================

def get_field(csvpath, fieldname, to_float=True):
    '''Extract information from csv file
    (path to the file) for the asked fieldname
    '''

    with open(csvpath, 'r') as f:
        csv = f.readlines()

    line = next(line for line in csv if line.startswith(fieldname))
    line = line.strip().split(',')[1:]

    if to_float:
        return [float(u) for u in line]
    else:
        return line


def open_range(start, stop, step):
    """custom `arange` function to include the last elt."""
    return np.arange(start, stop+step/2, step)


def read_polefig_csv(file_path):

    # Try to get the number of header line:
    with open(file_path, 'r') as f:
        csv = f.readlines()

    for k, line in enumerate(csv):
        if line.startswith('Time per step'):
            break

    # Load data
    data = np.genfromtxt(file_path,
                         skip_header=k+1, delimiter=',')

    # Define axis range
    psi_range = get_field(file_path, 'Psi range')
    phi_range = get_field(file_path, 'Phi range')

    psi_span_deg = open_range(*psi_range)
    phi_span_deg = open_range(*phi_range)

    # duplicate first line at the end
    #  used to close white gap in polar graph
    data = np.vstack([data, data[0, :]]).T

    return phi_span_deg, psi_span_deg, data

