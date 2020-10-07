# -*- coding: utf-8 -*-

import os, re
import numpy as np
import matplotlib.pylab as plt


import cristallo as cr

# # Figure de pôle
# - export (convert) data as a single .csv file


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


def plot_polefigure(phi_deg, psi_deg, intensity,
                    title=None, show_max=True, unit='cps',
                    figsize=(6, 6),
                    cmap='YlOrRd'):
    """polar heat-map graph of intensity"""
    
    # Change unit for the axis
    phi_rad, psi_stereo = stereographic_projection(phi_deg, psi_deg)

    # Create the figure and axis
    fig, ax = polar_axis(figsize=figsize)
    m = ax.pcolormesh(phi_rad, psi_stereo, intensity, cmap=cmap)
    ax.grid(True, alpha=0.4, color='black')
    cbar = fig.colorbar(m, shrink=0.7, pad=0.07, aspect=35)
    if unit:
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(unit, rotation=270)
    
    if show_max:
        ax.text(0.01, 0.01, f'I_max={np.max(intensity):.0f}{unit}',
                    transform=ax.transAxes,
                    fontfamily='monospace',
                    verticalalignment='top')

    if title:
        ax.set_title(title);

    return fig, ax


# =====================
#  Annotation methods
# =====================

def plot_direction(ax, phi_deg, psi_deg, 
                   color='black', marker='d', markersize=3,
                   label=None, label_position='right', weight='normal'):
    """Plot a point for the corresponding (phi, psi) direction"""
    phi_rad = phi_deg*np.pi/180 + 0.001
    psi_stereo = 2*np.tan(psi_deg*np.pi/180 /2)  #  Stereographic projection
    psi_stereo_annotate = psi_stereo if psi_stereo > 0.1 else 0.1   # Bug... ?

    if label:
        va = 'center' if label_position == 'right' else 'baseline'
        ha = 'center' if label_position == 'center' else 'left'
        
        ax.annotate(label, (phi_rad, psi_stereo_annotate),
                    textcoords='offset points', xytext=(0, 5),
                    rotation=0, alpha=0.9, color=color, family='sans-serif',
                    horizontalalignment=ha, va=va, weight=weight)

    ax.plot(phi_rad, psi_stereo, marker, color=color, markersize=markersize)


def plot_many_directions(ax, angles, 
                         color='black', marker='d', markersize=3, 
                         label=None, label_position='right', weight='normal'):
    """Loop around `plot_direction`

    Parameters
    ----------
    ax : figure axes object
    angles : list of dictionary {'phi':..., 'psi':..., 'hkl':...}
        list of directions
    color : str, by default 'black'
    marker : str, by default 'd'
    markersize : int, by default 3
    label : str,  by default None
    label_position : str, by default 'right'
    weight : bool, by default normal
    """
    for d in angles:
        if d['psi']>90:
            continue
        
        if label == True:
            text = d['hkl']
        elif isinstance(label, str):
            text = label
        else:
            text = None
            
        plot_direction(ax, d['phi'], d['psi'], label=text,
                       color=color, marker=marker, markersize=markersize,
                       label_position=label_position, weight=weight)


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
        #if psi <= 90:
        d = {'hkl': hkl_tuple_to_str(hkl_eq),
                'phi': phi,
                'psi': psi}
        eq_directions.append(d)

    return eq_directions


def stereographic_projection(phi_degree, psi_degree):
    """stereographic projection for the PSI angle
        and conversion to radian for the phi angle
        i.e. unit for the matplotlib polar axis
    """
    psi_rad = psi_degree *np.pi/180
    psi_stereo = 2*np.tan(psi_rad/2)

    phi_rad = phi_degree *np.pi/180
    return phi_rad, psi_stereo


# ==================
#  Import csv data
# ==================

def get_field(csvpath, fieldname, to_float=True):
    '''Extract information from csv file
    (path to the file) for the asked fieldname

    returns a list
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


def read_polefig_csv(file_path, norm_intensity=True):
    """Import csv file

    Parameters
    ----------
    file_path : string
        [description]
    norm_intensity : bool, default True
        search for "Time per step" value in the csv file (in seconds)
        and norm intensity value to obtain count per second values (cps) 

    Returns
    -------
    [type]
        [description]
    """    
    # Try to get the number of header line:
    with open(file_path, 'r') as f:
        csv = f.readlines()

    for k, line in enumerate(csv):
        if line.startswith('Time per step'):
            break

    # Load data
    intensity = np.genfromtxt(file_path,
                         skip_header=k+1, delimiter=',')

    # Define axis range
    psi_range = get_field(file_path, 'Psi range')
    phi_range = get_field(file_path, 'Phi range')

    psi_span_deg = open_range(*psi_range)
    phi_span_deg = open_range(*phi_range)

    # duplicate first line at the end
    #  used to close white gap in polar graph
    intensity = np.vstack([intensity, intensity[0, :]]).T

    # norm intensity  counts--> counts per seconds
    if norm_intensity:
        time_per_step =  get_field(file_path, 'Time per step')[0]
        intensity = intensity / np.float(time_per_step)

    return phi_span_deg, psi_span_deg, intensity

