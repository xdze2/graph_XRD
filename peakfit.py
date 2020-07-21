# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# # Fit Peak

# +
# How it works:
#
#    function a -> parameters are [(a1, a2), ]
#    function b -> parameters are [(b1, b2, b3), ]
#
#    note the 1-element list
#    it allows to simply write `param_a+b = param_a + param_b`
#    when two functions are added
#
#    Sum(f_a, f_b) -> parameters are [(a1, a2), (b1, b2, b3)]
#
#    `estimate_param` returns a flattened version of the parameter
#    (a1, a2, b1, b2 b3)
#
# -

class Linear:
    """Linear function"""
    def __init__(self, slope=None, intercept=None):
        self.slope_init = slope
        self.intercept_init = intercept
        self.param_names = [('slope', 'intercept'), ]
        self.name = ['Linear', ]

    def __call__(self, x, slope, intercept):
        return x*slope + intercept

    def estimate_param(self, x, y):
        if not self.slope_init:
            self.slope_init = (y[-1] - y[0])/(x[-1] - x[0])
        if not self.intercept_init:
            self.intercept_init = y[0] - self.slope_init*x[0]

        return self.slope_init, self.intercept_init


class Gauss:
    """Gaussian function"""
    def __init__(self, x0=None, fwhm=None, ampl=None):
        self.x0_init = x0
        self.fwhm_init = fwhm
        self.amplitude_init = ampl
        self.param_names = [('x0', 'fwhm', 'amplitude'), ]
        self.name = ['Gaussian', ]

    def __call__(self, x, x0, fwhm, amplitude):
        sigma = fwhm /( 2*np.sqrt(2*np.log(2)) )
        return amplitude * np.exp( -(x-x0)**2/(2*sigma**2) )

    def estimate_param(self, x, y):
        if not self.x0_init:
            self.x0_init = x[np.argmax(y)]
        if not self.fwhm_init:
            self.fwhm_init = np.ptp( x[ y  > (y.min() + y.max())/2 ] )
        if not self.amplitude_init:
            self.amplitude_init = np.ptp(y)

        return self.x0_init, self.fwhm_init, self.amplitude_init


class Lorentzian:
    """Lorentzian function (or Cauchy distribution)

       I = 1/( 1 + x^2 )
    """
    def __init__(self, x0=None, fwhm=None, ampl=None):
        self.x0_init = x0
        self.fwhm_init = fwhm
        self.amplitude_init = ampl
        self.param_names = [('x0', 'fwhm', 'amplitude'), ]
        self.name = ['Lorentzian', ]

    def __call__(self, x, x0, fwhm, amplitude):
        hwhm = fwhm / 2
        u = x - x0
        return amplitude/( 1 + (u/hwhm)**2 )

    def estimate_param(self, x, y):
        if not self.x0_init:
            self.x0_init = x[np.argmax(y)]
        if not self.fwhm_init:
            self.fwhm_init = np.ptp( x[ y  > (y.min() + y.max())/2 ] )
        if not self.amplitude_init:
            self.amplitude_init = np.ptp(y)

        return self.x0_init, self.fwhm_init, self.amplitude_init


class PseudoVoigt:
    """PseudoVoigt function

    approximation of the Voigt function
    weighted sum of Gaussian and Lorentzian function

        PV(x) = eta*G(x) + (1-eta)*L(x)

    # see:
    # https://docs.mantidproject.org/nightly/fitting/fitfunctions/PseudoVoigt.html
    """
    def __init__(self, x0=None, fwhm=None, ampl=None, eta=None):
        self.x0_init = x0
        self.fwhm_init = fwhm
        self.amplitude_init = ampl
        self.eta_init = eta
        self.param_names = [('x0', 'fwhm', 'amplitude', 'eta'), ]
        self.name = ['PseudoVoigt', ]

    def __call__(self, x, x0, fwhm, amplitude, eta):
        hwhm = fwhm / 2
        u = x - x0

        L = hwhm/(u**2 + hwhm**2)/np.pi

        sigma = hwhm / np.sqrt(2*np.log(2))
        norm_G = 1/(sigma * np.sqrt(2*np.pi))
        G = norm_G * np.exp( -(x-x0)**2/(2*sigma**2) )

        I = amplitude*np.pi*hwhm/(1 + eta*(np.sqrt(np.pi*np.log(2)) - 1))
        return ( eta*G + (1 - eta)*L ) * I

    def estimate_param(self, x, y):
        if not self.x0_init:
            self.x0_init = x[np.argmax(y)]
        if not self.fwhm_init:
            self.fwhm_init = np.ptp( x[ y  > (y.min() + y.max())/2 ] )
        if not self.amplitude_init:
            self.amplitude_init = np.ptp(y)
        if not self.eta_init:
            self.eta_init = 0.5

        return self.x0_init, self.fwhm_init, \
                self.amplitude_init, self.eta_init


class Sum:
    """Build a new function as the sum of two functions"""
    def __init__(self, a, b):
        self.a = a
        self.b = b

        # param are list of list
        self.param_names = a.param_names + b.param_names
        self.name = (*a.name, *b.name)

    def __call__(self, x, *p):
        nargs_a = sum(len(u) for u in self.a.param_names)
        p_a = p[:nargs_a]
        p_b = p[nargs_a:]
        return self.a(x, *p_a) + self.b(x, *p_b)

    def estimate_param(self, x, y):
        p_a = self.a.estimate_param(x, y)
        p_b = self.b.estimate_param(x, y)
        return (*p_a, *p_b)


def peakfit(x, y, function=Gauss(), background=Linear()):
    """Fit the data (x, y) using the provided function

       The background function is summed to the function
       - Default function is `Gauss()`
       - Default background is `Linear()`
       - Set to `̀None` if no background is wanted.

       Returns:
       - list of dictionary parameters (one for each function)
       - global fit function with optimal parameters
    """

    if background is not None:
        function = Sum(function, background)

    p0 = function.estimate_param(x, y)

    popt, pcov = curve_fit(function, x, y, p0)
    parameter_err = np.sqrt(np.diag(pcov))

    result = []
    idx = 0
    for f_name, params in zip(function.name, function.param_names):
        res = {'function':f_name}
        for param in params:
            res[param] = popt[idx]
            res[param + '_std'] = parameter_err[idx]
            idx += 1

        result.append(res)

    return result, lambda x:function(x, *popt)


def results_summary(results):
    """Generate text summary of the parameters"""
    max_length = max(len(k) for r in results for k in r.keys()
                     if k != 'function' and 'std' not in k) + 1
    summary = ''
    for r in results:
        summary += r['function'] + '\n'
        for name, value in r.items():
            if name == 'function' or 'std' in name:
                continue
            summary += f" {name+':': <{max_length}}{value: 0.3f}\n"
    return summary


def plot_results(x, y,
                 results=None, fit=None,
                 save_path=None, save_name=None):
    """Generate summary graph of fit results"""
    fig, ax = plt.subplots()
    ax.plot(x, y, '.k', label='data')
    if fit:
        ax.plot(x, fit(x), 'r-', label='fit')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    if results:
        ax.text(0.01, 0.99, results_summary(results),
                transform=ax.transAxes,
                fontfamily='monospace',
                verticalalignment='top')
    if save_name:
        ax.set_title(save_name)

    if save_path:
        if save_name:
            figname = save_name+'.png'
        else:
            hashvalues = list(f"{k}_{str(v)[:5].replace('.', 'p')}"
                              for r in results for k, v in r.items()
                              if k != 'function' and 'std' not in k)
            figname = '_'.join(hashvalues[:2]) + '.png'

        path = os.path.join(save_path, figname)
        fig.savefig(path)
        
    return fig
