import itertools as it
import numpy as np


# =====================
#  3D Vector geometry
# =====================

nbr_digit = 8  # used to round results

def cross_product(a, b):
    ab = (a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0])
    return np.asarray(ab)


def dot_product(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.round(np.sum(a*b), decimals=nbr_digit)


def norm(a):
    """returns vector length"""
    return np.sqrt(dot_product(a, a))


def norm_it(a):
    """returns same direction unit vector"""
    a = np.asarray(a)
    return np.copy(a) / norm(a)


def angle(a, b):
    '''Angle between vector a and b in degree'''
    cos_ab = dot_product(a, b)/(norm(a)*norm(b))
    cos_ab = np.round(cos_ab, decimals=nbr_digit)
    return np.arccos(cos_ab) * 180/np.pi


def plan_projection(a, n):
    ''' Project the vector `a` in the plan of normal `n`
    '''
    n = norm_it(n)
    scale_n = dot_product(a, n)
    a_plan = tuple(ai - scale_n*ni for ai, ni in zip(a, n))
    return round_it(a_plan)


def round_it(a):
    return tuple(round(ai, nbr_digit) for ai in a)


# =================================
#  Crystallography (cubic system)
# =================================

def equivalent_directions(hkl):
    '''List the equivalent direction
        for a cubic lattice

        hkl: tuple of Miller indices
        returns a set of hkl tuple
    '''
    perms = it.permutations(hkl)
    sign = it.product([1, -1], [-1, 1], [-1, 1])
    dirs = it.product(perms, sign)
    dirs = {tuple(u*s for u, s in zip(xys, signs))
            for xys, signs in dirs}
    return dirs


def phi_psi_angles(u, phi0, n):
    ''' Convert the u direction (given as hkl indices)
        to phi & psi angles (pole figure axes)

    PANalytical notations
    =====================
    XYZ basis of the sample cradle:
    when the sample cradle position is (psi=0°, psi=0°, omega=0°)
        - X is positive towards the left (incident beam direction)
        - Y is positive downward (vertical)
        - Z is positive towards you

    phi & psi angles:
        - phi, rotation around Z, negative X->Z i.e. not right-handed
        - psi, rotation around X, positive Y->Z i.e. right -handed

    Crystal basis is inverse of the sample cradle basis:
    (it is the "natural" sample basis)
        - a = -X = phi0 i.e. positive towards the X-ray source
        - b = -Y  i.e. upward
        - c = +Z = n  i.e. normal to the sample surface

    Returns
    =======
    phi0: hkl indices corresponding to (phi=0°, psi=90°) direction
        (-X axis on the sample stage, or a in crystal basis)
    n: hkl indices corresponding to the psi=0° direction
        (sample surface normal, or +Z)

    both phi and psi angles are in degree

    Examples
    ========
    phi_psi_angles((1, 1, 0), (1, 0, 0), (0, 0, 1))
    >>> (45.0, 90.0)
    phi_psi_angles((0, 0, 1), (1, 0, 0), (0, 0, 1))
    >>> (0.0, 0.0)
    phi_psi_angles((1, 0, 1), (1, 0, 0), (0, 0, 1))
    >>> (0.0, 45.00000009614413)
    '''
    n = norm_it(n)
    phi0 = norm_it(phi0)

    if dot_product(n, phi0) != 0:
        raise NameError('phi0 not perpendicular to n')

    phi90 = cross_product(n, phi0)
    psi = angle(u, n)

    # degenerate cases for phi:
    if abs(psi) < 1e-4 or abs(psi-180) < 1e-4:
        return 0.0, psi

    u_plan = norm_it(plan_projection(u, n))

    cos_phi = dot_product(phi0, u_plan)
    sin_phi = dot_product(phi90, u_plan)

    phi = np.arctan2(sin_phi, cos_phi)*180/np.pi

    return phi, psi


def rodrigues_rotation(v, k_axe, theta):
    ''' Generic rotation using the Rodrigues' rotation formula
         https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        v: the vector to rotate
        k: the axis of rotation (normed before computation)
        theta: angle of rotation in radian
    '''
    k = k_axe
    
    v, k = np.asarray(v), np.asarray(k)
    k = k / np.linalg.norm(k)

    return v * np.cos(theta) + np.cross(k, v)*np.sin(theta) \
            + k * np.inner(k, v)*(1 - np.cos(theta))

# test
# rodrigues_rotation([1, 0, 0], [1, 1, 1], np.pi/180*120)