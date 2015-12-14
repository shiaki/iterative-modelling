#!/usr/bin/python

import numpy as np
np.seterr(divide = 'ignore')

import scipy.optimize as    opt
import scipy.interpolate as ipt
import scipy.integrate as   itg

if "DEBUG" in locals():
  import matplotlib.pyplot as plt
  from matplotlib.colors import LogNorm
  import time

def Make_exp_disk(N_pcs, h_R, h_Z, R_cut, Z_cut):

  # Generate Z
  rds = np.random.random(N_pcs) * (1. - np.exp(-Z_cut / h_Z))
  sgn = np.where(np.random.random(N_pcs) < 0.5, 1., -1.)
  Z   = -h_Z * np.log(1. - rds) * sgn

  # Generate radial distribution
  eqs  = lambda K, R: h_R * (1. - K) - np.exp(-R / h_R) * (h_R + R)
  R_pt = np.linspace(0., R_cut, 1024)  # <== Number of interp points
  K_pt = np.zeros(shape = (R_pt.size,))

  for I_t, R_t in enumerate(R_pt):
    K_pt[I_t] = opt.brentq(eqs, 0., 1., R_t)

  # construct interpolator and do intp.
  R_k = ipt.interp1d(K_pt, R_pt, kind = 'linear', assume_sorted = True)
  R   = R_k(np.random.random(N_pcs) * K_pt[-1])

  # Generate phi angle
  phi = 2. * np.pi * np.random.random(N_pcs)
  cos_phi, sin_phi = np.cos(phi), np.sin(phi)

  # cylindrical -> cartesian
  X = R * cos_phi; Y = R * sin_phi

  # return array
  return np.vstack((X, Y, Z)).T

'''
  rho(m) = rho0 * (m / ac)^n * exp(-(m / mc)^2)
'''
def Make_pow_bulge(N_pcs, ac, mc, n, q, m_cut):

  intfc = lambda m, aci, mci, ni: np.power(m / aci, -ni) * \
                                  np.exp(-(m / mci) ** 2) * (m ** 2)
  #                                                         |metric|

  # M for mass, m for "radius"

  # Total mass and half-mass radius (with bsch)
  Mt = itg.quad(intfc, 0., m_cut, args = (ac, mc, n))[0]
  itv_a, itv_b = 0., m_cut
  while(itv_b - itv_a > 0.01):
    itv_M = itg.quad(intfc, 0., 0.5 * (itv_a + itv_b),
                     args = (ac, mc, n))[0]
    if itv_M > Mt / 2.: itv_b = 0.5 * (itv_a + itv_b)
    else: itv_a = 0.5 * (itv_a + itv_b)
  m_hm = 0.5 * (itv_a + itv_b)

  # Make new axis
  N_ipt = 104
  A = m_hm ** 2 / (m_cut - 2. * m_hm)
  B = (2. / N_ipt) * np.log((m_cut - m_hm) / m_hm)
  m_ax = A * (np.exp(np.arange(N_ipt) * B) - 1.)

  # Make axis for interpolation
  M_m = np.zeros(N_ipt)
  for m_i, m_t in enumerate(m_ax):
    if m_i: M_m[m_i] = itg.quad(intfc, 0., m_t, args = (ac, mc, n))[0]
  M_m[0] = 0. # Fix the singularity
  M_m /= M_m[-1]

  # Make reverse interpolator
  m_ipt = ipt.interp1d(M_m, m_ax, kind = 'linear', assume_sorted = True)
  m_i   = m_ipt(np.random.random(N_pcs))

  # Sample uniform sphere
  X, Y, Z = np.random.standard_normal((3, N_pcs))
  Rsp = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
  X /= Rsp; Y /= Rsp; Z /= Rsp

  # push them to M
  X *= m_i; Y *= m_i; Z *= m_i * q

  return np.vstack((X, Y, Z)).T

'''
if 0:
  t1 = time.time()
  arr1 = Make_pow_bulge(1000000, 5., 2., 1., 0.5, 15.);
  arr2 = Make_exp_disk(1000000, 2.5, 0.25, 16., 5.)
  arr = np.concatenate((arr1, arr2))
  t2 = time.time()

  print 'delta_t', t2 - t1

  plt.hist2d(arr[:, 0], arr[:, 2], range = [[-12., 12.], [-12., 12.]],
             norm = LogNorm(), bins = 128, cmap = 'gray')
  plt.show()

if 0:
  t1 = time.time()
  arr = Make_exp_disk(1000000, 2.5, 0.4, 16., 5.)
  t2 = time.time()
  print 'delta_t', t2 - t1
  plt.hist2d(arr[:, 0], arr[:, 2], range = [[-16., 16.], [-16., 16.]],
             norm = LogNorm(), bins = 128)
  plt.show()
'''
