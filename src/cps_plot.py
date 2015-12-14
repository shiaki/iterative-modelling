
import numpy as np
import adapt_hist

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.colorbar as cbar

from mpl_toolkits.axes_grid1 import make_axes_locatable

valfc_num  = lambda idx, arg: float(idx.size)
valfc_mean = lambda idx, arg: np.mean(arg[idx])
valfc_std  = lambda idx, arg: np.std(arg[idx])

def valfc_cros(idx, args):
  P = args[0][idx]; Q = args[1][idx]
  return np.mean((P - np.mean(P)) * (Q - np.mean(Q)))

def scatter_plot(pcs, R_max = None, Z_max = None,
                 N_pcs_max = 20000, saveto = None):

  if R_max == None: R_max = np.linalg.norm(pcs[:, :2], axis = 1).max()
  if Z_max == None: Z_max = np.abs(pcs[:, 2]).max()
  if Z_max > R_max: R_max = Z_max

  if N_pcs_max == None: pcs_t = pcs
  else: pcs_t = pcs[:N_pcs_max]

  # Make figure
  panel_ratio = Z_max / R_max
  if panel_ratio < 0.25: panel_ratio = 0.25
  gs_t = gs.GridSpec(2, 1, height_ratios = [panel_ratio, 1.])
  fig = plt.figure(figsize = (6., 6. * (1. + panel_ratio)))

  ax1 = fig.add_subplot(gs_t[1], aspect = 'equal')
  ax1.set_xlim(-R_max, R_max)
  ax1.set_ylim(-R_max, R_max)
  ax1.autoscale(enable = False)
  ax1.scatter(pcs_t[:, 0], pcs_t[:, 1], s = 0.1)

  ax2 = fig.add_subplot(gs_t[0], sharex = ax1, aspect = 'equal')
  ax2.set_xlim(-R_max, R_max)
  ax2.set_ylim(-Z_max, Z_max)
  ax2.autoscale(enable = False)
  ax2.scatter(pcs_t[:, 0], pcs_t[:, 2], s = 0.1)
  ax2.get_xaxis().set_visible(False)

  plt.subplots_adjust(wspace = 0., hspace = 0.)

  if saveto == None: plt.show()
  else: plt.savefig(saveto, bbox_inches = 'tight')

def cylindrical_anisotropy_plot(pcs, R_max = None, Z_max = None,
      base_pixel_size = 0.5, max_lv = 4, plot_figs = ['sigma_z',
      'sigma_r', 'sigma_phi','v_phi', 'sigma_rz'],
      figure_size = None, panel_alignment = 'vertical',
      cmap_scale_sigma = None, cmap_scale_cross = None,
      cmap_scale_vphi = None, saveto = None):

  # find pos & vel in cylindrical coord.
  R = np.linalg.norm(pcs[:, :2], axis = 1)
  Z, Vz = pcs[:, 2], pcs[:, 5]

  cos_phi, sin_phi = pcs[:, 0] / R, pcs[:, 1] / R
  Vr  = pcs[:, 3] * cos_phi + pcs[:, 4] * sin_phi
  Vaz = pcs[:, 4] * cos_phi - pcs[:, 3] * sin_phi

  # fold Z and velocity Vz
  Vzf = np.where(Z > 0, Vz, -Vz)
  Zf  = np.abs(Z)

  # find limits of histogram
  make_histrg = lambda ar, ps: np.ceil(np.abs(ar).max() / ps) * ps
  if R_max == None: R_max = make_histrg(R, base_pixel_size)
  if Z_max == None: Z_max = make_histrg(Zf, base_pixel_size)

  # find pixel number
  R_npx = np.round(R_max / base_pixel_size).astype(int)
  Z_npx = np.round(Z_max / base_pixel_size).astype(int)

  # make histogram
  print "Making histogram..."
  hist = adapt_hist.adaptive_hist2d(R, Zf, hist_bins = (R_npx, Z_npx),
             hist_lim = [[0., R_max], [0., Z_max]], N_subgrid = (2, 2),
             N_maxlv = max_lv, if_refine = lambda K: K.size > 300)

  # evaluate:
  print "Making figures..."

  # Figs: sigma_r, sigma_z, sigma_phi, sigma_rz, mean_vphi
  if figure_size == None: fsize = 18.
  else:                   fsize = figure_size

  N_panels, I_panel = len(plot_figs), 1

  if panel_alignment == 'vertical':
    fig = plt.figure(figsize = (fsize, fsize * N_panels * Z_max / R_max))
    panel_cols, panel_rows = 1, N_panels
  elif panel_alignment == 'horizontal':
    fig = plt.figure(figsize = (fsize * N_panels, fsize * Z_max / R_max))
    panel_cols, panel_rows = N_panels, 1
  else: raise RuntimeError('Bad "panel_alignment" parameter.')

  # set limit for colorbar
  if cmap_scale_vphi == None: vphi_min, vphi_max = None, None
  elif isinstance(cmap_scale_vphi, (tuple, list)):
    vphi_min, vphi_max = cmap_scale_vphi
  else: vphi_min = -cmap_scale_vphi; vphi_max = -vphi_min

  if cmap_scale_sigma == None: sigma_min, sigma_max = None, None
  else: sigma_min, sigma_max = 0., cmap_scale_sigma

  if cmap_scale_cross == None: cross_min, cross_max = None, None
  else: cross_min = -cmap_scale_cross; cross_max = -cross_min

  # draw sigma_r
  if 'sigma_r' in plot_figs:
    ax1 = fig.add_subplot(panel_rows, panel_cols, I_panel,
                          aspect = 'equal', axisbg = '0.5')
    ax1.set_xlim(0., R_max); ax1.set_ylim(0., Z_max)
    sigma_r, smap = hist.eval_plot(valfc_std, args = Vr,
                      unit_area = False, cmap = 'jet',
                      cmap_vmin = sigma_min, cmap_vmax = sigma_max)
    ax1.annotate(r'$\sigma_{R}$', xy = (0.1, 0.9),
                 xycoords = 'axes fraction',
                 horizontalalignment = 'left', verticalalignment = 'top',
                 fontsize = 48)
    ax1.add_collection(sigma_r)
    div = make_axes_locatable(ax1)
    cax = div.append_axes('right', size = '5%', pad = 0.05)
    plt.colorbar(smap, cax = cax, ticks = [0., sigma_max])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    if I_panel == 1: first_ax = ax1
    last_ax = ax1
    I_panel += 1

  # draw sigma_z
  if 'sigma_z' in plot_figs:
    ax2 = fig.add_subplot(panel_rows, panel_cols, I_panel,
                          aspect = 'equal', axisbg = '0.5')
    ax2.set_xlim(0., R_max); ax2.set_ylim(0., Z_max)
    sigma_z, smap = hist.eval_plot(valfc_std, args = Vzf,
                      unit_area = False, cmap = 'jet',
                      cmap_vmin = sigma_min, cmap_vmax = sigma_max)
    ax2.annotate(r'$\sigma_{Z}$', xy = (0.1, 0.9),
                 xycoords = 'axes fraction',
                 horizontalalignment = 'left', verticalalignment = 'top',
                 fontsize = 48)
    ax2.add_collection(sigma_z)
    div = make_axes_locatable(ax2)
    cax = div.append_axes('right', size = '5%', pad = 0.05)
    plt.colorbar(smap, cax = cax, ticks = [0., sigma_max])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    if I_panel == 1: first_ax = ax2
    last_ax  = ax2
    I_panel += 1

  # draw sigma_phi
  if 'sigma_phi' in plot_figs:
    ax3 = fig.add_subplot(panel_rows, panel_cols, I_panel,
                          aspect = 'equal', axisbg = '0.5')
    ax3.set_xlim(0., R_max); ax3.set_ylim(0., Z_max)
    sigma_az, smap = hist.eval_plot(valfc_std, args = Vaz,
                       unit_area = False, cmap = 'jet',
                       cmap_vmin = sigma_min, cmap_vmax = sigma_max)
    ax3.annotate(r'$\sigma_{\phi}$', xy = (0.1, 0.9),
                 xycoords = 'axes fraction',
                 horizontalalignment = 'left', verticalalignment = 'top',
                 fontsize = 48)
    ax3.add_collection(sigma_az)
    div = make_axes_locatable(ax3)
    cax = div.append_axes('right', size = '5%', pad = 0.05)
    plt.colorbar(smap, cax = cax, ticks = [0., sigma_max])
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    if I_panel == 1: first_ax = ax3
    last_ax = ax3
    I_panel += 1

  # draw sigma_rz
  if 'sigma_rz' in plot_figs:
    ax4 = fig.add_subplot(panel_rows, panel_cols, I_panel,
                          aspect = 'equal', axisbg = '0.5')
    ax4.set_xlim(0., R_max); ax4.set_ylim(0., Z_max)
    sigma_rz, smap = hist.eval_plot(valfc_cros, args = (Vr, Vzf),
                       unit_area = False, cmap = 'jet',
                       cmap_vmin = cross_min, cmap_vmax = cross_max)
    ax4.annotate(r'$\sigma_{RZ}$', xy = (0.1, 0.9),
                 xycoords = 'axes fraction',
                 horizontalalignment = 'left', verticalalignment = 'top',
                 fontsize = 48)
    ax4.add_collection(sigma_rz)
    div = make_axes_locatable(ax4)
    cax = div.append_axes('right', size = '5%', pad = 0.05)
    plt.colorbar(smap, cax = cax, ticks = [cross_min, 0., cross_max])
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    if I_panel == 1: first_ax = ax4
    last_ax  = ax4
    I_panel += 1

  # draw mean_v_phi
  if 'v_phi' in plot_figs:
    ax5 = fig.add_subplot(panel_rows, panel_cols, I_panel,
                          aspect = 'equal', axisbg = '0.5')
    ax5.set_xlim(0., R_max); ax5.set_ylim(0., Z_max)
    mean_vaz, smap = hist.eval_plot(valfc_mean, args = Vaz,
                       unit_area = False, cmap = 'jet',
                       cmap_vmin = vphi_min, cmap_vmax = vphi_max)
    ax5.annotate(r'$\overline{V_{\phi}}$', xy = (0.1, 0.9),
                 xycoords = 'axes fraction',
                 horizontalalignment = 'left', verticalalignment = 'top',
                 fontsize = 48)
    ax5.add_collection(mean_vaz)
    div = make_axes_locatable(ax5)
    cax = div.append_axes('right', size = '5%', pad = 0.05)
    plt.colorbar(smap, cax = cax, ticks = [vphi_min, 0., vphi_max])
    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    if I_panel == 1: first_ax = ax5
    last_ax = ax5
    I_panel += 1

  if panel_alignment == 'horizontal':
    first_ax.get_xaxis().set_visible(True)
    first_ax.get_yaxis().set_visible(True)
    first_ax.set_xlabel('R')
    first_ax.set_ylabel('|Z|')
  elif panel_alignment == 'vertical':
    last_ax.get_xaxis().set_visible(True)
    last_ax.get_yaxis().set_visible(True)
    last_ax.set_xlabel('R')
    last_ax.set_ylabel('|Z|')
  else: raise RuntimeError('Meow.')

  if saveto == None: plt.show()
  else: plt.savefig(saveto, bbox_inches = 'tight')

  plt.clf()
  plt.close()

  return 0

# TEST

if __name__ == '__main__':

  import pcs_snap

  snap = pcs_snap.read_snap_galaxy_pcs('./run999/run999.pcs13')

  '''
  cylindrical_anisotropy_plot(snap['cps']['C2']['pcs'],
    R_max = 2.5, Z_max = 2.5, base_pixel_size = 0.25,
    panel_alignment = 'horizontal',
    cmap_scale_cross = 0.65 ** 2, cmap_scale_sigma = 0.65,
    cmap_scale_vphi  = 0.65, saveto = 'halo.png', figure_size = 6.)

  cylindrical_anisotropy_plot(snap['cps']['C1']['pcs'],
    R_max = 2.5, Z_max = 0.5, base_pixel_size = 0.25,
    panel_alignment = 'vertical',
    cmap_scale_cross = 0.01, cmap_scale_sigma = 0.3,
    cmap_scale_vphi  = 1., saveto = 'disk.png', figure_size = 18.)
  '''

  scatter_plot(snap['cps']['C1']['pcs'], R_max = 4., Z_max = 4.)
  scatter_plot(snap['cps']['C2']['pcs'], R_max = 4., Z_max = 4.)
