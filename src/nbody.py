#!/usr/bin/python

'''
  N-body backends
'''

TEST = False
TEST = True

work_dir = './playground/'

galaxy_progs_path = '/dgalex2/qinyj/galaxy/progs/'
galaxy_runid      = 999
galaxy_dump_time  = 1

import os, sys, glob, shutil, time
import subprocess as spc

import numpy      as np

import pcs_snap

r_tx = lambda s: '\033[31m'+s+'\033[0m'
g_tx = lambda s: '\033[32m'+s+'\033[0m'
y_tx = lambda s: '\033[33m'+s+'\033[0m'
b_tx = lambda s: '\033[34m'+s+'\033[0m'
m_tx = lambda s: '\033[35m'+s+'\033[0m'
c_tx = lambda s: '\033[36m'+s+'\033[0m'

dashed = '---------------------------------------------'

# start a GALAXY session with given snap
def run_galaxy(snap, I_iter, clean_only = False, **kw):

  print '\n\n\n\n'
  print y_tx(dashed)
  print r_tx('Starting I_iter ='), r_tx(str(I_iter)), '...'
  sys.stdout.flush()

  # go to the working directory
  wdir_init = os.getcwd()
  os.chdir(work_dir)
  wdir_t = os.getcwd() + '/'

  # if I_iter = 0, generate a script for GALAXY

  print y_tx(dashed)
  print g_tx("Writing restart file...")
  sys.stdout.flush()
  # first save this snapshot to somewhere else
  pcs_snap.save_snap_galaxy_pcs(\
      '%s/run%u.pcs0'%(wdir_t, galaxy_runid), snap)

  # convert pcs file to dmp
  print y_tx(dashed)
  print g_tx("Initializing GALAXY...")
  sys.stdout.flush()
  sp_t = spc.Popen(galaxy_progs_path + 'begin', stdin = spc.PIPE)
  sp_t.communicate(input = '%s\n0\n'%(galaxy_runid,))
  sp_t.wait()

  # run galaxy
  print y_tx(dashed)
  print g_tx("Running GALAXY main program...")
  sys.stdout.flush()
  sp_t = spc.Popen(galaxy_progs_path + 'galaxy', stdin = spc.PIPE)
  sp_t.communicate(input = '%s\n0\n'%(galaxy_runid,))
  sp_t.wait()

  print y_tx(dashed)
  print g_tx("Finalizing GALAXY...")
  sys.stdout.flush()
  # convert dmp to pcs again
  sp_t = spc.Popen(galaxy_progs_path + 'finish', stdin = spc.PIPE)
  sp_t.communicate(input = '%s\n%u\n'%(galaxy_runid, galaxy_dump_time))
  sp_t.wait()

  # load into memory
  print y_tx(dashed)
  print g_tx("Loading results...")
  sys.stdout.flush()
  new_snap = pcs_snap.read_snap_galaxy_pcs( \
      wdir_t + 'run%u.pcs%u'%(galaxy_runid, galaxy_dump_time))
  new_snap['time'] = 0.
  # otherwise, the converted dump file will not be runX.dmp0

  # make directory to save previous snapshots
  print y_tx(dashed)
  print g_tx("Cleaning working dir...")
  sys.stdout.flush()
  new_dir = wdir_t + 'dump%04u'%(I_iter,)
  if not os.path.exists(new_dir): os.makedirs(new_dir)

  # move files generated in previous runs (except for .dat file)
  for fname_t in glob.glob(wdir_t + 'run%u.*'%(galaxy_runid,)):
    if not fname_t.endswith('.dat'): shutil.move(fname_t, new_dir)
  print y_tx(dashed)
  print r_tx('Iter ='), I_iter, r_tx('Finished.')
  print y_tx(dashed)
  sys.stdout.flush()

  os.chdir(wdir_init)

  return new_snap

#=========================================================#
# Test routines

if TEST:

  if 1: 

    snap = pcs_snap.read_snap_galaxy_pcs('./dumps/run999.pcs0')
    t0 = time.time()
    snap = run_galaxy(snap, 1)
    snap = run_galaxy(snap, 2)
    snap = run_galaxy(snap, 3)
    snap = run_galaxy(snap, 4)
    snap = run_galaxy(snap, 5)
    snap = run_galaxy(snap, 6)
    snap = run_galaxy(snap, 7)
    snap = run_galaxy(snap, 8)
    snap = run_galaxy(snap, 9)
    snap = run_galaxy(snap, 10)
    t1 = time.time()

    print '\n\n\n\nEach step takes', (t1 - t0) / 10., 'sec.\n\n'
