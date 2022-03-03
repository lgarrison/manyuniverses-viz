import sys
import os

os.environ['ABACUS'] = os.environ['HOME'] + '/abacus'
sys.path.insert(0, os.environ['ABACUS'])

from pathlib import Path

from abacusnbody.data import read_abacus
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import astropy.table
from astropy.table import Table
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import Corrfunc

box = 2000.
AbacusSummit = Path('/mnt/home/lgarrison/ceph/AbacusSummit/')

def count_pairs(t, domain, nthread=1,):
    bins = np.linspace(0.1, 10.1, 2**4 * 3**4 + 1)

    print(f'Using {nthread=}')
    res = Corrfunc.theory.DD(1, nthread, bins, *t['pos'].T, periodic=False, verbose=True, bin_type='lin')

    res = Table(res, meta=t.meta.copy())
    res['rmid'] = (res['rmin'] + res['rmax'])/2.
    res.meta['N_DD'] = len(t)
    res.meta['domain_DD'] = domain

    return res


def cutout_sim_particles(sim, cen, width, nthread=1):
    t = []
    for pfn in sorted(sim.glob('halos/z0.500/*_rv_A/*_000.asdf')):
        t += [read_abacus.read_asdf(pfn, load=('pos','pid'))]
    t = astropy.table.vstack(t)

    pid = []
    for pfn in sorted(sim.glob('halos/z0.500/*_pid_A/*_000.asdf')):
        pid += [read_abacus.read_asdf(pfn)]
    pid = astropy.table.vstack(pid)

    t = astropy.table.hstack([t,pid])

    L = t.meta['BoxSize']
    domain = [L/34, L, L]
    pairs = count_pairs(t, domain, nthread=nthread)

    iord = argcutout(t['pos'], cen, width, t.meta['BoxSize'])

    t = t[iord]

    return t, pairs

def plot_tsc(p, cen, width, ngrid, box=box, tscbox=None):
    #from Abacus.Analysis.PowerSpectrum import TSC
    if tscbox is None:
        tscbox = box
    subp = cutout(p, cen, width, box)
    subp -= cen
    subp = np.roll(subp, -1, axis=1)  # TSC only lets you project out the z axis
    dens = TSC.BinParticlesFromMem(subp, (ngrid,ngrid), tscbox, nthreads=4, inplace=True, norm=True)
    
    fig, ax = plt.subplots(dpi=144)
    ax.set_aspect('equal')
    ax.imshow(np.log(dens.T + 2), origin='lower')

@nb.njit
def cutout(p, c, w, box):
    w = w/2
    new = np.empty_like(p)
    j = 0
    for i in range(len(p)):
        for k in range(3):
            dx = np.abs(p[i][k] - c[k])
            while dx > box/2:
                dx = np.abs(dx - box)
            if dx > w[k]:
                break
        else:
            for k in range(3):
                new[j][k] = p[i][k]
            j += 1
    
    return new[:j]


@nb.njit
def argcutout(p, c, w, box):
    w = w/2
    iord = np.empty(len(p), dtype=np.int64)
    j = 0
    for i in range(len(p)):
        for k in range(3):
            dx = np.abs(p[i][k] - c[k])
            while dx > box/2:
                dx = np.abs(dx - box)
            if dx > w[k]:
                break
        else:
            iord[j] = i
            j += 1
    
    return iord[:j]
