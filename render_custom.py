#!/usr/bin/env python3

import os
import sys
import io
import multiprocessing
from time import perf_counter
import argparse

import asdf
import numpy as np
import ahlive
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
#import ffmpeg
import imageio as iio
import colorcet
from tqdm import trange, tqdm
import threadpoolctl
import numba as nb
from astropy.table import Table

os.environ['ABACUS'] = '/mnt/home/lgarrison/abacus'
sys.path.insert(0,os.environ['ABACUS'])

os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-lgarrison'

NTHREAD = len(os.sched_getaffinity(0))


def zoom(p, wold, wnew):
    p = p - wold/2
    # if a particle peeks inside the drawing region during any of its trajectory, keep it
    mask = np.any(np.abs(p[:,:,:]) < wnew/2, axis=(1,2))
    p = p[mask,:] + wnew/2

    return p


def cut(p, w):
    mask = np.all(np.abs(p) < w/2, axis=-1)
    return p[mask]


def scatter_density(x, y, ax, size=10., bw=.03, adaptive=False, **scatter_kwargs):
    #from matplotlib.mlab import griddata
    from sklearn.neighbors import KernelDensity
    #from scipy.stats import gaussian_kde
    #from KDEpy.TreeKDE import TreeKDE

    xy = np.vstack([x,y]).T
    
    # Method 1: scipy
    #color = gaussian_kde(xy)(xy)
    
    # Method 2: sklearn
    kde = KernelDensity(kernel='gaussian', bandwidth=bw, rtol=1e-1).fit(xy)
    color = kde.score_samples(xy)

    # Method 3: KDEpy
    #kde = TreeKDE(bw=bw).fit(xy)
    #color = kde.evaluate(xy)
    #color = np.log(color)

    if adaptive:
        # Shrink the bw in high-density regions; grow it in low density regions
        # TODO: very little effect.  Might need to shrink in log space.

        # scale to 0 to 1
        color -= color.min()
        color /= color.max()
        #import pdb; pdb.set_trace()
        alpha = 0.99999
        newbw = bw*(1 - alpha*color)

        print('Starting new KDE...')
        #kde = TreeKDE(bw=newbw).fit(xy)
        color = kde.evaluate(xy)
        color = np.log(color)

    
    idx = color.argsort()
    x, y, color = x[idx], y[idx], color[idx]
    
    sc = ax.scatter(x, y, s=size, c=color, **scatter_kwargs)
        
    return x, y, color, idx, kde, sc


def outline_text(text, w):
    text.set_path_effects([path_effects.Stroke(linewidth=w, foreground='black'),
            path_effects.Normal()])


def get_text_height(t):
    bb = t.get_window_extent()
    #width = bb.width
    height = bb.height
    return height


def do_cosmology_text(ax, p, px, py):
    '''Display the current cosmology params, and which ones are changing
    '''

    lines = []
    color = {-1:'red', 0:'white', 1:'lime'}
    for k in p:
        if k in ('pos','cosm', r'\Omega_c', r'\Omega_b', 'Growth', 'xi', 'rmid', 'framenum'):
            continue
        v, flag = p[k]
        c = color[flag]
        lines += [dict(left=f'${k}$', right=f'$ = {v:.3g}$', color=c)]

    #fontproperties = mpl.font_manager.FontProperties(weight='bold')
    lh = dict(viswall=0.0222, slide=0.0263)[context]
    for i,line in enumerate(lines[::-1]):
        #text = ax.text(0.08, 0.2 - i*lh, lines[i]['left'],
        text = ax.text(px, py + i*lh, line['left'],
                    fontsize=16., c=line['color'],
                    transform=ax.transAxes,
                    #transform=None,  # pixels
                    #fontfamily='monospace',
                    ha='right', va='bottom',
                )
        #th = get_text_height(text)
        outline_text(text, w=4.)
        text = ax.text(px+2/1080, py + i*lh, line['right'],
                    fontsize=16., c=line['color'],
                    transform=ax.transAxes,
                    #transform=None,
                    #fontfamily='monospace',
                    ha='left', va='bottom',
                )
        outline_text(text, w=4.)


def do_2pcf_inset(fig, p, x=0.5, y=None, w=0.3):
    w /= figaspect
    ins = fig.add_axes((x-w/2, y, w, w/1.62*figaspect), zorder=3000)
    ins.set_facecolor('none')
    for spine in ins.spines.values():
        spine.set_edgecolor('white')
        spine.set(linewidth=1.5)
    ins.tick_params(axis='both', color='white', labelcolor='white', labelsize='x-large')
    rexp = 1.5
    ins.plot(p['rmid'], p['xi']*p['rmid']**rexp, color='white', lw=2.)
    xtxt = ins.set_xlabel('Distance $r$ [Mpc/$h$]', color='white', labelpad=0., fontsize='x-large')
    ytxt = ins.set_ylabel(rf'Clustering $r^{{{rexp}}}\xi(r)$', color='white', labelpad=0., fontsize='x-large')
    ins.set_ylim(4,21)
    ins.set_xlim(0.08, 12)
    ins.set_xscale('log')
    ins.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,pos: f'{v:.2g}'))

    outline_text(xtxt, 2)
    outline_text(ytxt, 2)

    rightedge = x - w/2 + w
    return rightedge


def render_one(p):
    X,Y = p['pos']
    c = p['cosm']
    framenum = p['framenum']

    fig, ax = plt.subplots(figsize=(figinchesw,figinchesh), dpi=dpi, subplot_kw=dict(facecolor=colorcet.cm.fire(0)))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim(-w/2,w/2)
    ax.set_ylim(-h/2,h/2)
    #ax.set_aspect('equal')
    
    if mode == 'scatter':
        #ax.scatter(X, Y)
        # transpose to better align with the annotations
        _, _, color, _, _, _ = scatter_density(-Y, -X, ax, cmap=colorcet.cm.fire, ec=None,
                                                bw=0.02, size=4,
                                                vmin=-7, vmax=-1.5,
                                                )
        #if c == 1:
        #    print(color.min(),color.max())
    #elif mode == 'tsc':
    #    p = np.stack((X,Y,np.full_like(X, w/2)), axis=-1) - w/2
    #    p = cut(p, w)
    #    delta = TSC.BinParticlesFromMem(p, (256,256), w, nthreads=1, inplace=True, norm=True,
    #                                    )
    #    ax.imshow(np.log10(delta + 1.2).T, origin='lower', interpolation='none', cmap=colorcet.cm.fire, extent=(0,w,0,w),
    #                vmin=-0.5, vmax=2,
    #                )

    # Do 2PCF inset
    if context == 'viswall':
        insety = 0.2
        insetw = 0.25
    else:
        insety = 0.12
        insetw = 0.3
    insetrightedge = do_2pcf_inset(fig, p, y=insety, w=insetw)
    
    fontproperties = mpl.font_manager.FontProperties(weight='bold')

    do_cosmology_text(ax, p, px=insetrightedge + 0.05, py=insety-0.04)

    title = ax.text(0.5, 0.96,
            'The Many Universes of AbacusSummit',
            va='top', ha='center',
            fontproperties=fontproperties,
            fontsize=24., c='white',
            #bbox=dict(color='white', alpha=0.3),
            transform=ax.transAxes)
    outline_text(title, w=3)
    
    line, = ax.plot([-8.6, 8.6], [8.5, 8.5], c='w',
            #transform=(ax.transAxes),
            lw=2)
    outline_text(line, w=3)

    cosm_text = ax.text(0.5, 0.90,
            f'Cosmology {c} of {ncosmo}',
            va='top', ha='center',
            fontproperties=fontproperties,
            fontsize=20., c='white',
            #bbox=dict(color='white', alpha=0.3),
            transform=ax.transAxes)
    outline_text(cosm_text, w=3)

    if center_url:=False:
        urltext = ax.text(0.5, 0.03, r'https://AbacusSummit.ReadTheDocs.io',
                va='bottom', ha='center',
                c='white', transform=ax.transAxes,
                fontfamily='monospace',
                fontsize='x-large',
                )
    else:
        urltext = ax.text(0.99, 0.01, r'AbacusSummit.ReadTheDocs.io/',
                va='bottom', ha='right',
                c='white', transform=ax.transAxes,
                fontfamily='monospace',
                fontsize='x-large',
                )
    outline_text(urltext, w=1.5)

    ax.add_patch(
        (arrow:=mpl.patches.FancyArrowPatch((-4, 5), (+4, 5),
                                arrowstyle='<->', color='w', zorder=3000, mutation_scale=5*4))
    )
    scaletext = ax.text(0.5, 0.76, r'$8\ \mathrm{Mpc}/h$', ha='center', va='bottom', transform=ax.transAxes,
            size=18., color='w', #fontproperties=fontproperties,
            #bbox=dict(alpha=0.2, color='k', pad=0.5),
            )
    outline_text(arrow, w=2)  # ?
    outline_text(scaletext, w=2)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')

    if framenum == 0:
        fig.savefig('c000.png', format='png')
    plt.close(fig)
    return buf


def interp_between(p12, iframes=6):
    '''Render between `p1` and `p2`, using the given number of interpolation frames and easing function

    `p12` is shape (2,ndim,NP)
    '''
    _, ndim, NP = p12.shape
    easing = ahlive.Easing(frames=iframes-1)

    pi = np.empty((iframes,ndim,NP), dtype=p12.dtype)
    for d in range(ndim):
        pi[:,d] = easing.interpolate(p12[:,d].T).T

    return pi


def interp_all_frames(p, pairs, iframes=6, introframes=6, xi_smooth=2):
    nstate, ndim, NP = p['pos'].shape
    
    nframe = (nstate-1)*iframes + introframes

    cosm_params = { r'\Omega_b':        lambda c: c['omega_b']/(c['H0']/100.)**2,
                    r'\Omega_c':        lambda c: c['omega_cdm']/(c['H0']/100.)**2,
                    r'\Omega_M':        lambda c: c['Omega_M'],
                    r'\Omega_\Lambda':  lambda c: c['Omega_DE'],
                    r'\Omega_\nu':      lambda c: c['Omega_Smooth'],
                    r'h':               lambda c: c['H0']/100.,
                    r'\sigma_8':        lambda c: c['sigma8_m'],
                    r'n_s':             lambda c: c['n_s'],
                    r'\alpha_s':        lambda c: c['alpha_s'],
                    r'w_0':             lambda c: c['w0'],
                    r'w_a':             lambda c: c['wa'],
                    #r'\mathrm{Growth}': lambda c: c['Growth'],
                }
    
    # We did DD in fine bins; coarsify
    nbin = pairs['npair'].shape[0] // xi_smooth
    smoothpairs = Table({}, meta=pairs.meta.copy())
    DD = pairs['npair'].reshape(nbin, xi_smooth, -1).sum(axis=1)
    RR = pairs['RR'].reshape(nbin, xi_smooth, -1).sum(axis=1)
    rmin = pairs['rmin'][::xi_smooth]
    rmax = pairs['rmax'][xi_smooth-1::xi_smooth]  # ?
    smoothpairs['xi'] = DD/RR - 1  # shape (nbin, nstate)
    smoothpairs['rmid'] = (rmax + rmin) / 2
    #print((smoothpairs['xi'].T*smoothpairs['rmid']**1.5).min())

    frames = Table({'pos': np.empty((nframe,2,NP), dtype=p['pos'].dtype),
                    'cosm': np.empty((nframe), dtype=int),
                    'framenum': np.arange(nframe), 
                    
                    'rmid': np.empty((nframe,nbin)),
                    'xi': np.empty((nframe,nbin)),
                    
                    } | {param: np.empty((nframe,2)) for param in cosm_params}
                    )

    # intro frames are static copies of the first state
    frames[:introframes]['pos'] = p['pos'][0,:2]
    frames[:introframes]['cosm'] = 1
    frames[:introframes]['rmid'] = smoothpairs['rmid']
    frames[:introframes]['xi'] = smoothpairs['xi'][:,0]
    for param in cosm_params:
        frames[:introframes][param][:,0] = cosm_params[param](p.meta[0])
        frames[:introframes][param][:,1] = 0

    for i in trange(nstate-1, unit='interp'):
        _frames = frames[introframes + i*iframes : introframes + (i + 1)*iframes]
        _frames['pos'][:] = interp_between(p['pos'][i:i+2,:2], iframes=iframes)
        _frames['cosm'][:] = i + 1  # the cosmology number will change at the end of the transition
        
        _frames['rmid'][:] = smoothpairs['rmid']
        _xi = interp_between(smoothpairs['xi'][:,i:i+2].T[:,None],  # shape (nbin,2) -> (2,1,nbin)
                                        iframes=iframes)  # shape (iframes, 1, nbin)
        _frames['xi'][:] = _xi[:,0]  # shape (iframes, nbin)
        
        # TODO: interpolate/scroll these numbers?
        for param in cosm_params:
            # the cosmology params will change at the beginning of the transition
            _frames[param][:,0] = cosm_params[param](p.meta[i+1])
            if cosm_params[param](p.meta[i+1]) > cosm_params[param](p.meta[i]):
                _frames[param][:,1] = 1  # increasing
            elif cosm_params[param](p.meta[i+1]) < cosm_params[param](p.meta[i]):
                _frames[param][:,1] = -1  # decreasing
            else:
                _frames[param][:,1] = 0  # equal

    # frames['pos']: shape (nframes, ndim, NP)
    # frames[param]: shape (nframes, 2)
    
    frames[-1]['cosm'] = nstate

    return frames


def render_all_frames(p, nworker=-1):
    nframe, ndim, NP = p['pos'].shape
    frames = []

    if nworker < 1:
        nworker = NTHREAD

    _tlimiter = threadpoolctl.threadpool_limits(1)
    tstart = perf_counter()
    print(f'Rendering {nframe=} with {nworker=}...', flush=True)
    with multiprocessing.Pool(processes=nworker, initializer=None, initargs=None) as pool:
        # one table row per worker
        frames = pool.map(render_one, [dict(r) for r in p], chunksize=1)
    t = perf_counter() - tstart
    print(f'{t:.3g} sec, {len(frames)/t:.3g} frame/sec')

    return frames


def write_movie_from_frames(frames, fps=12, hpix=None, mode=None):
    fn = f'abacussummit_universes_{mode}_fps{fps}_{hpix}p.webm'

    import logging
    logging.getLogger('imageio_ffmpeg').setLevel('INFO')

    crf = {720:'24', 1080: '31', 1440: '24', 1920: '17'}

    ffmpeg_params = [# webm
                     '-crf', crf[hpix],
                     '-b:v', '0',
                     '-threads', str(NTHREAD),
                     '-row-mt', '1',
                     '-quality', 'good',
                    ]

    if onepass:=False:
        w = iio.get_writer(f'/mnt/home/lgarrison/ceph/multicosmo-viz/onepass.{fn}',
                       format='FFMPEG', mode='I', fps=fps,
                       quality=None,
                       codec='libvpx-vp9',
                       output_params=ffmpeg_params,
                       ffmpeg_log_level='info',
                       macro_block_size=None,
                      )
    else:
        w = iio.get_writer(f'/mnt/home/lgarrison/ceph/multicosmo-viz/pass1.{fn}',
                       format='FFMPEG', mode='I', fps=fps,
                       quality=None,
                       codec='libvpx-vp9',
                       output_params=ffmpeg_params + [
                                      '-pass', '1',
                                      '-f', 'null'
                                      ],
                       ffmpeg_log_level='info',
                       macro_block_size=None,
                      )
    for f in frames:
        im = iio.imread(f)
        w.append_data(im)

    if not onepass:
        w = iio.get_writer(f'/mnt/home/lgarrison/ceph/multicosmo-viz/{fn}',
                       format='FFMPEG', mode='I', fps=fps,
                       quality=None,
                       codec='libvpx-vp9',
                       output_params=ffmpeg_params + [
                                      '-pass', '2',
                                      ],
                       ffmpeg_log_level='info',
                       macro_block_size=None,
                      )
        for f in frames:
            im = iio.imread(f)
            w.append_data(im)
    w.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dpi', default=108, type=int)
    parser.add_argument('-wpix', default=1920, type=int)
    parser.add_argument('-hpix', default=1080, type=int)
    parser.add_argument('-context', choices=['viswall','slide'], default=None)
    parser.add_argument('-ncosmo', type=int)

    # lobby display, 2x downsample: -dpi 192 -wpix 2700 -hpix 1920 -context viswall
    # 1080p slide: -dpi 128 -wpix 1920 -hpix 1080 -context slide
    # 1440p slide: -dpi 171 -wpix 2560 -hpix 1440 -context slide
    # 1080p half-slide: -dpi 128 -wpix 1260 -hpix 1080 -context slide

    args = parser.parse_args()
    args = vars(args)

    with asdf.open('/mnt/home/lgarrison/ceph/multicosmo-viz/traj.asdf') as af:
        traj = af['particles']
        pair_traj = af['pairs']
    p = traj['pos']
    
    depth = 10
    oldlen = len(p)
    p = p[(np.abs(p[:,:,2]) < depth).any(axis=1)]
    print(f'{len(p)=:.3g} particles shown')
    
    #p = zoom(p, wold=40., wnew=30.)
    p = np.moveaxis(p, [1,2], [0,1])  # shape (ncosmo, ndim, NP)
    p = Table(dict(pos=p), meta=traj.meta.copy())

    mode = 'scatter'
    fps = 48
    dpi = args['dpi']
    figinchesw = args['wpix']/dpi
    figinchesh = args['hpix']/dpi
    ncosmo = args['ncosmo'] or len(p)
    context = args['context']

    h = 20.  # cutout height
    figaspect = figinchesw/figinchesh
    w = h*figaspect
    print(f'{h=:.3g},{w=:.3g},{depth=:.3g} Mpc/h')

    ########

    #if mode == 'tsc':
    #    from Abacus.Analysis.PowerSpectrum import TSC

    pinterp = interp_all_frames(p[:ncosmo], pair_traj, iframes=fps, introframes=fps)
    frames = render_all_frames(pinterp)
    write_movie_from_frames(frames, fps=fps, hpix=int(dpi*figinchesh), mode=mode)

    # TODO: different depth cut?
    # info on parameter changes
