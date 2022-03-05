# manyuniverses-viz

This directory contains the scripts used to generate the "Many Universes of AbacusSummit" video, available here: https://abacussummit.readthedocs.io/en/latest/visualizations.html#video-many-universes

## Author/License
Lehman Garrison (lgarrison.github.io)

Apache-2.0 License

## Usage
1. Make a cubic cutout (a subbox) from each sim using the `cutout_onesim.py` script. LHG used [disBatch](https://github.com/flatironinstitute/disBatch/) with the `cutout.disbatch` task file to run the jobs on the Rusty cluster at Flatiron.
1. Combine the cutouts into a single "trajectory" file, containing the location of each particle in each cosmology (discarding particles not present in all cosmologies)
1. Render the movie from the trajectory file, using `render_custom.py`

## Source Layout
- `cutout_onesim.py`, `make_traj.py`, `render_custom.py`: the main executable scripts to generate the movie
- `multicosmo.py`: some utilities used by `cutout_onesim.py`
- `cutout.disbatch`: disBatch script used to run `cutout_onesim.py` over all sims
- `old_notebooks/`: various explorations of ways to select the particles and render the movie
- `cosmologies.csv`: the cosmologies table, used for loading in some extra info
- `c000.png`: the first frame of the video, saved to disk before encoding starts
- `environment-rusty.sh`: commands LHG used to set up a shell environment on Rusty at Flatiron

## Details for Developers
### Cutouts
We chose to center the cutout on the biggest halo in the `000` slab.  We're only using the 3% particle subsample, which seems like more than enough for this particular zoom scale and plotting style.  It's interesting to think what we could do with 10% or 100%, though.

We did 40 Mpc/h cubic cutouts, to support a 20 Mpc/h height and a variable width, depending on the aspect ratio.  We did a depth of 20 Mpc/h.  But the cutouts are cubic to support flipping the axes later.

While doing the cutouts, we also compute the 2PCF *on the whole slab*, to get a smooth function to interpolate later.

### Trajectories
The matching of particles between cosmologies is done by PID, using the fast `np.intersect1d` function.  About 80% of particles are present in all cosmologies.

While we do the trajectories, we also compute the RR term on the slab level, ignoring any details of the fuzzy/overlapping slab edges.  The shift in the slab-level DC term is negligible.

### Rendering
The rendering (`render_custom.py`) is the most complicated part, and most of the effort went into that.

Each frame is rendered as a scatter plot, using the `scatter_density` method that LHG has been using for a while now. The idea is just to color each point by a KDE estimate of the local density of points, and then plot the densest points on top.

We use the perceptually-uniform colorcet "fire" colormap, and then manually tune the scatter plot vmin/vmax and KDE radius.

The text is outlined in black to aid readability against the colorful background.

Since we want to smoothly transition between cosmologies, we interpolate particle positions using the "easing" functions from `ahlive.Easing`, to give the transitions a "kinetic" feel.  We likewise interpolate the 2PCF.  The transitions are timed at about 1 second each.

One key is that we target a fairly high framerate of 48 fps.  This really helps the video seem silky-smooth, at the cost of more rendering time and bigger files.

So we end up with a few-thousand sets of interpolated particle positions, and then we farm out the rendering using Python multiprocessing.  The memory load isn't that big, so we just use 128 cores of a single AMD Rome node at Flatiron; it takes a minute or two.  The worker processes send their result back in the form of a `io.BytesIO()` buffer object holding a PNG, so the frames never touch disk.

Once we have all the frames, then we pass them to the `imageio` library, which, for our purposes, is useful because it knows how to pass raw PNG to ffmpeg.  It's also easy to customize the ffmpeg parameters.

ffmpeg takes a while to run; it can't use many cores very effectively.  But we did enable `-row-mt` for row multithreading, which helps.  We do two-pass encoding, which doesn't add much rendering time, but does produce smaller files.

The following websites had useful advice on tuning ffmpeg for VP9: https://trac.ffmpeg.org/wiki/Encode/VP9, https://developers.google.com/media/vp9/settings/vod.

We opted to use Constant Quantizer/Quality mode in the encoding for maximum fidelity, which honestly may have been overkill. The main problem is that it produces high fluctuations in video bitrate, so the playback may stutter on some platforms. But it hasn't been a big problem in practice, and results in consistently sharp video.

The `crf` value is the key tuning parameter for balancing file size and quality, and changes based on resolution.  We ended up using values basically equal to what Google recommends in their encoding guide.

We produced a few different resolutions: 1080p marginally resolves the particles; 1440p looks like the sweet spot. We played around with 720p, but the particles were underresolved. We could have gone back and made the particles bigger, but LHG was mostly interested in targeting high-resolution displays.

Some comments are left in `render_custom.py` at the bottom near the argument parsing to record what parameters were used to make the various resolutions.

We use the VP9 codec, which is more modern than H.264 and is royalty-free.  H.264 is not, so many Linux installations won't be able to natively play those vidoes, for example.  AV1 is more modern than VP9, but may not have as widespread support by browsers and video players.

We use the webm container format as it supports VP9 videos and has good support among modern web browsers.

One leftover annoyance is that PNG uses RGB colors, while VP9 wants to use CMYK. The colors look a little washed out in the video compared to the PNGs, to my eye. I wasn't able to get RGB-native VP9 to work. But it still looks pretty good overall.

### Hosting
LHG tried hosting the video on YouTube and Vimeo, and it looked awful. The videos were reencoded in the upload process and lost all their quality.

So we use "native" cloud hosting instead.  Backblaze B2 had enough free capacity but not enough free bandwidth. Amazon S3 has free capacity but not free bandwith; however, you can hook up the S3 bucket to the CloudFront load balancer, which has enough free monthly bandwidth for many downloads. So that's how you access the videos now.
