import util
import pdb
import numpy as np
# import matplotlib.pyplot as plt #might not need to include 

filename = 'take_home_movie_compresed.tiff'
## Segment objects in each tiff

# specify input parameters 
threshold = 6
# Initialize array to store masks and coordinates
im_labeled = np.empty((600,800,15))

tiffs_to_process = range(0,15)
df_all = [] # initialize list for object information
for itiff in tiffs_to_process:
    [im_labeled[:,:,itiff], df_itiff] = util.segment_objects(filename, tstack = itiff, bkg_thresh = threshold)
    df_all.append(df_itiff) # collect object information 

# Begin object tracking on first and second frame of the tiff stack 
frame0 = 0
frame1 = 1
[df_tracks, t] = util.link_frames(df_all,frame0,frame1)

# Complete object tracking by building tracks in df_tracks
df_tracks = util.track_objects(df_all,df_tracks,tiffs_to_process,t)

# Visualize object masks and coordinates 
util.make_gif(tiffs_to_process,filename,itiff,im_labeled,df_tracks)

# Visualize tracking results and fidelity 
print('Visualizing tracked objects (close figure to continue)')
channel = 1
clabel = 'x coordinate'
util.plot_tracks(df_tracks,channel,clabel)


# Plot object intensity 
print('Visualizing object intensities (close figure to continue)')
channel = 3
ylabel = 'Intensity'
util.plot_objects(df_tracks,channel,ylabel)





