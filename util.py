# Functions for object detection and track linking 
import pdb
import numpy as np
import skimage.io as skio
import skimage.filters as skf
from skimage import morphology, color, segmentation
from skimage.morphology import remove_small_objects, square, dilation
from skimage.filters import median
import matplotlib.pyplot as plt
from scipy  import ndimage
import imageio



def segment_objects(filename,**kwargs):
    """ Function for object detection for each stack in the provided tiff file """
    # read in tiff stack
    tiffs  = skio.imread(filename, plugin="tifffile")
    # parse inputs
    if kwargs:
        # assign current stack
        if 'tstack' in kwargs:
            tstack = kwargs['tstack']
        else: 
            tstack = 0
        # assign background threshold
        if 'bkg_thresh' in kwargs:
            bkg_thresh = kwargs['bkg_thresh']
        else:
            print('No threshold provdied...calculating threshold')
            # assign background frame if provided (default = 0)
            if 'bkg_frame' in kwargs:
                bkg_frame = kwargs['bkg_frame']
            else:
                bkg_frame = 0
            bkg_im = tiffs[bkg_frame]
            # assign default background coordinates and threshold if not provided
            if 'bkg_region' in kwargs:
                bkg_region = kwargs['bkg_region']
            else:
                # calculate threshold if threshold not provided
                print('No background coordinates provided...using default region (not recommended)')
                bkg_reg = np.array([[0, 0],[100,100]]) #[x1 y1, x2, y2] these are corners of rectangular region
            # coordinates
            x1 = bkg_reg[0,0]
            x2 = bkg_reg[1,0]
            y1 = bkg_reg[0,1]
            y2 = bkg_reg[1,1]
            # calculate background intensity(threshold)
            print('Calculating background intensity threshold...')
            bkg_thresh = (np.mean(bkg_im[y1:y2,x1:x2]) * 2) #threshold will be 2x the average background 
        # assign noisy pixel size
        if 'noisy_pixel_size' in kwargs:
            noisy_pixel_size = kwargs['noisy_pixel_size']
        else:
            noisy_pixel_size = 6 #default 
        # assign pixels to dilate objects by
        if 'dilation_size' in kwargs:
            dilation_size = kwargs['dilation_size']
        else:
            dilation_size = 5 #default            
    print('Processing frame ' + str(tstack+1))

    #To remove the noise in the image, binarize the image with a threshold value
    im = tiffs[tstack]

    # Binarize image
    im_binarized = im > bkg_thresh

    # Remove small pixels that are in still in the image
    im_filtered = remove_small_objects(im_binarized, min_size=noisy_pixel_size)

    # Dilate squares to get better object masks
    se = square(dilation_size)
    im_dilated = dilation(im_filtered, se) 

    # Fill in holes in each square, this can be useful if the dilation step leaves some holes in objects
    im_filled = ndimage.binary_fill_holes(im_dilated)

    # Erode the squares back to pre-dilaition size
    im_eroded = morphology.erosion(im_filled, se)

    # Label the connected components in the binary image
    im_labeled, num_objects = morphology.label(im_eroded, return_num=True)

    # Initialize figure axes
    
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    fig.subplots_adjust(hspace=0.5) # adjust the height spacing between subplots
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[0, 2]
    ax4 = axs[1, 0]
    ax5 = axs[1, 1]
    ax6 = axs[1, 2]

    # Plot results 
    ax1.imshow(im)
    ax1.set_title('Orginal')
    ax2.imshow(im_filtered)
    ax2.set_title('Thresholded')
    ax3.imshow(im_dilated)
    ax3.set_title('Dilated')
    ax4.imshow(im_filled)
    ax4.set_title('Filled')
    ax5.imshow(im_eroded)
    ax5.set_title('Eroded')
    ax6.imshow(im_eroded)
    ax6.set_title('Labeled Masks ('+str(num_objects)+' objs)')
    fig.suptitle('Frame '+str(tstack+1)+ ' segmentation')

    # Add text labels for each object, extract coordinates for each object 
    df = np.full((num_objects, 5), np.nan)
    for i in range(1, num_objects+1):
        # Find the center of mass of the i-th object
        y, x = np.where(im_labeled == i)
        xc, yc = np.mean(x), np.mean(y)
        # Find current object
        obj_mask = im_labeled == i
        # Add a text label with the object number at the center of mass
        ax6.text(xc, yc, str(i), color='r', fontsize=10)
        # Build data table to keep track of objects
        df[i-1,0] = i #object ID for current frame
        df[i-1,1] = xc #x coordinate
        df[i-1,2] = yc #y coordinate
        df[i-1,3] = np.mean(im[obj_mask])#average mask intenisty
        df[i-1,4] = tstack #current stack in tiff

    if tstack == 0:
        plt.show()
        print('Plotting segmentation results (close figure to continue)')
    
    plt.savefig(str(tstack) + '_segmentation.png')
    print('Segementation results saved as .png file')
    plt.close()
    return(im_labeled, df)


def link_frames(df_all,frame1,frame2,**kwargs):
    """ Beging tracking by connect objects in the first and second frame"""
    #search_range: range to serach for object in the next frame 
    if kwargs:
        if 'search_range' in kwargs:
            search_range = kwargs['search_range']
    else:
        search_range = 20 # pixels to search for same object in the next frame 
    df_tracks = np.full((20, len(df_all), len(df_all[0][0, :])), np.nan) # initialize track data frame (100 by frames, by 5 measurements)
    currtiff = 0
    num_objs = len(df_all[currtiff])
    remaining_objs = df_all[currtiff][:,0]
    
    # Loop through objects in tiff, look for similar oobjects in the next tiff
    next_tiff = df_all[currtiff+1]
    t = -1
    for iobj in range(0,num_objs):
        curr_obj = df_all[currtiff][iobj,0] # current object
        x = df_all[currtiff][iobj,1] # x coord
        y = df_all[currtiff][iobj,2] # y coord
        #find object in next tiff in a range
        xrange1 = x - search_range
        xrange2 = x + search_range
        yrange1 = y - search_range
        yrange2 = y + search_range
        objs_x = [x for x, i in enumerate(next_tiff[:,1]) if xrange1 <= i <= xrange2]
        objs_y = [x for x, i in enumerate(next_tiff[:,2]) if yrange1 <= i <= yrange2]
        linked_obj_ind = list(set(objs_x) & set(objs_y)) # index of the linked object
        linked_obj = next_tiff[linked_obj_ind,0] # ID of linked object
        
        # If object is linked, build the track
        if len(linked_obj_ind) > 0: 
            t = t + 1
            index = np.where(remaining_objs == curr_obj)[0]
            remaining_objs = np.delete(remaining_objs, index)
            df_tracks[t,currtiff,:] = np.array(df_all[currtiff][iobj,:]) # collect frame 1 information
            df_tracks[t,currtiff+1,:] = np.array(next_tiff[linked_obj_ind,:]) # collect linked object info 

    # Create new tracks for remaining objects
    if len(remaining_objs) > 0:
        for r_obj in remaining_objs:
            t = t+1
            index = np.where(df_all[currtiff][:,0] == r_obj)[0]
            df_tracks[t,currtiff,:] = np.array(df_all[currtiff][index,:]) # non linked object info 
    return(df_tracks,t)

def track_objects(df_all,df_tracks,tiffs_to_process,t,**kwargs):
    """Complete object tracking by building tracks in df_tracks"""
    #optional inputs:
    #search_range: range to serach for object in the next frame 
    if kwargs:
        if 'search_range' in kwargs:
            search_range = kwargs['search_range']
    else:
        search_range = 20 #pixels to search for same object in the next frame 
    print('Linking objects across frames...')
    for iframe in range(1,(tiffs_to_process[-1]+1)):
        currtiff = iframe
        num_objs = len(df_all[currtiff])
        remaining_objs = df_all[currtiff][:,0]
        # loop through objs in new frame and match with df_track
        for iobj in range(0,num_objs):
            curr_obj = df_all[currtiff][iobj,0] # current object
            x = df_all[currtiff][iobj,1] # x coord
            y = df_all[currtiff][iobj,2] # y coord
            #find object in next tiff in a range
            xrange1 = x - search_range
            xrange2 = x + search_range
            yrange1 = y - search_range
            yrange2 = y + search_range
            previous_frame = currtiff
            while previous_frame > - 1: # only check until first frame 
                previous_frame = previous_frame - 1
                objs_x = [x for x, i in enumerate(df_tracks[:,(previous_frame),1]) if xrange1 <= i <= xrange2]
                objs_y = [x for x, i in enumerate(df_tracks[:,(previous_frame),2]) if yrange1 <= i <= yrange2]
                linked_obj_ind = list(set(objs_x) & set(objs_y)) # index of the linked object
                if len(linked_obj_ind) > 0: # if object is linked, continue building the track
                    index = np.where(remaining_objs == curr_obj)[0]
                    remaining_objs = np.delete(remaining_objs, index)
                    df_tracks[linked_obj_ind,currtiff,:] = np.array(df_all[currtiff][iobj,:]) # collect frame 1 information
                    previous_frame = -2 # stop while loop when obj is linked
            # if track is not found create a new track
            if previous_frame == -1: 
                t = t + 1
                index = np.where(df_all[currtiff][:,0] == curr_obj)[0]
                df_tracks[t,currtiff,:] = np.array(df_all[currtiff][index,:]) # non linked object info 

    # Remove rows with all nans, or objects that only appear for one or two frame of the movie
    min_track_length = 2
    max_nans =  len(df_tracks[0,:,0]) - min_track_length + 1 
    n_nans = np.sum(np.isnan(df_tracks[:,:,0]),axis=1)
    mask = n_nans < max_nans
    df_tracks = df_tracks[mask]
    return(df_tracks)

def plot_tracks(df_tracks,channel,clabel):
    """Visualize tracking results and fidelity. Function plots heatmap of objects"""
    plt.imshow(df_tracks[:,:,channel],cmap='viridis')
    plt.xlabel("Frame")
    plt.ylabel("Object")
    plt.title("Tracked Objects")
    numobjs = len(df_tracks[:,:,channel])
    x_min, x_max = 0, df_tracks.shape[1] 
    plt.yticks(range(0, numobjs), [str(j) for j in range(1, numobjs + 1)])
    plt.xticks(range(0, x_max,2),[str(j) for j in range(1, x_max+2, 2)])
    cbar = plt.colorbar(shrink=0.5)
    cbar.set_label(clabel, labelpad=10,rotation=90)
    plt.show()


def plot_objects(df_tracks,channel,ylabel):
    """Plot objects in df_tracks. Funciton displays one chart per 
     object in df_tracks and plots the requested channel over the time frames """ 
    # create a figure with subplots for each row
    num_rows, num_cols = df_tracks[:,:,channel].shape
    fig, axes = plt.subplots(nrows=num_rows, ncols=1,figsize=(4, 8))
    # set x and y limits 
    x_min, x_max = 0, df_tracks.shape[1] - 1
    y_min, y_max = np.nanmin(df_tracks[:,:,channel]), np.nanmax(df_tracks[:,:,channel])
    # loop over the rows and plot each one separately
    for i in range(num_rows):
        axes[i].plot(df_tracks[i,:,channel])
        axes[i].set_title(f"Object {i+1}",fontsize=9)
        # axes[i].set_xlabel('Frame')
        axes[i].set_ylabel(ylabel)
        axes[i].set_xlabel('Frame')
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(0, y_max)
        axes[i].set_xticklabels([str(j) for j in range(1, x_max+2, 2)])
        # remove x tick labels
        if i < df_tracks.shape[0] - 1:
            axes[i].set_xticklabels([])
            axes[i].set_xlabel('')
    # adjust the spacing between subplots
    fig.tight_layout()
    # show the plot
    plt.show()
    

def show_object_overlay(filename,itiff,im_labeled,df_tracks):
    """ """
    tiffs  = skio.imread(filename, plugin="tifffile")
    img = tiffs[itiff]
    masks = im_labeled[:,:,itiff]
    edges = segmentation.find_boundaries(masks)
    # plot the TIFF image
    # itiff = 0
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.imshow(edges, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"Frame {itiff+1}")
    for i,row in enumerate(df_tracks[:,:,0]):
        x = df_tracks[i,itiff,1] # x coord
        y = df_tracks[i,itiff,2] # y coord
        if x > 0:
            label1 = '#' + str(i+1)
            label2 = f"({round(x)}, {round(y)})"
            ax.text(x+10, y, label1, color='red', fontsize=12)
            ax.text(x+10, y+30, label2, color='red', fontsize=10)
    plt.savefig(f'./tracking_{itiff}.png')
    plt.close()


def make_gif(tiffs_to_process,filename,itiff,im_labeled,df_tracks):
    gif = []
    for itiff in tiffs_to_process:
        # overlay image with mask edges
        show_object_overlay(filename,itiff,im_labeled,df_tracks)
        image = imageio.v2.imread(f'./tracking_{itiff}.png')
        # append overlay for animated gif
        gif.append(image)
    # Save as gif 
    imageio.mimsave('./example.gif', gif, fps = 1)     

