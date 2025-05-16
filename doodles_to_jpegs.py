# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020-2022, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ##========================================================

# allows loading of functions from the src directory
import sys, os, getopt, shutil
sys.path.insert(1, '../')
# from annotations_to_segmentations import *
# from image_segmentation import *
# from doodler_engine.annotations_to_segmentations import *
# from doodler_engine.image_segmentation import *


from glob import glob
import matplotlib.pyplot as plt
import skimage.io as io
from tqdm import tqdm

# from tkinter import Tk
# from tkinter.filedialog import askopenfilename, askdirectory
import plotly.express as px
import matplotlib

from numpy.lib.npyio import load

import numpy as np

###===========================================================
try:
    from my_defaults import *
    print("Your session defaults loaded")
except:
    from defaults import *

###===========================================================
# Change these <3
# direc = '/sciclone/home/ncrupert/dash_doodler/results'
# direc = '/sciclone/home/ncrupert/dash_doodler/results/results'
# direc = '/sciclone/home/ncrupert/dash_doodler/results/results_beach_pics'   #go back to this if it breaks
direc = '/sciclone/home/ncrupert/dash_doodler/results/results_exp2_120624'
classfile_dir = '/sciclone/home/ncrupert/dash_doodler/classes.txt'


# Here is a line from one of your older files. This DID recognize the files and open the folders.
#  glob_pattern_npz = dash_doodler_results_path + os.sep + '**' + os.sep + '*.npz'

###===========================================================

# Add glob here to open the folders of results so they are loose in results_exp_1

# source_dir = '/sciclone/home/ncrupert/dash_doodler/results'
# source_dir = '/sciclone/home/ncrupert/dash_doodler/results/results_beach_pics'           # go back to these two if it breaks
# destination_dir = '/sciclone/home/ncrupert/dash_doodler/results/results_beach_pics'
source_dir = '/sciclone/home/ncrupert/dash_doodler/results/results_exp2_120624'
destination_dir = '/sciclone/home/ncrupert/dash_doodler/results/results_exp2_120624'

# Globbing for npz

npz_files = glob(os.path.join(source_dir, '**/*.npz'), recursive=True)

# Globbing for png

png_files = glob(os.path.join(source_dir, '**/*.png'), recursive=True)

# Combining the lists of files to move them both at once
all_files = npz_files + png_files

# Move each file to the destination directory
for file_path in all_files:
    try:
        shutil.move(file_path, destination_dir)
        print(f'Moved: {file_path}')
    except Exception as e:
        print(f'Error moving {file_path}: {e}')


#==================================================================================================================

def make_dir(dirname):
    # check that the directory does not already exist
    if not os.path.isdir(dirname):
        # if not, try to create the directory
        try:
            os.mkdir(dirname)
        # if there is an exception, print to screen and try to continue
        except Exception as e:
            print(e)
    # if the dir already exists, let the user know
    else:
        print('{} directory already exists'.format(dirname))

def move_files(files, outdirec):
    for a_file in files:
        shutil.move(a_file, outdirec+os.sep+a_file.split(os.sep)[-1])


def make_jpegs():

    # Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    # direc = askdirectory(title='Select directory of results (annotations)', initialdir=os.getcwd()+os.sep+'results')

    # MAKE THIS A DIRECTORY wITH AS MANY RESULTS AS YOU wANT

# Do we also need to add a recursive glob here? I think this is why we can't find the things later on!!!!!!
    
    files = sorted(glob(direc+ '/*.npz'))   #ORIGINAL LINE

    # files = sorted(glob(direc+ 'results', '**', '/*.npz'))
    # files = sorted(glob.glob(direc + 'results/**/*.npz', recursive=True))
    # print(files)
    # files_npz = glob(os.path.join(direc, 'results', '**' + '/*.npz'))
    # print(files_npz)
    # files_png = glob(os.path.join(direc, 'results', '**' + '/*.png'))
    # print(files_png)
    # lafiles = glob(os.path.join(direc, 'results', '**', '*_label.jpg'), recursive=True)
    # files = sorted(glob(direc + '/*.npz'), recursive=True)
    print("Files: "+str(len(files)))
    files = [f for f in files if 'labelgen' not in f]
    files = [f for f in files if '4zoo' not in f]

    #### loop through each file
    for counter, anno_file in enumerate(files):

        # print("Working on %s" % (file))
        print("Working on %s" % (anno_file))

        # try:
        #     dat = np.load(anno_file)
        # except:
        #     dat = np.load(anno_file, allow_pickle=True)
        #
        # finally:
        #     print("Could not load"+anno_file)
        #     pass

        data = dict()
        with load(anno_file, allow_pickle=True) as dat:
            #create a dictionary of variables
            #automatically converted the keys in the npz file, dat to keys in the dictionary, data, then assigns the arrays to data
            for k in dat.keys():
                data[k] = dat[k]
            del dat

        # if 'dat' in locals():
        #
        #     data = dict()
        #     for k in dat.keys():
        #         try:
        #             data[k] = dat[k]
        #         except Exception as e:
        #             print(e)
        #             pass
        #     del dat

        try:
            classes = data['classes']
        except:
            # print('No classes found in settings! Using defaults of "water" and "land"')
            # classes = ['water', 'land']
            # Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
            # classfile = askopenfilename(title='Select file containing class (label) names', filetypes=[("Pick classes.txt file","*.txt")])

## CHANGE THIS ONE TOO
#             
            classfile = classfile_dir = '/sciclone/home/ncrupert/dash_doodler/classes.txt'

            with open(classfile) as f:
                classes = f.readlines()

        NCLASSES  = len(classes)
        class_string = '_'.join([c.strip() for c in classes])

        #Make the original images as jpg
        if 'orig_image' in data.keys():
            im = np.squeeze(data['orig_image'].astype('uint8'))[:,:,:3]
        else:
            if data['image'].shape[-1]==4:
                im=np.squeeze(data['image'].astype('uint8'))[:,:,:-1]
                band4=np.squeeze(data['image'].astype('uint8'))[:,:,-1]
            else:
                im = np.squeeze(data['image'].astype('uint8'))[:,:,:3]

        io.imsave(anno_file.replace('.npz','.jpg'),
                  im, quality=100, chroma_subsampling=False)

        if 'band4' in locals():
                io.imsave(anno_file.replace('.npz','_band4.jpg'),
                          band4, quality=100, chroma_subsampling=False)
                del band4


        #Make the label as jpg
        l = np.argmax(data['label'],-1).astype('uint8')+1
        nx,ny = l.shape
        lstack = np.zeros((nx,ny,NCLASSES))
        lstack[:,:,:NCLASSES] = (np.arange(NCLASSES) == l[...,None]-1).astype(int) #one-hot encode
        l = np.argmax(lstack,-1).astype('uint8')

        io.imsave(anno_file.replace('.npz','_label.jpg'),
                  l, quality=100, chroma_subsampling=False, check_contrast=False)

        # if 'classes' not in locals():
        #
        #     try:
        #         classes = data['classes']
        #     except:
        #         Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        #         classfile = askopenfilename(title='Select file containing class (label) names', filetypes=[("Pick classes.txt file","*.txt")])
        #
        #         with open(classfile) as f:
        #             classes = f.readlines()

        class_label_names = [c.strip() for c in classes]

        NUM_LABEL_CLASSES = len(class_label_names)

        if NUM_LABEL_CLASSES<=10:
            class_label_colormap = px.colors.qualitative.G10
        else:
            class_label_colormap = px.colors.qualitative.Light24

        # we can't have fewer colors than classes
        assert NUM_LABEL_CLASSES <= len(class_label_colormap)

        # colormap = [
        #     tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        #     for h in [c.replace("#", "") for c in class_label_colormap]
        # ]

        cmap = matplotlib.colors.ListedColormap(class_label_colormap[:NUM_LABEL_CLASSES+1])
        # cmap2 = matplotlib.colors.ListedColormap(['#000000']+class_label_colormap[:NUM_LABEL_CLASSES])

        #Make an overlay
        plt.imshow(im)
        plt.imshow(l, cmap=cmap, alpha=0.5, vmin=0, vmax=NCLASSES)
        plt.axis('off')
        plt.savefig(anno_file.replace('.npz','_overlay.png'), dpi=200, bbox_inches='tight')


        #Make an doodles overlay
        # plt = matplotlib.pyplot
        try:
            doodles = data['doodles'].astype('float')
            doodles[doodles<1] = np.nan
            doodles -= 1
            plt.imshow(im)
            plt.imshow(doodles, cmap=cmap, alpha=0.5, vmin=0, vmax=NCLASSES)
            plt.axis('off')
            plt.savefig(anno_file.replace('.npz','_doodles.png'), dpi=200, bbox_inches='tight')
        except:
            print('no doodles in {}'.format(anno_file))
        del im

        plt.close('all')

# # So we can run the rest if the directories already exist. make_dir exists at the top already but was still giving issues? Trying the above one rn.
def make_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


# # We want to move the files. Let's use the one at the top of the page and comment this out for now.
def move_files(file_list, outdir):
    for file in file_list:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        os.rename(file, os.path.join(outdir, os.path.basename(file)))


# Making a new test directory inside of results so we don't use EVERYTHING 

# results_exp_1 = '/sciclone/home/ncrupert/dash_doodler/results/results_exp_1'
results_beach_pics = '/sciclone/home/ncrupert/dash_doodler/results/results_beach_pics'
# results_exp_1 = os.path.join(direc, 'results', 'results_exp_1')
# print(f"results_exp_1 directory: {results_exp_1}")
print(f"results_beach_pics directory: {results_beach_pics}")

# Commented out below is the original code

# #make directories for labels and images, to make transition to zoo easy

imdir = os.path.join(direc, 'images')   # original
ladir = os.path.join(direc, 'labels')
overdir = os.path.join(direc, 'overlays')
doodlesdir = os.path.join(direc, 'doodles')

# imdir = os.path.join(results_exp_1, 'images')
# ladir = os.path.join(results_exp_1, 'labels')
# overdir = os.path.join(results_exp_1, 'overlays')
# doodlesdir = os.path.join(results_exp_1, 'doodles')

make_dir(imdir)
make_dir(ladir)
make_dir(overdir)
make_dir(doodlesdir)


# Commented out is the original code, BEFORE implementing a recursive glob

lafiles = glob(direc+'/*_label.jpg')    #this is the original
#lafiles = glob(doodler_results_opened + os.sep + '**' + '/*_label.jpg')
outdirec = os.path.normpath(direc + os.sep+'labels')
move_files(lafiles, outdirec)
print(f"Found and moved {len(lafiles)} label files.")

doodlefiles = glob(direc+'/*_doodles.png')
outdirec = os.path.normpath(direc + os.sep+'doodles')
move_files(doodlefiles, outdirec)
print(f"Found and moved {len(doodlefiles)} doodle files.")

imfiles = glob(direc+'/*.jpg')
outdirec = os.path.normpath(direc + os.sep+'images')
move_files(imfiles, outdirec)
print(f"Found and moved {len(doodlefiles)} doodle files.")

ovfiles = glob(direc+'/*_overlay.png')
outdirec = os.path.normpath(direc + os.sep+'overlays')
move_files(ovfiles, outdirec)
print(f"Found and moved {len(doodlefiles)} doodle files.")


# WITH recursive globbing

# lafiles = glob(os.path.join(direc, 'results', '**', '*_label.jpg'), recursive=True)
# outdirec = os.path.normpath(ladir)
# move_files(lafiles, outdirec)
# print(f"Found and moved {len(lafiles)} label files.")

# doodlefiles = glob(os.path.join(direc, 'results', '**', '*_doodles.png'), recursive=True)
# outdirec = os.path.normpath(doodlesdir)
# move_files(doodlefiles, outdirec)
# print(f"Found and moved {len(doodlefiles)} doodle files.")

# imfiles = glob(os.path.join(direc, 'results', '**', '*.jpg'), recursive=True)
# outdirec = os.path.normpath(imdir)
# move_files(imfiles, outdirec)
# print(f"Found and moved {len(doodlefiles)} doodle files.")

# ovfiles = glob(os.path.join(direc, 'results', '**', '*_overlay.png'), recursive=True)
# outdirec = os.path.normpath(overdir)
# move_files(ovfiles, outdirec)
# print(f"Found and moved {len(doodlefiles)} doodle files.")



###==================================================================
#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:") #m:p:l:")
    except getopt.GetoptError:
        print('======================================')
        print('python gen_images_and_labels_4_zoo.py') #
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('======================================')
            print('Example usage: python gen_images_and_labels_4_zoo.py') #, save mode mode 1 (default, minimal), make plots 0 (no), print labels 0 (no)
            print('======================================')
            sys.exit()
    #ok, dooo it
    make_jpegs()
