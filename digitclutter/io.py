'''
Contains input/output functions for clutter stimuli
'''
import os
import csv
import time
from shutil import rmtree
import numpy as np
from PIL import Image
from scipy.io import savemat
from digitclutter.character import Character
from digitclutter.clutter import Clutter
from digitclutter.utils import shlex_cmd, DIGITS

def name_files(save_dir, clutter_list=None, n_images=None, prefix='image'):
    '''
    Gives file names to a list of Clutter objects of the form, prefix_0000,
    prefix_0001, ..., prefix_9999

    save_path:    path to the directory where the images will be saved. It will
                  create the directory if it does not already exist
    clutter_list: a sequence of Clutter objects, if none (default), will return
                  list of paths
    n_images:     number of images to generate file names for, must be given if
                  clutter_list is None
    prefix:       a str giving the prefix for image file names
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not clutter_list is None:
        n_images = len(clutter_list)
    else:
        fname_list = [None] * n_images

    n_zeros = len(str(n_images-1))

    # Create a file name for each instance
    for i in range(n_images):
        image_name = '{0}_{1}'.format(prefix, str(i).zfill(n_zeros))
        if clutter_list is None:
            fname_list[i] = os.path.abspath(os.path.join(save_dir, image_name))
        else:
            clutter_list[i].fname = os.path.abspath(os.path.join(save_dir, image_name))

    if clutter_list is None:
        return fname_list
    else:
        return clutter_list

def save_image_set(clutter_list, csv_fname):
    '''
    Saves the list of Clutter objects to a CSV file.

    Args:
        clutter_list: a list of Clutter objects
        csv_fname:    a str giving a path to save the csv
    '''
    with open(csv_fname, 'w') as csvfile:
        # Open the csv file
        fwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        for clutter in clutter_list:
            # Make a list containing all of the parameters for a single clutter
            clutter_chars = []
            for char in clutter.clutter_sample:
                clutter_chars += [char.identity] + char.offset\
                                 + list(char.size_scale) + [char.font, char.fontsize]\
                                 + list(char.face_colour) + list(char.edge_colour)\
                                 + [char.linewidth]
            # Add the parameters for the whole image and save the row
            fwriter.writerow([clutter.fname, clutter.composition_type, clutter.n_characters] 
                             + list(clutter.image_size) + clutter_chars)

def read_image_set(csv_fname):
    '''
    Reads a list of Clutter objects from a csv file saved with save_image_set.

    Args:
        csv_fname: a str with the path to the csv file

    Returns:
        clutter_list: a list of Clutter objects
    '''
    with open(csv_fname) as csv_file:
        freader = csv.reader(csv_file, delimiter=',', quotechar='|')

        # Iterate through each instance of Clutter
        clutter_list = []
        char_opt = {}
        for row in freader:
            n_chars = int(row[2])
            composition_type = row[1]
            # Initialise the clutter sample
            clutter_sample = [None] * n_chars
            # Iterate through each character in the clutter
            for i in range(n_chars):
                char_opt['image_size'] = (float(row[3]), float(row[4]))
                char_opt['identity'] = row[i*16+5]
                char_opt['offset'] = [float(row[i*16+6]), float(row[i*16+7])]
                char_opt['size_scale'] = (float(row[i*16+8]), float(row[i*16+9]))
                char_opt['font'] = row[i*16+10]
                char_opt['fontsize'] = float(row[i*16+11])
                char_opt['face_colour'] = (int(row[i*16+12]), int(row[i*16+13]),
                                           int(row[i*16+14]), float(row[i*16+15]))
                char_opt['edge_colour'] = (int(row[i*16+16]), int(row[i*16+17]),
                                           int(row[i*16+18]), float(row[i*16+19]))
                char_opt['linewidth'] = int(row[i*16+20])
                clutter_sample[i] = Character(char_opt)
            # Add the clutter that has been read to the clutter list
            clutter_list += [Clutter(clutter_sample, composition_type)]

    return clutter_list

def save_images_as_mat(mat_fname, clutter_list, image_save_size, fname_list=None,
                       character_set=DIGITS, grayscale=True, wdir='./temp_workspace'):
    '''
    Saves a mat file containing the images and labels. Labels are in the format
    of integers or binary vectors

    Args:
        mat_fname:       a str giving the path to save the mat file to
        image_save_size: size of the images that are saved
        fname_list:      contains the list of paths to image files if they have
                         already been rendered, otherwise, images are rendered
                         first
        grayscale:       a bool indicating whether to convert images to grayscale
        wdir:            working directory for modifying images, should not
                        already exist and will be deleted afterwards
    '''
    n_images = len(clutter_list)

    # Create a key for each of the possible characters
    character_key = {char: i for i, char in enumerate(character_set)}

    # Convert the wdir to an absolute path and set up working directory
    wdir = os.path.abspath(wdir)
    print('Using '+wdir+' as the working directory')
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    else:
        raise FileExistsError('The defined working directory'+wdir+' exists. \
        Please use another one')

    # Render the images if they are not give in fname_list
    if fname_list is None:
        print('Rendering images...')
        orig_images_dir = os.path.join(wdir, 'original')
        os.mkdir(orig_images_dir)
        clutter_list = name_files(orig_images_dir, clutter_list=clutter_list)

        # Render images
        fname_list = [None] * n_images
        start_time = time.time()
        for i, clutter in enumerate(clutter_list):
            clutter.render_letter_clutter()
            fname_list[i] = clutter.fname
            if i == 0:
                render_time = (time.time() - start_time) * n_images
                eta = time.time() + render_time
                print('Estimated time to finish rendering {0}'.format(eta))

    # Resize the images
    print('Resizing the images')
    resized_dir = os.path.join(wdir, 'resized')
    os.mkdir(resized_dir)
    resize_fname_list = name_files(resized_dir, n_images=n_images)
    for i in range(n_images):
        resize_cmd = 'convert {0}.bmp -scale {1}x{2} BMP3:{3}.bmp'.format(
            fname_list[i], image_save_size[0], image_save_size[1], resize_fname_list[i])
        shlex_cmd(resize_cmd)

    # Generate image array
    print('Generating image arrays')
    images = np.zeros((n_images, image_save_size[0], image_save_size[1], 3))
    for i in range(n_images):
        images[i] = np.array(Image.open(resize_fname_list[i]+'.bmp'))
    if grayscale:
        images = images.mean(axis=3, keepdims=True)

    # Generate target arrays
    print('Generating target arrays')
    max_chars = np.max([clutter.n_characters for clutter in clutter_list])
    targets = np.zeros((n_images, max_chars))
    binary_targets = np.zeros((n_images, len(character_set)))

    for i, clutter in enumerate(clutter_list):
        char_list = clutter.get_character_list()
        targets[i] = [character_key[char] for char in char_list]
        cond_list = [np.in1d(character_set, char_list), True]
        choice_list = [1, 0]
        binary_targets[i] = np.select(cond_list, choice_list)

    # Save the mat file
    savemat(mat_fname, {'images':images, 'targets':targets,
                        'binary_targets':binary_targets})
    print('Images and target arrays saved to '+mat_fname)

    # Remove wdir and contents
    rmtree(wdir)

    return {'images':images, 'targets':targets, 'binary_targets':binary_targets}


