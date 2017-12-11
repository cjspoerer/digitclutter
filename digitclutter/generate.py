'''
Contains functions for generating stimuli
'''

import os
import random
from warnings import warn
from shutil import rmtree
import numpy as np
from PIL import Image
from digitclutter.character import Character
from digitclutter.clutter import Clutter
from digitclutter.utils import shlex_cmd, DIGITS
from digitclutter.io import name_files

def truncated_normal_2d(minimum, maximum, mean, covariance):
    '''
    Draws a sample from a 2d truncated normal distribution
    '''
    while True:
        sample = np.random.multivariate_normal(mean, covariance, 1)
        if np.all(minimum <= sample) and np.all(sample <= maximum):
            return np.squeeze(sample)

def sample_clutter(**kwargs):
    '''
    Returns a list of character objects that can be used to initialise a clutter
    object.

    kwargs:
        image_size:         as a sequence [x-size, y-size]
        n_letters:          an int for the number of characters present in each image
        font_set:           a list of TrueType fonts to be sampled from,
                            e.g. ['helvetica-bold']
        character_set:      a sequence of characters to sampled from
        face_colour_set:    a list of RGBA sequences to sample from
        edge_colour_set:    a list of RGBA sequences to sample from
        linewidth:          an int giving the width of character edges in pixels
        offset_sample_type: the distribution that offsets are drawn, 'uniform'
                            or 'gaussian'
        offset_mean:        a sequence that is the mean of the two-dimensional
                            Gaussian that the offsets are sampled from
        offset_cov:         if offset_sample_type is 'gaussian', is is the 2x2
                            covariance matrix, if offset_sample_type is
                            'uniform' then it is the parameters of the uniform
                            distribution [[x-low,x-high],[y-low,y-high]]
        size_sample_type:   the distribution that character scalings are drawn
                            from, 'gaussian' or 'truncnorm'
        size_mean:          a sequence that is the mean of the two-dimensional
                            Gaussian that the scaling coefficients are sampled from
        size_cov:           if size_sample_type is 'gaussian', is is the 2x2
                            covariance matrix, if size_sample_type is 'uniform'
                            then it is the parameters of the uniform
                            distribution [[x-low,x-high],[y-low,y-high]]
        size_min:           a sequence giving minimum scaling in each dimension
                            [x-min, y-min], only used for 'truncnorm'
        size_max:           a sequence giving minimum scaling in each dimension
                            [x-max, y-max], only used for 'truncnorm'
        fontsize:           pointsize of character as an integer

    Returns:
        clutter_sample: a list of Character objects
    '''

    image_size = kwargs.get('image_size', (512, 512))
    n_letters = kwargs.get('n_letters', 1)
    font_set = kwargs.get('font_set', ['helvetica-bold'])
    character_set = kwargs.get('character_set', DIGITS)
    face_colour_set = kwargs.get('face_colour_set', [(0, 0, 0, 1.0)])
    edge_colour_set = kwargs.get('edge_colour_set', [(255, 255, 255, 1.0)])
    linewidth = kwargs.get('linewidth', 20)
    offset_sample_type = kwargs.get('offset_sample_type', 'uniform')
    offset_mean = kwargs.get('offset_mean', (0, 0.076))
    offset_cov = kwargs.get('offset_cov', ((-0.29, 0.29), (-0.19, 0.19)))
    size_sample_type = kwargs.get('size_sample_type', 'truncnorm')
    size_min = kwargs.get('size_min', (0.7, 0.7))
    size_max = kwargs.get('size_max', (1.0, 1.0))
    size_mean = kwargs.get('size_mean', (1, 1))
    size_cov = kwargs.get('size_cov', ((0, 0), (0, 0)))
    fontsize = kwargs.get('fontsize', 384)

    # Sample characters without replacement
    characters = np.random.choice(character_set, n_letters,
                                  replace=False)
    # Initialise the clutter sample list
    clutter_sample = [None] * n_letters

    # Draw samples to get the parameters for individual characters
    char_opt = {}
    char_opt['image_size'] = image_size
    char_opt['linewidth'] = linewidth
    char_opt['fontsize'] = fontsize
    for i in range(n_letters):
        char_opt['identity'] = characters[i]
        char_opt['face_colour'] = random.choice(face_colour_set)
        char_opt['edge_colour'] = random.choice(edge_colour_set)
        char_opt['font'] = random.choice(font_set)

        # Sample the offset
        if tuple(offset_cov) == ((0, 0), (0, 0)):
            char_opt['offset'] = offset_mean
        elif offset_sample_type == 'uniform':
            x_offset = offset_mean[0] + np.random.uniform(offset_cov[0][0],
                                                       offset_cov[0][1])
            y_offset = offset_mean[1] + np.random.uniform(offset_cov[1][0],
                                                       offset_cov[1][1])
            char_opt['offset'] = [x_offset, y_offset]
        elif offset_sample_type == 'gaussian':
            char_opt['offset'] = np.random.multivariate_normal(offset_mean,
                                                               offset_cov)
        else:
            raise ValueError('{0} not a valid offset sampling type'\
            .format(offset_sample_type))

        # Sample the size coefficient
        if tuple(size_cov) == ((0, 0), (0, 0)):
            char_opt['size_scale'] = size_mean
        elif size_sample_type == 'gaussian':
            size_sample = np.random.multivariate_normal(size_mean, size_cov)
            char_opt['size_scale'] = (max(0, size_sample[0]), max(0, size_sample[1]))
        elif size_sample_type == 'truncated_normal_2d':
            size_sample = truncated_normal_2d(size_min, size_max, size_mean, size_cov)
            char_opt['size_scale'] = (max(0, size_sample[0]), max(0, size_sample[1]))
        else:
            raise ValueError('{0} is not a valid size sampling type'\
            .format(size_sample_type))

        clutter_sample[i] = Character(char_opt)

    return Clutter(clutter_sample)

def make_debris_templates(**kwargs):
    '''
    Makes debris templates inclduing all possible character font combinations

    kwargs:
        character_set: a sequence of characters to generate templates from
        font_set:      a list of TrueType fonts to generate templates from,
                       e.g. ['helvetica-bold']
        fontsize:      fontsize as an integer
        linewidth:     linewidth as an integer
        size_mean:     a sequence that contains the scaling coefficient for
                       each dimension, [x_scale, y_scale]
        image_size:    size that images are designed at, as a sequence [x-size, y-size]
        image_resize:  size that images are saved at, as a sequence [x-size, y-size]
        wdir:          path to working directory, should already exist
    '''
    character_set = kwargs.get('character_set')
    font_set = kwargs.get('font_set')
    fontsize = kwargs.get('fontsize')
    linewidth = kwargs.get('linewidth')
    size_mean = kwargs.get('size_mean')
    image_size = kwargs.get('image_size')
    image_resize = kwargs.get('image_resize')
    wdir = kwargs.get('wdir')

    n_templates = len(character_set) * len(font_set)
    debris_templates = [None] * n_templates
    template_fnames = name_files(wdir, prefix='debris_char', n_images=n_templates)
    mask_fnames = name_files(wdir, prefix='debris_mask', n_images=n_templates)

    k = 0
    for font in font_set:
        for char in character_set:
            # There is some numerical imprecision which means that there is a
            # small chance that the trimmed images will not match on between
            # mask and the image. Reattempting a small number of times should
            # avoid this problem
            success, attempts = False, 0
            while not success and attempts < 2:
                # Generate the template image
                image_size_str = '{0}x{1}'.format(image_size[0], image_size[1])
                image_cmd = 'magick xc:rgba(119,119,119,1.0) -resize '\
                +image_size_str+'! +antialias '
                # Write the outline command
                im_kwargs = {'font':font, 'fontsize':fontsize,
                             'face_col':'rgba(0.0,0.0,0.0,1.0)',
                             'edge_col':'rgba(255,255,255,1.0)',
                             'linewidth':linewidth, 'xscale':size_mean[0],
                             'yscale':size_mean[1], 'identity':char}
                outline_cmd = '-draw "gravity Center font {font} \
                               font-size {fontsize!r} fill rgba(0.0,0.0,0.0,0.0) \
                               stroke {edge_col!r} stroke-width {linewidth!r} \
                               scale {xscale!r},{yscale!r} \
                               text 0,0 {identity!r}" '.format(**im_kwargs)
                # Write the face command
                face_cmd = '-draw "gravity Center font {font} \
                            font-size {fontsize!r} fill {face_col!r} \
                            stroke rgba(0.0,0.0,0.0,0.0) stroke-width {linewidth!r} \
                            scale {xscale!r},{yscale!r} \
                            text 0,0 {identity!r}" '.format(**im_kwargs)
                image_cmd += outline_cmd + face_cmd + 'BMP3:{0!r}'.format(template_fnames[k]+'.bmp')
                shlex_cmd(image_cmd)
                # Resize and trim the image
                resize_cmd = 'magick {0!r} -scale {1}x{2} BMP3:{0!r}'\
                .format(template_fnames[k]+'.bmp', image_resize[0], image_resize[1])
                shlex_cmd(resize_cmd)
                trim_cmd = 'magick {0!r} -trim +repage {0!r}'.format(template_fnames[k]+'.bmp')
                shlex_cmd(trim_cmd)

                # Generate the mask
                mask_cmd = 'magick xc:rgba(0,0,0,1.0) -resize '+image_size_str\
                +'! +antialias '
                # Write the mask command
                im_kwargs = {'font':font, 'fontsize':fontsize,
                             'face_col':'rgba(255,255,255,1.0)',
                             'edge_col':'rgba(255,255,255,1.0)',
                             'linewidth':linewidth, 'identity':char,
                             'xscale':size_mean[0], 'yscale':size_mean[1]}

                mask_cmd += '-draw "gravity Center font {font} \
                             font-size {fontsize!r} fill {face_col!r} \
                             stroke {edge_col!r} stroke-width {linewidth!r} \
                             scale {xscale!r},{yscale!r} \
                             text 0,0 {identity!r}" '.format(**im_kwargs)
                mask_cmd += 'BMP3:{0!r}'.format(mask_fnames[k]+'.bmp')
                shlex_cmd(mask_cmd)
                # Resize and trim the image
                resize_cmd = 'magick {0!r} -scale {1}x{2} BMP3:{0!r}'\
                .format(mask_fnames[k]+'.bmp', image_resize[0], image_resize[1])
                shlex_cmd(resize_cmd)
                trim_cmd = 'magick {0!r} -trim +repage {0!r}'.format(mask_fnames[k]+'.bmp')
                shlex_cmd(trim_cmd)

                # Open and the mask and the template image
                temp_template = np.array(Image.open(template_fnames[k]+'.bmp'))
                temp_template = temp_template.mean(axis=2) # Greyscale
                temp_mask = np.array(Image.open(mask_fnames[k]+'.bmp'))
                temp_mask = temp_mask.mean(axis=2) # Greyscale

                # Check the size of the template and mask match
                if temp_template.shape != temp_mask.shape and attempts == 0:
                    attempts += 1
                    warn('Size of image and mask do not match. Attempt'+str(attempts)+'of 2')
                    continue
                elif temp_template.shape != temp_mask.shape:
                    raise ValueError('Size of mask and template do not match. Attempt 2 of 2')
                else:
                    success = True

                # Binarise the mask
                cond_list = [temp_mask >= 127, temp_mask < 127]
                choice_list = [1, 0]
                temp_mask = np.select(cond_list, choice_list)

                debris_templates[k] = np.transpose(
                    np.array((temp_template, temp_mask)), (1, 2, 0))
                
                k += 1

    return debris_templates


def make_debris(n_images, wdir='./temp_workspace', **kwargs):
    '''
    Generates an array of debris that can be added to images

    Args:
        n_images: number of instances of debris to be generated as an int
        wdir:     path to working directory, it should not already exist and
                  will be deleted afterwards

    kwargs:
        character_set:   a sequence of characters to sampled from
        font_set:        a list of TrueType fonts to be sampled from,
                         e.g. ['helvetica-bold']
        image_size:      size that images are designed at, as a sequence [x-size, y-size]
        image_save_size: size that images are saved at, as a sequence [x-size, y-size]
        debris_size:     max and min size of each fragment constituting the debris,
                         as a sequence [size_min, size_max]
        n_debris:        max and min number of fragmetns in debris, as a
                         sequence, [n_debris_min, n_debris_max]
        fontsize:        fontsize as an integer
        linewidth:       linewidth as an integer
        size_mean:       a sequence that contains the scaling coefficient for
                         each dimension, [x_scale, y_scale]

    returns:
        debris_arr: an array containing the debris generated
    '''

    # Get the kwargs
    character_set = kwargs.get('character_set', DIGITS)
    font_set = kwargs.get('font_set', ['helvetica-bold'])
    image_size = kwargs.get('image_size', (512, 512))
    image_save_size = kwargs.get('image_save_size', (32, 32))
    debris_size = kwargs.get('debris_size', [6, 9])
    n_debris = kwargs.get('n_debris', [50, 51])
    fontsize = kwargs.get('fontsize', 384)
    size_mean = kwargs.get('size_mean', [1.0, 1.0])
    linewidth = kwargs.get('linewidth', 20)

    # Set-up wdir
    wdir = os.path.abspath(wdir)
    print('Using '+wdir+' as the working directory')
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    else:
        raise FileExistsError('The defined working directory'+wdir+' exists. \
        Please use another one')

    # First make the debris templates
    deb_kwargs = {
        'character_set':character_set,
        'font_set':font_set,
        'fontsize':fontsize,
        'linewidth':linewidth,
        'size_mean':size_mean,
        'image_size':image_size,
        'image_resize':image_save_size,
        'wdir':wdir,
    }
    debris_templates = make_debris_templates(**deb_kwargs)

    # Initialise debris array
    debris_arr = np.full((n_images, image_save_size[0], image_save_size[1], 2),
                         119, dtype=np.uint8)
    debris_arr[:, :, :, 1] = 0
    for i in range(n_images):
        # Select number of debris to appear in this instance
        n_debris_i = np.random.randint(low=n_debris[0], high=n_debris[1])

        for j in range(n_debris_i):
            # Select the width of the debris for this fragment
            debris_size_j = np.random.randint(low=debris_size[0],
                                              high=debris_size[1]+1)

            # Select the character to provide the fragment
            deb_char = debris_templates[np.random.randint(len(debris_templates))]

            # Check that the size for the debris is not bigger than the character
            if deb_char.shape[0] <= debris_size_j and deb_char.shape[1] <= debris_size_j:
                new_deb_size = np.min(deb_char.shape) - 1
                warn('Debris size {0} bigger than character size {1}. Using {2}\
                      instead'.format(debris_size_j, deb_char.shape, new_deb_size))
                debris_size_j = new_deb_size
                if isinstance(debris_size, tuple):
                    debris_size = list(debris_size)
                debris_size[1] = debris_size_j

            # Set the possible start and end points for taking the crop from
            # the character that will make the fragment
            if deb_char.shape[0] - debris_size_j == 0:
                crop_x_start = 0
            else:
                crop_x_start = np.random.randint(deb_char.shape[0] - debris_size_j)
            if deb_char.shape[1] - debris_size_j == 0:
                crop_y_start = 0
            else:
                crop_y_start = np.random.randint(deb_char.shape[1] - debris_size_j)

            # Get the crop
            debris_ij = deb_char[crop_x_start:crop_x_start+debris_size_j,
                                 crop_y_start:crop_y_start+debris_size_j]

            # Set the window where the debris will be placed
            if image_save_size[0] - debris_size_j == 0:
                deb_x_start = 0
            else:
                deb_x_start = np.random.randint(image_save_size[0] - debris_size_j)
            if image_save_size[1] - debris_size_j == 0:
                deb_y_start = 0
            else:
                deb_y_start = np.random.randint(image_save_size[1] - debris_size_j)

            # Set the window where the debris will be placed
            x_0, x_1 = deb_x_start, deb_x_start+debris_size_j
            y_0, y_1 = deb_y_start, deb_y_start+debris_size_j
            deb_window = debris_arr[i, x_0:x_1, y_0:y_1, :]

            # Use the binary mask to select the "non-background" parts of the
            # character
            cond_list = [debris_ij[:, :, 1] == 0, True]
            choice_list = [deb_window[:, :, 0], debris_ij[:, :, 0]]
            debris_arr[i, x_0:x_1, y_0:y_1, 0] = np.select(cond_list, choice_list)
            choice_list = [0, 1]
            debris_arr[i, x_0:x_1, y_0:y_1, 1] = np.select(cond_list, choice_list)

    rmtree(wdir)
    return debris_arr

def add_debris(clutter, debris):
    '''
    Add previously generated debris arrays to images
    '''

    clutter = np.squeeze(clutter, axis=3)
    clutter_with_debris = np.empty(clutter.shape)

    for i in range(clutter.shape[0]):
        cond_list = [debris[i, :, :, 1] == 0, True]
        choice_list = [clutter[i, :, :], debris[i, :, :, 0]]
        clutter_with_debris[i] = np.select(cond_list, choice_list)

    return np.expand_dims(clutter_with_debris, axis=3).astype(np.uint8)

def get_character_masks():
    raise NotImplementedError

def calculate_occlussion():
    raise NotImplementedError

