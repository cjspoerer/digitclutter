'''
Contains the clutter class
'''

from digitclutter.utils import shlex_cmd

class Clutter:
    '''
    A class that contains all of the paramters for a single clutter image
    '''
    def __init__(self, clutter_sample, composition_type='occlusion'):
        '''
        Initialises a Clutter object. It is assumed that all Character objects
        have the same image size

        Args:
            clutter_sample:   a sequence of Character objects
            composition_type: only occlusion is currently supported
        '''
        self.back_color = '#777777' # Background colour
        self.n_characters = len(clutter_sample)
        self.composition_type = composition_type
        self.clutter_sample = clutter_sample
        self.image_size = clutter_sample[0].image_size
        self.fname = None

    def get_character_list(self):
        '''
        Returns a list of all of the identities of chracters present in the
        sample
        '''
        character_list = [c.identity for c in self.clutter_sample]
        return character_list

    def render_occlusion(self, fname=None, thread_limit=None):
        '''
        Renders the corresponding image using occlusion. Functions by building
        a sequence of strings that are submitted as ImageMagick commands.
        '''
        if fname is None and self.fname is None:
            raise ValueError('fname is not defined')
        elif fname is None:
            fname = self.fname

        # Initialise the background
        image_size_str = '{0}x{1}'.format(self.image_size[0], self.image_size[1])
        kwargs = {'bg_colour':'rgba(119,119,119,1.0)', 'image_size_str':image_size_str}
        image_cmd = 'convert xc:{bg_colour} -resize {image_size_str}! '.format(**kwargs)

        # Build a layer of each character by cycling through the letters
        for char in self.clutter_sample:
            x_pos, y_pos = char.position[0], char.position[1]

            # Make the string for face and edge colours
            face_col_str = 'rgba({0},{1},{2},{3})'.format(char.face_colour[0],
                                                          char.face_colour[1],
                                                          char.face_colour[2],
                                                          char.face_colour[3])
            edge_col_str = 'rgba({0},{1},{2},{3})'.format(char.edge_colour[0],
                                                          char.edge_colour[1],
                                                          char.edge_colour[2],
                                                          char.edge_colour[3])

            # First write the command for the outline of the character
            kwargs = {'font':char.font, 'fontsize':char.fontsize,
                      'face_col':'rgba(0.0,0.0,0.0,0.0)',
                      'edge_col':edge_col_str, 'linewidth':char.linewidth,
                      'x':x_pos, 'y':y_pos, 'xscale':char.size_scale[0],
                      'yscale':char.size_scale[1], 'identity':char.identity}
            outline_cmd = '''-draw "gravity Center font {font}
                    font-size {fontsize!r} fill {face_col}
                    stroke {edge_col} stroke-width {linewidth!r}
                    scale {xscale!r},{yscale!r} text {x!r},{y!r}
                    {identity!r}" '''.format(**kwargs)

            # The write the command for the face of the character
            kwargs = {'font':char.font, 'fontsize':char.fontsize,
                      'face_col':face_col_str, 'edge_col':'rgba(0.0,0.0,0.0,0.0)',
                      'linewidth':char.linewidth, 'x':x_pos, 'y':y_pos,
                      'xscale':char.size_scale[0], 'yscale':char.size_scale[1],
                      'identity':char.identity}
            face_cmd = '''-draw "gravity Center font {font}
                        font-size {fontsize!r} fill {face_col}
                        stroke {edge_col} stroke-width {linewidth!r}
                        scale {xscale!r},{yscale!r} text {x!r},{y!r}
                        {identity!r}" '''.format(**kwargs)
            image_cmd += outline_cmd + face_cmd

        # Add the command to save the image as a BMP
        image_cmd += 'BMP3:{0}.bmp'.format(fname)

        # Add a thread limit if necessary
        if thread_limit is not None:
            image_cmd += ' -limit thread {0}'.format(thread_limit)

        # Submit the image command
        shlex_cmd(image_cmd)

    def render_clutter(self, fname=None, thread_limit=None):
        '''
        Renders the image for the Clutter object

        Args:
            fname:        a str giving the desired path to save the file. Will
                          default to self.fname if not given.
            thread_limit: an int giving the number of threads to be used when rendering
        '''
        if fname is None and self.fname is None:
            raise ValueError('Both fname and self.fname are undefined')
        elif fname is None:
            fname = self.fname

        if self.composition_type == 'occlusion':
            self.render_occlussion(fname, thread_limit)
        else:
            raise ValueError('Composition type {0} is not recognised'\
            .format(self.composition_type))

