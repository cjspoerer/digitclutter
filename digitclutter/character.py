'''
Contains the character class
'''

class Character:
    '''
    Contains all parameters for a given character
    '''
    def __init__(self, opt):
        '''
        Initialises character class.

        opt should contain the following keys:
            'identity':    a str containing a single character, e.g. '4'
            'face_colour': a sequence (R,G,B,A). RGB should be in [0,255]
            'edge_colour': a sequence of (R,G,B,A). RGB shoud be in [0,255]
            'linewidth':   a float giving the width of the edge in pixels
            'font':        a str corresponding to the TrueType font object,
                           e.g. 'helvetica-bold'
            'fontsize':    a int giving the fontsize of the character
            'offset':      a sequence (x-offset, y-offset) expressed as proportions
                           of the image size
            'size_scale':  a sequence (x-scale, y-scale) for scaling height and
                           width of characters

            'image_size':  a sequence (width, height) giving the image size in pixels

        Args:
            opt: a dict containing the options for defining a character
        '''

        self.opt = opt
        self.identity = opt['identity']
        self.face_colour = opt['face_colour']
        self.edge_colour = opt['edge_colour']
        self.linewidth = opt['linewidth']
        self.font = opt['font']
        self.fontsize = opt['fontsize']
        self.offset = opt['offset']
        self.size_scale = opt['size_scale']
        self.image_size = opt['image_size']
        self.get_letter_positions()

    def get_letter_positions(self):
        '''
        Returns the offsets from the centre of the image in pixels
        '''
        x_pos = int(self.image_size[0] * self.offset[0])
        y_pos = int(self.image_size[1] * self.offset[1])
        self.position = (x_pos, y_pos)

