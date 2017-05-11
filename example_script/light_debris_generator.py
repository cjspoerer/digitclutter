import sys
sys.path.append('../')
import platform
from os.path import abspath
from digitclutter import generate, io
from scipy.io import savemat
from progressbar import ProgressBar

'''
Generates an image set with the same statistics and the light debris data set describred here
(https://doi.org/10.1101/133330s)
'''

n_samples = 1000
n_debris = [10, 11]
if platform.system() == 'Windows':
    font_set = ['arial-bold']
else:
    font_set = ['helvetica-bold']

# Generate samples
clutter_list = [generate.sample_clutter(font_set=font_set) for i in range(n_samples)]

# Save image set
clutter_list = io.name_files('light_debris', clutter_list=clutter_list)
io.save_image_set(clutter_list, 'light_debris/light_debris.csv')

# Render images and save as mat file
print('Rendering images...')
bar = ProgressBar(max_value=len(clutter_list))
for i, cl in enumerate(clutter_list):
    cl.render_occlusion()
    bar.update(i+1)
print('Saving mat file...')
fname_list = [cl.fname for cl in clutter_list]
images_dict = io.save_images_as_mat(abspath('light_debris/light_debris.mat'), clutter_list, (32,32),
                                    fname_list=fname_list, delete_bmps=True, overwrite_wdir=True)

# Make debris 
debris_array = generate.make_debris(n_samples, n_debris=n_debris, font_set=font_set)
images_with_debris = generate.add_debris(images_dict['images'], debris_array)

images_with_debris_dict = {
    'images':images_with_debris,
    'targets':images_dict['targets'],
    'binary_targets':['binary_targets']
    }
savemat('light_debris/light_debris_with_debris.mat', images_with_debris_dict)
print('Done. Images saved at {0}'.format(abspath('light_debris/light_debris_with_debris.mat')))
