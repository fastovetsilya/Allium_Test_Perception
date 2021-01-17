"""
Split huge images into many small images for training YOLO. 
"""
from PIL import Image
from glob import glob
from shutil import rmtree
from os import mkdir, path
Image.MAX_IMAGE_PIXELS = 933120000 * 10**2 #Increase max image size

##################################################################
# Initialize input and output paths here
input_dir = 'input/'
output_dir = 'output/'
##################################################################

# Define the method            
def imgcrop_res(input_path, xRes, yRes):
    filename = input_path.split('/')[-1].split('.')[-2]
    file_extention = '.' + input_path.split('/')[-1].split('.')[-1].replace('JPG', 'jpg')
    im = Image.open(input_path)
    imgwidth, imgheight = im.size
    yPieces = imgheight // yRes
    xPieces = imgwidth // xRes
    print('\nProcessing {}'.format(filename))
    print('Splitting into {} X pieces and {} Y pieces\n'.format(xPieces, yPieces))
    height = imgheight // yPieces
    width = imgwidth // xPieces

    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            a = im.crop(box)
            try: 
                if not path.isdir(output_dir + filename):
                    mkdir(output_dir + filename)
                a.save(output_dir + '/' + filename + '/' + 
                       filename + '__' + str(i) + '-' + str(j) + '.png')
            except:
                pass

# Reinitialize output folder
rmtree(output_dir)
mkdir(output_dir)

# Process the images
images_list = glob(input_dir + '*.jpg')
if images_list == []:
    images_list = glob(input_dir + '*.JPG')

for image_path in images_list:
    imgcrop_res(image_path, 4000, 4000)
