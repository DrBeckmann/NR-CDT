import numpy as np
import cv2

# Black background
bg = np.zeros((512,512), dtype='uint8')

# White foreground
fg = (255, 255, 255)
thickness = 3


def resize(image, size):
    image = cv2.resize(image, dsize=(size,size), interpolation=cv2.INTER_AREA)
    image[image < 64] = 0
    image[image >= 64] = 255
    return image

def gen_circle(image, seed, size):
    im = np.copy(image)
    im = cv2.circle(im, (seed[0],seed[1]+size), size, fg, thickness)
    return im

def gen_square(image, seed, size):
    im = np.copy(image)
    im = cv2.rectangle(im, (seed[0]-size,seed[1]), (seed[0]+size,seed[1]+2*size), fg, thickness)
    return im

def gen_triangle(image, seed, size):
    im = np.copy(image)
    im = cv2.line(im, (seed[0]-size,seed[1]+2*size), (seed[0]+size,seed[1]+2*size), fg, thickness)
    im = cv2.line(im, (seed[0]+size,seed[1]+2*size), seed, fg, thickness)
    im = cv2.line(im, seed, (seed[0]-size,seed[1]+2*size), fg, thickness)
    return im

def gen_bar(image, seed, length):
    im = np.copy(image)
    im = cv2.line(im, seed, (seed[0],seed[1]-length), fg, thickness) 
    return im

def gen_cross(image, seed, height, branch, width):
    im = np.copy(image)
    im = cv2.line(im, seed, (seed[0],seed[1]-height), fg, thickness) 
    im = cv2.line(im, (branch[0]-width,branch[1]), (branch[0]+width,branch[1]), fg, thickness)
    return im

def gen_star(image, seed, height, branch, width):
    im = np.copy(image)
    im = cv2.line(im, seed, (seed[0],seed[1]-height), fg, thickness)
    im = cv2.line(im, (branch[0]-width,branch[1]+width), (branch[0]+width,branch[1]-width), fg, thickness)
    im = cv2.line(im, (branch[0]-width,branch[1]-width), (branch[0]+width,branch[1]+width), fg, thickness)
    return im

# Size of images
image_size = 128

# Parameter
seed = (256, 304)
base_size = 80
bar_length = 214
cross_height = 214
cross_branch = (256,150)
cross_length = 56
star_height = 214
star_branch = (256,150)
star_width = 56

circle = gen_circle(bg, seed, base_size)
circle_bar = gen_bar(circle, seed, bar_length)
circle_cross = gen_cross(circle, seed, cross_height, cross_branch, cross_length)
circle_star = gen_star(circle, seed, star_height, star_branch, star_width)
square = gen_square(bg, seed, base_size)
square_bar = gen_bar(square, seed, bar_length)
square_cross = gen_cross(square, seed, cross_height, cross_branch, cross_length)
square_star = gen_star(square, seed, star_height, star_branch, star_width)
triangle = gen_triangle(bg, seed, base_size)
triangle_bar = gen_bar(triangle, seed, bar_length)
triangle_cross = gen_cross(triangle, seed, cross_height, cross_branch, cross_length)
triangle_star = gen_star(triangle, seed, star_height, star_branch, star_width)


'''Save templates'''
#Resize images
circle_bar = resize(circle_bar, image_size)
circle_cross = resize(circle_cross, image_size)
circle_star = resize(circle_star, image_size)
square_bar = resize(square_bar, image_size)
square_cross = resize(square_cross, image_size)
square_star = resize(square_star, image_size)
triangle_bar = resize(triangle_bar, image_size)
triangle_cross = resize(triangle_cross, image_size)
triangle_star = resize(triangle_star, image_size)


templates = np.array([circle_bar, circle_cross, circle_star, square_bar, square_cross, square_star, triangle_bar, triangle_cross, triangle_star])
np.save('templates.npy', templates)