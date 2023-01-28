################################################################
# FILE: cartoonify.py
# EXERCISE: Intro2cs2 ex6 2021-2022
# WRITER: Nicole Gurevich
# DESCRIPTION: functions receiving lists
# WEB PAGES I USED: Giks for giks, youtube,w3school,google
################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
from sys import argv

from ex6_helper import *
from typing import Optional


def separate_channels(image: ColoredImage) -> List[List[List[int]]]:
    """separate a colored image to different channels"""
    l1 = []  # for channels
    for i in range(len(image[0][0])):
        list1 = []  # for ist in channel
        for j in range(len(image)):
            list2 = []  # for number in list
            for x in range(len(image[j])):
                list2.append(image[j][x][i])
            list1.append(list2)
        l1.append(list1)
    return l1


def combine_channels(channels: List[List[List[int]]]) -> ColoredImage:
    """gets a few channels and combines them into colored image"""
    l1 = []  # for combined channels
    for i in range(len(channels[0])):
        list1 = []
        for j in range(len(channels[0][0])):
            list2 = []
            for x in range(len(channels)):
                list2.append(channels[x][i][j])
            list1.append(list2)
        l1.append(list1)
    return l1


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    """gets colored image and make it gray"""
    img_copy = deepcopy(colored_image)  # don't want to work on the original
    black_white = [[0 for j in range(len(img_copy[0]))]
                   for i in range(len(img_copy))]  # list comprehension
    for i in range(len(black_white)):
        for j in range(len(black_white[0])):  # new gray scale according to formula
            black_white[i][j] = round(0.299 * img_copy[i][j][0]
                                      + 0.587 * img_copy[i][j][1] + 0.114
                                      * img_copy[i][j][2])
    return black_white


def blur_kernel(size: int) -> Kernel:
    """gets kernel size and returns the right kernel"""
    l1 = []
    for i in range(size):
        l2 = []
        for j in range(size):
            val = 1 / (size ** 2)
            l2.append(val)
        l1.append(l2)
    return l1


def out_of_range(image, pixel_x, pixel_y):
    """checking if the pixel is in range """
    range_x = len(image)
    range_y = len(image[0])
    if pixel_x < 0 or pixel_y < 0 \
            or pixel_x >= range_x or pixel_y >= range_y:
        return True


def blur_singlpxl(pixel, pixels, kernel, image):
    """calculate single pixel according to the kernel """
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    pixel_new = 0
    for i in range(len(kernel)):
        for j in range(len(kernel)):
            # maching between the pixel position and the kernel
            pixel_x = pixels - kernel_height // 2 + i
            pixel_y = pixel - kernel_width // 2 + j
            if out_of_range(image, pixel_x, pixel_y):
                # if pixel is in the corner gives the new
                # pixel the old pixels value
                pixel_new += image[pixels][pixel] * kernel[i][j]
            else:  # if is not corner than calculate for the new x and y
                pixel_new += image[pixel_x][pixel_y] * kernel[i][j]
    if pixel_new < 0:  # if lower than 0 than the value is 0
        pixel_new = 0
    if pixel_new > 255:  # if bigger than 255 the value is 255
        pixel_new = 255
    return round(pixel_new)


def apply_kernel(image: SingleChannelImage, kernel: Kernel) \
        -> SingleChannelImage:
    """apply kernel to a single channel image and return the
    new image """
    img_copy = deepcopy(image)  # don't want to change the original
    for i in range(len(image)):
        for j in range(len(image[0])):
            img_copy[i][j] = blur_singlpxl(j, i, kernel, image)  # apply
            # to every single pixel
    return img_copy


def bilinear_interpolation(image: SingleChannelImage,
                           y: float, x: float) -> int:
    """adjusts the pixels to the new position """
    dx = x - int(x)
    dy = y - int(y)
    a = image[int(y)][int(x)]  # initial position
    b = c = d = 0
    # geting b,c and d value for the calculation
    if not out_of_range(image, int(y) + 1, int(x)):
        b = image[int(y) + 1][int(x)]
    if not out_of_range(image, int(y), int(x) + 1):
        c = image[int(y)][int(x) + 1]
    if not out_of_range(image, int(y) + 1, int(x) + 1):
        d = image[int(y) + 1][int(x) + 1]
    # return value according to tha given formula
    return round(a * (1 - dx) * (1 - dy) + b * dy * (1 - dx) + c * dx * (1 - dy) + d * dx * dy)


def resize(image: SingleChannelImage, new_height: int, new_width: int) \
        -> SingleChannelImage:
    """gets image and new height and width and resize the image to t
    he new height and width"""
    new_image = [[0 for j in range(new_width)] for i in range(new_height)]
    # list comprehension
    height = len(image)
    width = len(image[0])
    for i in range(new_height):
        for j in range(new_width):  # resizing according to the
            # ratio between the old value and the new
            new_image[i][j] = bilinear_interpolation \
                (image, (i * ((height -1)/ (new_height-1))), (j * ((width -1)/ (new_width-1))))
    new_image[0][0] = image[0][0]  # the corners stay the same
    new_image[0][-1] = image[0][-1]
    new_image[-1][0] = image[-1][0]
    new_image[-1][-1] = image[-1][-1]
    return new_image


def scale_down_colored_image(image: ColoredImage, max_size: int) \
        -> Optional[ColoredImage]:
    """gets image and max size, if image parameters is lower
    than max size than do nothing, else scale down the
    image to the right size"""
    height = len(image)
    width = len(image[0])
    if height <= max_size and width <= max_size:  # in this case
        # doesn't need changes
        return None
    channels = separate_channels(image)  # separate to channels
    # now we calculate to itch channel using list comprehension
    if height > max_size and height > width:
        res_channels = [resize(channel, max_size,
                               round(max_size * (width / height)))
                        for channel in channels]
        return combine_channels(res_channels)
    if width > max_size and width > height:
        res_channels = [resize(channel,
                               round(max_size * (height / width)), max_size)
                        for channel in channels]
        return combine_channels(res_channels)
    if height > max_size and height == width:
        res_channels = [resize(channel, max_size, max_size)
                        for channel in channels]
        return combine_channels(res_channels)


def rotate_90(image: Image, direction: str) -> Image:
    """gets image and direction and rotate it in
    90 degrees to the given direction """
    for i in range(len(image)):
        for j in range(len(image[0])):
            if direction == 'L':
                return rotate(image)
            if direction == 'R':  # rotate to right is
                # like rotate to left 3 times
                return rotate(rotate(rotate(image)))


def rotate(image):
    """rotate the image
    90 degrees to the left"""
    new_matrix = [[image[j][i]
                   for j in range(len(image))]
                  for i in range(len(image[0]) - 1, -1, -1)]
    return new_matrix


def get_edges(image: SingleChannelImage, blur_size: int,
              block_size: int, c: int) -> SingleChannelImage:
    """gets image,then check if its darker or lighter,
     if darker than makes the pixel black else white """
    img_copy = deepcopy(image)  # Copy image data
    blured_image = apply_kernel(img_copy, blur_kernel(blur_size))
    # blur image using blur_size param
    r = block_size // 2  # radius of colour points to take in account
    new_image = []  # final picture result
    for i in range(len(blured_image)):  # for each line
        temp = []
        for j in range(len(blured_image[0])):  # for each row
            # calculate threshold
            # calculation for what to do with colour point
            threshold = threshold_helper(blured_image, i, j, r) // \
                        (block_size * block_size) - c
            if blured_image[i][j] < threshold:  # turn black
                temp.append(0)
            else:  # turne white
                temp.append(255)
        new_image.append(temp)
    return new_image


def threshold_helper(image, i, j, r):
    """count all the values in the pixels that
     is in the range"""
    counter = 0
    for x in range(i - r, i + r + 1):
        for y in range(j - r, j + r + 1):
            temp = out_of_range(image, x, y)  # check if not in range
            if (temp):  # if not in range than it adds the origin pixel value
                counter += image[i][j]
            else:  # adds in range values
                counter += image[x][y]
    return counter


def quantize(image: SingleChannelImage, N: int) \
        -> SingleChannelImage:
    """gets image , recalculate according to the formula"""
    img_copy = deepcopy(image)  # don't want to change the original image
    for i in range(len(img_copy)):
        for j in range(len(img_copy[0])):
            img_copy[i][j] = round(int(image[i][j] * (N / 256)) * (255 / (N - 1)))
    return img_copy


def quantize_colored_image(image: ColoredImage, N: int) \
        -> ColoredImage:
    """quantize colored image by separating it to channels"""
    channeles = separate_channels(image)  # separete for channels
    channel_lst = []
    for i in channeles:
        chanel = quantize(i, N)  # quantize for itch channel
        channel_lst.append(chanel)
    return combine_channels(channel_lst)


def add_mask(image1: Image, image2: Image, mask: List[List[float]]) \
        -> Image:
    """gets 2 images and a mask image , mask the images according
    to ths formula"""
    if isinstance(image1[0][0], list):  # checking if the images are colored
        channels1 = separate_channels(image1)  # separate image1 to channels
        channels2 = separate_channels(image2)  # separate image2 to channels
        chanel_lst = []  # for the values from itch channel
        for chanel in range(len(channels1)):
            new_image = [[0 for j in range(len(image1[0]))] for i in
                         range(len(image1))]  # list comprehension
            for i in range(len(channels1[0])):
                for j in range(len(channels1[0][0])):
                    # calculate for itch channel according to the formula
                    new_image[i][j] = round(channels1[chanel][i][j] *
                                            mask[i][j] + channels2[chanel][i][j]
                                            * (1 - mask[i][j]))
            chanel_lst.append(new_image)
        return combine_channels(chanel_lst)
    else:  # if image has a single channel then calculate
        # according to the formula
        new_image = [[0 for j in range(len(image1[0]))] for i in
                     range(len(image1))]
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                new_image[i][j] = round(image1[i][j] * mask[i][j]
                                        + image2[i][j] * (1 - mask[i][j]))
        return new_image


def cartoonify(image: ColoredImage, blur_size: int, th_block_size: int,
               th_c: int, quant_num_shades: int) -> ColoredImage:
    """gets image and parameters, get the edges and quantize of
    the image,i created a mask from the edges of the image and then
     made a new image by the add_mask function, the result is a
     cartoon picture"""
    edges_of_image = get_edges(RGB2grayscale(image), blur_size
                               , th_block_size, th_c)
    quantized_image = quantize_colored_image(image, quant_num_shades)
    mask = [[0 for j in range(len(image[0]))] for i in
            range(len(image))]  # list comprehension
    for i in range(len(image)):
        for j in range(len(image[0])):  # divide
            # itch pixel by 255 to get value between 0 and 1
            mask[i][j] = edges_of_image[i][j] / 255
    channels = separate_channels(quantized_image)  # separate to channels
    new_image_channels = []
    for channel in channels:  # do mask for every channel then combine them
        new_image = add_mask(channel, edges_of_image, mask)
        new_image_channels.append(new_image)
    return combine_channels(new_image_channels)

#
# if __name__ == '__main__':
#     if len(argv) != 0:#checking the number of arguments
#         print("Number of arguments is not valid")
#     else:
#         image = load_image(argv[1])
#         scale_image = scale_down_colored_image(image, int(argv[3]))
#         if scale_image is None:#if image is in correct size then continue with the image
#             save_image(cartoonify(image, int(argv[4]), int(argv[5])
#                                   , int(argv[6]), int(argv[7])), argv[2])
#         else:#if its to big then resize the pitcher and then run
#             save_image(cartoonify(scale_image, int(argv[4]), int(argv[5])
#                                   , int(argv[6]), int(argv[7])), argv[2])

    # image = load_image("ziggy.png")
    # save_image(get_edges(RGB2grayscale(image),5,13,11), "get_edge_test.png")
    # save_image(apply_kernel(RGB2grayscale(image),blur_kernel(5)),"kernel_test.png")
    # save_image(RGB2grayscale(image),'RGB_test.png')
    # save_image(quantize_colored_image(image, 8),"quan_test.png")
    # save_image(cartoonify(image,5,13,11,8),"cartoon_test.png")
    # channels = separate_channels(image)
    # resized = [resize(channel, 258, 460) for channel in channels]
    # resized = combine_channels(resized)
    # save_image(resized, "resize_demo.png")
    # save_image(quantize_colored_image(resized, 8), "quantize_demo.png")
    # save_image(scale_down_colored_image(image, 460),"scale_test")
    # print("Completed!")
