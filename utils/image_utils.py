from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def resize_pil_image(image, output_shape, resize_shape, offset, flip=False):
    # output_shape: x,y
    # resize_shape: x,y
    # offset: x,y
    image = image.resize(resize_shape, Image.BICUBIC)
    new_image = Image.new('RGB', output_shape, (128, 128, 128))
    new_image.paste(image, offset)
    if flip:
        new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
    return new_image


def distort_image(image, hue, sat, val):
    x = rgb_to_hsv(image)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image = hsv_to_rgb(x)  # numpy array, 0 to 1
    return image
