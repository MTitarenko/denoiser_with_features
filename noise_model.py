import numpy as np
import my_colour_demosaicing as cd


# def get_noise_model(source_noise_model):
#     tokens = source_noise_model.split(",")
#     a_min = float(tokens[0])
#     a_max = float(tokens[1])
#     b_min = float(tokens[2])
#     b_max = float(tokens[3])
#
#     def poisson_gauss_bayer(in_img):
#         img = np.float32(in_img)
#         mosaic_synth = cd.mosaicing_CFA_Bayer(img, pattern="GRBG")
#         a = np.float_power(10., np.random.uniform(a_min, a_max))
#         b = np.float_power(10., np.random.uniform(b_min, b_max))
#         norm1 = np.random.randn(mosaic_synth.shape[0], mosaic_synth.shape[1])
#         norm2 = np.random.randn(mosaic_synth.shape[0], mosaic_synth.shape[1])
#
#         max = np.amax(mosaic_synth)
#         mosaic_synth /= max
#         # Adding noise z(x) = y(x)+sqrt(a*y(x)+b)*norm(x)
#         mosaic_synth += a*np.sqrt(mosaic_synth) * norm1 + b * norm2
#         mosaic_synth *= max
#         mosaic_norm = np.clip(mosaic_synth, 0, 1)
#         mosaic_norm = mosaic_norm.astype(np.float32)
#         noise_image = cd.demosaicing_CFA_Bayer_bilinear(mosaic_norm, pattern="GRBG")
#         return noise_image
#     return poisson_gauss_bayer


def get_noise_model(source_noise_model):
    tokens = source_noise_model.split(",")
    a_min = float(tokens[0])
    a_max = float(tokens[1])

    def gauss(in_img):
        a = np.float_power(10., np.random.uniform(a_min, a_max))
        noise = np.random.randn(*in_img.shape) * a
        noise_image = in_img + noise
        noise_image = np.clip(noise_image, 0., 1.)
        return noise_image
    return gauss
