import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
from scipy import misc
from motion_blur.generate_PSF import PSF
from motion_blur.generate_trajectory import Trajectory


class BlurImage(object):

    def __init__(self, image_path, PSFs=None, part=None, path__to_save=None):
        """

        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        if os.path.isfile(image_path):
            self.image_path = image_path
            self.original = misc.imread(self.image_path)
            self.shape = self.original.shape
            if len(self.shape) < 3:
                raise Exception('We support only RGB images yet.')
            elif self.shape[0] != self.shape[1]:
                raise Exception('We support only square images yet.')
        else:
            raise Exception('Not correct path to image.')
        self.path_to_save = path__to_save
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[0]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[0], path_to_save=os.path.join(self.path_to_save,
                                                                                'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []

    def blur_image(self, save=False, show=False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        key, kex = self.PSFs[0].shape
        delta = yN - key
        assert delta >= 0, 'resolution of image should be higher than kernel'
        result=[]
        if len(psf) > 1:
            for p in psf:
                tmp = np.pad(p, delta // 2, 'constant')
                cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = np.zeros(self.shape)
                blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
                blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
                blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
                blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
                blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #Dont really know why shfit colors
                blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
                result.append(np.abs(blured))
        else:
            psf = psf[0]
            tmp = np.pad(psf, delta // 2, 'constant')
            cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
            blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
            blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
            blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
            blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            result.append(np.abs(blured))
        self.result = result
        if show or save:
            self.__plot_canvas(show, save)

    def __plot_canvas(self, show, save):
        if len(self.result) == 0:
            raise Exception('Please run blur_image() method first.')
        else:
            plt.close()
            plt.axis('off')
            fig, axes = plt.subplots(1, len(self.result), figsize=(10, 10))
            if len(self.result) > 1:
                for i in range(len(self.result)):
                        axes[i].imshow(self.result[i])
            else:
                plt.axis('off')

                plt.imshow(self.result[0])
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), self.result[0] * 255)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), self.result[0] * 255)
            elif show:
                plt.show()


if __name__ == '__main__':
    #folder = '/Users/mykolam/PycharmProjects/University/DeblurGAN2/results_sharp'
    #folder_to_save = '/Users/mykolam/PycharmProjects/University/DeblurGAN2/blured'
    folder = '/Users/apple/Desktop/research_summer/MSKCC_deblur/image/sharp'
    folder_to_save='/Users/apple/Desktop/research_summer/MSKCC_deblur/image/blur'
    if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
    #params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    #params=[0.2,0.15,0.1,0.07,0.05,0.03,0.01,0.007,0.005]
    params=[0.2,0.17,0.15,0.13,0.1,0.07,0.05,0.03,0.01,0.007] #Usual tasks (first 8437 pairs)
    #params=[0.3,0.25,0.2,0.17,0.15,0.13,0.1,0.07,0.05,0.01] #for the brain images, easier. 
    #for path in os.listdir(folder):
    for path in os.listdir(folder)[:]:
        print(path)
        expl=np.random.choice(params)
        trajectory = Trajectory(canvas=64, max_len=60, expl=expl).fit()
        print(expl)
        psf = PSF(canvas=64, trajectory=trajectory).fit()
        #part=np.random.choice([1,2,3])
        part=3
        print(part)
        BlurImage(os.path.join(folder, path), PSFs=psf,
                  path__to_save=folder_to_save, part=part).\
            blur_image(save=True)
