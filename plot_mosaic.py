from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from skimage import io

import cv2

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class PlotMosaic:

    def __init__(self, img_size: int = 28, unnorm: bool = False):
        if unnorm:
            self.unnorm = UnNormalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        self.img_size = img_size

    def img_reshape(self, img, new_size):
        img = Image.open(img).convert('RGB')
        img = img.resize((new_size, new_size))
        img = np.asarray(img)
        return img

    def img_list_to_df(self, img_list: list, nome_imagem: list = None):
        num_images = len(img_list)
        img_arr_1d = []
        for c in range(num_images):
            # image 2d to 1d
            img_arr_1d.append(img_list[c].flatten())
        print("Num images 1d: ", len(img_arr_1d))
        print("Image 1d shape: ", img_arr_1d[0].shape)


        x_img = []
        # array 1d to list
        for c in range(num_images):
            x_img.append(img_arr_1d[c].tolist())

        # df with 1d images
        tam_colum = len(img_arr_1d[0])
        nome_colum = []
        for c in range(tam_colum):
            colum = 'pixel ' + str(c)
            nome_colum.append(colum)
        df = pd.DataFrame(x_img, columns=nome_colum)
        if nome_imagem:
            df.insert(loc=0, column='image', value=nome_imagem)
        print(df.shape)
        return df

    def df_to_img_mosaic(self,
                         df,
                         save_img_name,
                         bmnist: bool = False,
                         img_names: bool = False):
        x_data = df
        print(x_data.shape)
        print(df.shape)
        # remove image name column
        if img_names:
            x_data = df.iloc[0:, 1:]

        # converte dataframe para vetor
        x_data = x_data.to_numpy()
        # calculate how many rows and columns we need (round up)
        cols = int(np.ceil(np.sqrt(x_data.shape[0])))
        rows = int(np.ceil(x_data.shape[0] * 1. / cols))
        im_w = im_h = self.img_size
        num_images = x_data.shape[0]

        X = np.reshape(x_data, (num_images, im_h, im_w, 3))

        if cols != rows:
            dif = cols - rows
            if cols > rows:
                rows = int(np.ceil((x_data.shape[0] * 1. / cols) + dif))
            else:
                cols = int(np.ceil(np.sqrt((x_data.shape[0]) + dif)))

        mosaic = np.zeros((rows * im_h, cols * im_w, 3), dtype=np.uint8)
        if bmnist:
            mosaic = np.zeros((rows * im_h, cols * im_w, 3), dtype=np.float32)

        for row in range(rows):
            for col in range(cols):
                im_index = col * rows + row
                if im_index < num_images:
                    mosaic[col * im_h:(col+1) * im_h,
                           row * im_w:(row+1) * im_w] = cv2.normalize(X[im_index, :, :],
                                                                      None, alpha = 0,
                                                                      beta = 255,
                                                                      norm_type =cv2.NORM_MINMAX,
                                                                      dtype = cv2.CV_8U)
        
        # write mosaic image to disk
        if not bmnist:
            cv2.imwrite(save_img_name, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
        else:
            plt.imsave(save_img_name, mosaic)
        return mosaic
