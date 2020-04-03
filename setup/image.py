import numpy             as np
import h5py              as hp
import matplotlib.pyplot as plt

class Image():

    def __init__(self, model_name, r):
        '''
        Constructor for the Image class
        :param model_name: name of the model file
        :param r: number of the ray along which the image is taken
        '''
        self.r = r
        self.modelName = model_name
        self.load(model_name, self.r)

    def load (self, model_name, r):

        if model_name.lower().endswith(('.hdf5', '.h5')):
            with hp.File (model_name, 'r') as file:
                self.Im = np.array (file.get(f'Simulation/Image/I_m_{r}'))
                self.Ip = np.array (file.get(f'Simulation/Image/I_p_{r}'))
                self.Xs = np.array (file.get(f'Simulation/Image/ImX_{r}'))
                self.Ys = np.array (file.get(f'Simulation/Image/ImY_{r}'))
        else:
            self.Im = np.loadtxt (f'{model_name}/Simulation/Image/I_m_{r}.txt')
            self.Ip = np.loadtxt (f'{model_name}/Simulation/Image/I_p_{r}.txt')
            self.Xs = np.loadtxt (f'{model_name}/Simulation/Image/ImX_{r}.txt')
            self.Ys = np.loadtxt (f'{model_name}/Simulation/Image/ImY_{r}.txt')

        if (self.Im.shape[1] == self.Ip.shape[1]):
            self.nfreqs = self.Im.shape[1]
        else:
            raise ValueError('Non matching nfreqs...')

    #     def crop (self):
    #         Xs_cropped = []
    #         Ys_cropped = []
    #         im_cropped = []
    #         for (p,(x,y,I)) in enumerate(zip(self.Xs,self.Ys,self.im)):
    #             if 55000<=p<65000:#simulation.geometry.boundary.boundary[p]:
    #                 Xs_cropped.append(x)
    #                 Ys_cropped.append(y)
    #                 im_cropped.append(I)
    #         self.Xs = np.array(Xs_cropped)
    #         self.Ys = np.array(Ys_cropped)
    #         self.im = np.array(im_cropped)

    def bin_axis (self):

        self.Ixs = []
        self.Iys = []

        self.im_bin    = np.zeros((nx_pixels,ny_pixels))
        self.im_bin_nr = np.zeros((nx_pixels,ny_pixels))

        x_min = np.min(self.Xs)
        y_min = np.min(self.Ys)

        x_max = np.max(self.Xs)
        y_max = np.max(self.Ys)

        for (p,(x,y)) in enumerate(zip(self.Xs, self.Ys)):

            xi = int((nx_pixels-1) * (x - x_min) / (x_max-x_min) )
            yi = int((ny_pixels-1) * (y - y_min) / (y_max-y_min) )

            self.im_bin   [xi,yi] += self.im[p]
            self.im_bin_nr[xi,yi] += 1

            self.Ixs.append(int( xi ))
            self.Iys.append(int( yi ))

        self.im_bin = self.im_bin / self.im_bin_nr

    def plot (self, f, name):
        plt.figure(dpi=300)
        plt.title(f'r={self.r}, f={f}')
        plt.tricontourf(self.Xs, self.Ys, self.Im[:,f], levels=100, cmap='inferno')
        # self.bin_axis()
        #plt.imshow(self.im_bin, cmap='gray')
        #plt.scatter(self.Xs, self.Ys)
        plt.colorbar().set_label(r'intensity [J/m$^2$]')
        plt.axes().set_aspect('equal')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.tight_layout()
        plt.savefig(f'image_{name}_f-{f:05}_r-{self.r:05}.png')

    def plot_all (self, name):
        for f in range(self.nfreqs):
            self.plot (f, name)