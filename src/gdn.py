import caffe
import numpy as np

class GlobalDeconvolutionLayer(caffe.Layer):
    """
    Implements global deconvolution as described in
    https://arxiv.org/abs/1602.03930
    """

    def setup(self, bottom, top):
        # check input pair of new height and width
        self.param = eval(self.param_str)
        if len(self.param.keys()) != 2:
            raise Exception("Need to provide both new height and width")
        self.new_h = int(self.param['h'])
        self.new_w = int(self.param['w'])

        # check that there is only one bottom blob
        if len(bottom) != 1:
            raise Exception("Only one blob is supported")

        # check the dimensions of the bottom blob
        if len(bottom[0].data.shape) != 4:
            raise Exception("The bottom blob should be of the form NxCxHxW")

        self.batch_size, self.channels, self.old_h, self.old_w = bottom[0].data.shape

        # initialise parameter
        if len(self.blobs) > 0:
            #assert self.blobs[0].shape == param['shape']
            return

        # create and initialise parameter
        self.blobs.add_blob(self.old_h, self.new_h) # w1
        fan_in = self.channels * self.old_h * self.old_w
        fan_out1 = self.channels * self.new_h * self.old_w
        low = -2. * np.sqrt(6.0/(fan_in + fan_out1))
        high = 2. * np.sqrt(6.0/(fan_in + fan_out1))
        self.blobs[0].data[...] = np.random.uniform(low = low, high = high, size = self.blobs[0].shape)

        self.blobs.add_blob(self.old_w, self.new_w) # w2
        fan_out2 = self.channels * self.new_h * self.new_w
        low = -2. * np.sqrt(6.0/(fan_in + fan_out2))
        high = 2. * np.sqrt(6.0/(fan_in + fan_out2))
        self.blobs[1].data[...] = np.random.uniform(low = low, high = high, size = self.blobs[1].shape)

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, self.channels, self.new_h, self.new_w)

    def forward(self, bottom, top):
        # for each channel and each batch
        # perform the same computations
        for batch in np.arange(self.batch_size):
            for channel in np.arange(self.channels):
                top[0].data[batch, channel] = np.dot(np.dot(self.blobs[0].data.T, bottom[0].data[batch, channel]), self.blobs[1].data)

    def backward(self, top, propagate_down, bottom):
        self.blobs[0].diff[...] = 0
        self.blobs[1].diff[...] = 0
        for batch in np.arange(self.batch_size):
            for channel in np.arange(self.channels):
                self.blobs[0].diff[...] += np.dot(np.dot(bottom[0].data[batch, channel], self.blobs[1].data), top[0].diff[batch, channel].T)
                self.blobs[1].diff[...] += np.dot(np.dot(bottom[0].data[batch, channel].T, self.blobs[0].data), top[0].diff[batch, channel])
                bottom[0].diff[batch, channel] = np.dot(self.blobs[0].data, np.dot(top[0].diff[batch, channel], self.blobs[1].data.T))
