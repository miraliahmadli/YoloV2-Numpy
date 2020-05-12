import os
import sys
import math
import networkx as nx
import numpy as np

import multiprocessing
import itertools
# from multiprocessing import set_start_method
num_cpu = multiprocessing.cpu_count()

class DnnInferenceEngine(object):
    def __init__(self, graph):
        self.g = graph

    def run(self, tin):
        self.g.in_node.set_input(tin)
        out = {}
        currents = [self.g.in_node]
        done = set()
        while (len(currents) != 0):
            print(currents)
            nexts = []
            for current in currents:
                skip_current = False
                predecessors = self.g.G.predecessors(current)
                for predecessor in predecessors:
                    if predecessor not in done:
                        nexts.append(predecessor)
                        skip_current = True
                if skip_current:
                    continue
                current.run()
                if self.g.is_out_node(current):
                    out = current.result
                done.add(current)
                for successor in self.g.G.successors(current):
                    nexts.append(successor)
            currents = nexts
        print(out[0,:, :, 0])
        return out

class DnnGraphBuilder(object):
    def __init__(self):
        self.G = nx.DiGraph()
        self.name_num = {"conv2d": 0, 
                         "bias_add": 0, 
                         "max_pool2d": 0, 
                         "batch_norm": 0, 
                         "leaky_relu": 0, 
                         "input": 0}
        self.in_node = None
        self.out_node = None

    def set_in_node(self, node):
        self.in_node = node

    def set_out_node(self, node):
        self.out_node = node

    def is_out_node(self, node):
        return self.out_node is node

    def get_name(self, layer_name):
        name = layer_name + "_" + str(self.name_num[layer_name])
        self.name_num[layer_name] += 1
        return name

    def create_conv2d(self, in_node, kernel, strides, padding):
        out_node = Conv2D(self.get_name("conv2d"), in_node, kernel, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_bias_add(self, in_node, biases):
        out_node = BiasAdd(self.get_name("bias_add"), in_node, biases)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_max_pool2d(self, in_node, ksize, strides, padding):
        out_node = MaxPool2D(self.get_name("max_pool2d"), in_node, ksize, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_batch_norm(self, in_node, mean, variance, gamma, epsilon):
        out_node = BatchNorm(self.get_name("batch_norm"), in_node, mean, variance, gamma, epsilon)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_leaky_relu(self, in_node):
        out_node = LeakyReLU(self.get_name("leaky_relu"), in_node)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_input(self, in_shape):
        out_node = Input(self.get_name("input"), in_shape)
        self.G.add_node(out_node) 
        self.set_in_node(out_node)  # Assume there's only one input
        return out_node

class DnnNode(object):
    def __init__(self):
        pass

    def run(self):
        self.result = None 

#
# Complete below classes.
#

class Conv2D(DnnNode):
    def __init__(self, name, in_node, kernel, strides, padding):
        batch, in_height, in_width, in_channels = in_node.out_shape
        filter_height, filter_width, kernel_in_channels, out_channels = kernel.shape

        assert in_channels == kernel_in_channels, \
        "Shape of filters must be same to number of input channels, %d is not equal to %d" % (kernel_in_channels, in_channels)

        assert padding in ["SAME", "VALID"], \
        "Invalid padding name: %s, should be either SAME or VALID" % padding

        self.name = name
        print(self.name)
        self.kernel = kernel
        padding = (padding == "SAME")
        self.strides = strides

        pad_h = 0
        pad_w = 0
        if padding:
            # ((s-1) * x + k -s)/ 2
            pad_h = ((self.strides[1] - 1) * in_height + filter_height - self.strides[1])
            pad_w = ((self.strides[2] - 1) * in_width + filter_width - self.strides[2])
            if pad_h == 0 and pad_w == 0:
                self.prev_res = in_node.result
            else:
                self.prev_res = np.zeros((batch, in_height + pad_h, in_width + pad_w, in_channels))
                self.prev_res[:, pad_h//2 : -((pad_h+1)//2), pad_w//2 : -((pad_w + 1)//2), :] = in_node.result
        else:
            self.prev_res = in_node.result
        output_height  = int(((in_height - filter_height + pad_h) / self.strides[1]) + 1)
        output_width   = int(((in_width - filter_width + pad_w) / self.strides[2]) + 1)
        self.result = np.zeros((batch, output_height, output_width, out_channels))
        self.out_shape = self.result.shape
        self.filter_height, self.filter_width, self.in_channels, out_channels = kernel.shape
        print(self.out_shape)
    
    def multi(self, vals):
        b, i, j, k, f_h, f_w = vals
        self.result[b, i, j, k] += \
            np.sum(np.multiply(self.prev_res[b, self.strides[1] * i : self.strides[1] * i + f_h, self.strides[2] * j : self.strides[2] * j + f_w, :], 
                self.kernel[:, :, :, k]))
        # for h in range(self.filter_height):
        #     for w in range(self.filter_width):
        #         for q in range(self.in_channels):
        #             self.result[b, i, j, k] = \
        #                 (self.prev_res[b, self.strides[1] * i + h, self.strides[2] * j + w, q] * 
        #                     self.kernel[h, w, q, k])
    def run(self):
        filter_height, filter_width, in_channels, out_channels = self.kernel.shape
        batch, output_height, output_width, out_channels = self.out_shape
        # arrays = [range(batch), range(output_height), range(output_width), range(out_channels)]
        arrays = [range(batch), range(output_height), range(output_width), range(out_channels), [filter_height], [filter_width]]
        args = itertools.product(*arrays)
        # for b in range(batch):
        #     for i in range(output_height):
        #         for j in range(output_width):
        #             for k in range(out_channels):
                        # # vectorized
                        # self.result[b, i, j, k] += \
                        #                 np.sum(np.multiply(self.prev_res[b, self.strides[1] * i : self.strides[1] * i + filter_height, self.strides[2] * j : self.strides[2] * j + filter_width, :], 
                        #                     self.kernel[:, :, :, k]))
                        #scalar
                        # for h in range(filter_height):
                        #     for w in range(filter_width):
                        #         for q in range(in_channels):
                        #             self.result[b, i, j, k] += \
                        #                 (self.prev_res[b, self.strides[1] * i + h, self.strides[2] * j + w, q] * 
                        #                     self.kernel[h, w, q, k])
        with multiprocessing.Pool(num_cpu) as p:
            p.map(self.multi, args) 

        # output[b, i, j, k] =
        #     sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q]
        #                     * filter[di, dj, q, k]
class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        self.prev_res =  in_node.result
        batch, in_height, in_width, in_channels = in_node.out_shape
        assert in_channels == biases.shape[0], \
        "Shape of biases must be equal to number of input channels, %d is not equal to %d" % (biases.shape[0], in_channels)

        self.biases = biases
        self.name = name
        print(self.name)
        self.result = np.zeros(in_node.out_shape)
        self.out_shape = self.result.shape
        print(self.out_shape)

    def multi(self, vals):
        b, h, w, c = vals
        self.result[b, h, w, c] = self.prev_res[b, h, w, c] + self.biases[c]

    def run(self):
        batch, output_height, output_width, out_channels = self.out_shape
        for b in range(batch):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(out_channels):
                        self.result[b, h, w, c] = self.prev_res[b, h, w, c] + self.biases[c]
        # arrays = [range(batch), range(output_height), range(output_width), range(out_channels)]
        # args = itertools.product(*arrays)
        # with multiprocessing.Pool(num_cpu) as p:
        #     p.map(self.multi, args) 
        

class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        batch, in_height, in_width, in_channels = in_node.out_shape
        out_channels = in_channels
        assert padding in ["SAME", "VALID"], \
        "Invalid padding name: %s, should be either SAME or VALID" % padding

        self.strides = strides
        self.ksize = ksize
        self.name = name
        print(self.name)
        
        padding = (padding == "SAME")

        pad_h = 0
        pad_w = 0
        if padding:
            # ((s-1) * x + k -s)/ 2
            pad_h = (self.ksize[1] - 1)
            pad_w = (self.ksize[2] - 1)
            if pad_h == 0 and pad_w == 0:
                self.prev_res = in_node.result
            else:
                self.prev_res = np.zeros((batch, in_height + pad_h, in_width + pad_w, in_channels))
                self.prev_res[:, pad_h//2 : -((pad_h+1)//2), pad_w//2 : -((pad_w + 1)//2), :] = in_node.result
        else:
            self.prev_res = in_node.result
        
        output_height  = int((in_height - self.ksize[1] + pad_h) / self.strides[1] + 1)
        output_width   = int((in_width - self.ksize[2] + pad_w) / self.strides[2] + 1)
        self.result = np.zeros((batch, output_height, output_width, out_channels))
        self.out_shape = self.result.shape
        print(self.out_shape)
        
    def run(self):
        batch, in_height, in_width, in_channels = self.prev_res.shape
        batch, output_height, output_width, out_channels = self.out_shape
        def get_max_from_window(b, x, y, c):
            max_num = 0
            for i in range(x, min(x + self.ksize[1], in_height)):
                for j in range(y, min(y + self.ksize[2], in_width)):
                    if self.prev_res[b, i, j, c] > max_num:
                        max_num = self.prev_res[b, i, j, c]
            return max_num

        # def multi(b, i, j, c):
        #     # m = get_max_from_window(b, i, j, c)
        #     self.result[b, i, j, c] = get_max_from_window(b, i, j, c)

        for b in range(batch):
            for i in range(0, output_height, self.strides[1]):
                for j in range(0, output_width, self.strides[2]):
                    for c in range(out_channels):
                        m = get_max_from_window(b, i, j, c)
                        self.result[b, i, j, c] = m
        # arrays = [range(batch), range(output_height), range(output_width), range(out_channels)]
        # args = itertools.product(*arrays)
        # with multiprocessing.Pool(num_cpu) as p:
        #     p.map(self.multi, args) 


class BatchNorm(DnnNode):
    def __init__(self, name, in_node, mean, variance, gamma, epsilon):
        self.prev_res = in_node.result
        batch, in_height, in_width, in_channels = in_node.out_shape

        assert in_channels == mean.shape[0], \
        "Shape of mean must be equal to number of input channels, %d is not equal to %d" % (mean.shape[0], in_channels)

        assert in_channels == variance.shape[0], \
        "Shape of variance must be equal to number of input channels, %d is not equal to %d" % (variance.shape[0], in_channels)

        assert in_channels == gamma.shape[0], \
        "Shape of gamma must be equal to number of input channels, %d is not equal to %d" % (gamma.shape[0], in_channels)
        
        self.name = name
        print(self.name)

        self.mean = mean
        self.epsilon = epsilon
        self.gamma = gamma
        self.variance = variance
        self.result = np.zeros(in_node.out_shape)
        self.out_shape = self.result.shape
        print(self.out_shape)

    def multi(self, vals):
        b, h, w, c = vals
        self.result[b, h, w, c] = \
            (self.prev_res[b, h, w, c] - self.mean[c]) / np.sqrt(self.variance[c] + self.epsilon)

    def run(self):
        batch, output_height, output_width, out_channels = self.out_shape
        for b in range(batch):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(out_channels):
                        self.result[b, h, w, c] = \
                            (self.prev_res[b, h, w, c] - self.mean[c]) / np.sqrt(self.variance[c] + self.epsilon)
        # arrays = [range(batch), range(output_height), range(output_width), range(out_channels)]
        # args = itertools.product(*arrays)
        # with multiprocessing.Pool(num_cpu) as p:
        #     p.map(self.multi, args) 

class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        self.prev_res = in_node.result
        batch, in_height, in_width, in_channels = in_node.out_shape
        self.alpha = 0.01
        self.name = name
        print(self.name)
        self.result = np.zeros(in_node.out_shape)
        self.out_shape = self.result.shape
        print(self.out_shape)
    
    def multi(self, vals):
        b, h, w, c = vals
        if self.prev_res[b, h, w, c] < 0:
            self.result[b, h, w, c] = self.prev_res[b, h, w, c] * self.alpha
        else:
            self.result[b, h, w, c] = self.prev_res[b, h, w, c]
    
    def run(self):
        batch, output_height, output_width, out_channels = self.out_shape

        for b in range(batch):
            for h in range(output_height):
                for w in range(output_width):
                    for c in range(out_channels):
                        if self.prev_res[b, h, w, c] < 0:
                            self.result[b, h, w, c] = self.prev_res[b, h, w, c] * self.alpha
                        else:
                            self.result[b, h, w, c] = self.prev_res[b, h, w, c]
        # arrays = [range(batch), range(output_height), range(output_width), range(out_channels)]
        # args = itertools.product(*arrays)
        # with multiprocessing.Pool(num_cpu) as p:
        #     p.map(self.multi, args) 


# Do not modify below
class Input(DnnNode):
    def __init__(self, name, in_shape):
        self.name = name
        # print(self.name)
        self.in_shape = in_shape 
        self.out_shape =in_shape
        # print(self.out_shape)
        self.result = np.ndarray(self.in_shape)

    def set_input(self, tensor):
        assert tuple(self.in_shape) == tuple(tensor.shape)
        self.result = tensor 

    def run(self):
        pass

