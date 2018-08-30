import nengo
import numpy as np
import nengo_spa as spa

class Amygdala(nengo.Network):
    def __init__(self, lateral_dim, basal_dim, central_dim,
                 lateral_n_per_d=100, 
                 basal_n_per_d=50,
                 central_n_per_d=50,
                 label=None, seed=None, add_to_container=None,
                 basal_config=1
                 ):
        super(Amygdala, self).__init__(label, seed, add_to_container)
        with self:
            self.basal_dim = basal_dim
            self.lateral_dim = lateral_dim
            self.central_dim = central_dim
            self.basal_config = basal_config

            self.lateral = nengo.Ensemble(n_neurons=lateral_n_per_d*lateral_dim,
                                          dimensions=lateral_dim)

            if (basal_config==0):
                self.basal = nengo.Network(label='basal')
                with self.basal:
                    self.basal.input = nengo.Node(None, size_in=basal_dim)
                    self.basal.combined = nengo.Ensemble(n_neurons=basal_n_per_d*basal_dim*2,
                                                      dimensions=basal_dim*2)
                    nengo.Connection(self.basal.input, self.basal.combined[:basal_dim], synapse=None)

                    self.basal.output = nengo.Ensemble(n_neurons=basal_n_per_d*basal_dim,dimensions=basal_dim)

                    def basal_func(x):
                        out1 = (x[0]+x[2])/2
                        out2 = (x[1]+x[3])/2
                        return out1, out2

                    nengo.Connection(self.basal.combined, self.basal.output, function=basal_func)
            elif (basal_config==1): # basal is just normal ensemble
                self.basal = nengo.Ensemble(n_neurons=basal_n_per_d*basal_dim,
                                          dimensions=basal_dim)
            else:
                print("Invalid basal config parameter")


            self.central = nengo.Network(label='central')
            with self.central:
                self.central.input = nengo.Ensemble(n_neurons=central_n_per_d*central_dim,
                                              dimensions=central_dim)
                self.central.wta = spa.networks.selection.WTA(
                                        n_neurons=50,
                                        n_ensembles=central_dim,
                                        threshold=0.3,
                                        label='central')
                nengo.Connection(self.central.input, self.central.wta.input)
                self.central.output = nengo.Node(None, size_in=central_dim)
                nengo.Connection(self.central.wta.output, self.central.output, synapse=None)

    def make_l2c(self, **kwargs):
        with self:
            nengo.Connection(self.lateral, self.central.input, **kwargs)

    def make_b2c(self, **kwargs):
        with self:
            if (self.basal_config==0):
                nengo.Connection(self.basal.output, self.central.input, **kwargs)
            elif(self.basal_config==1):
                nengo.Connection(self.basal, self.central.input, **kwargs)
            else:
                print('Invalid basal config parameter')

    def make_l2b(self, **kwargs):
        with self:
            if (self.basal_config==0):
                nengo.Connection(self.lateral, self.basal.combined[:self.basal_dim], **kwargs)
            elif(self.basal_config==1):
                nengo.Connection(self.lateral, self.basal, transform=0.5, **kwargs)
            else:
                print('Invalid basal config parameter')
