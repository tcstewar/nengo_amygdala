import nengo
import numpy as np

class Amygdala(nengo.Network):
    def __init__(self, lateral, basal, central,
                 lateral_n_per_d=100,
                 basal_n_per_d=50,
                 central_n_per_d=50,
                 label=None, seed=None, add_to_container=None
                 ):
        super(Amygdala, self).__init__(label, seed, add_to_container)
        with self:
            self.lateral = nengo.Ensemble(n_neurons=lateral_n_per_d*lateral,
                                          dimensions=lateral)
            self.basal = nengo.Ensemble(n_neurons=basal_n_per_d*basal,
                                          dimensions=basal)
            self.central = nengo.Ensemble(n_neurons=central_n_per_d*central,
                                          dimensions=central)
    def make_l2c(self, **kwargs):
        with self:
            nengo.Connection(self.lateral, self.central, **kwargs)

    def make_b2c(self, **kwargs):
        with self:
            nengo.Connection(self.basal, self.central, **kwargs)

    def make_l2b(self, **kwargs):
        with self:
            nengo.Connection(self.lateral, self.basal, **kwargs)
