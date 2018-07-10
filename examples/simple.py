import nengo
import numpy as np
import nengo_amygdala.amygdala as amy
from importlib import reload
reload(amy)



model = nengo.Network()
with model:
    L2B = {
           (-1, 1, -1, 1): [1,-1],
           (1, -1, -1, -1): [-1, 0.3], #angry eyes, sad mouth, no teeth, not familiar
           (1, -1, 1, 1): [-1, 1], #threat, someone you know
           (1, -1, 1, 0): [-1, 0.6], #threatening not someone you know
           (1, 0, -1, -1): [-0.5, -0.5], # sad someone you don't know
           (1, 0, -1, 1): [-1, -1], # sad someone you know
           (-1, 1, -1, -1): [0.5, 0.5], # happy someone you don't know
           (-1, 1, -1, 1): [1,1], # happy someone you know
           (-1, 0, -1, -1): [0.5, -0.5], # calm, someone you don't know
           (-1, 0, -1, 1): [1, -1], # calm, someone you don't know
          } 
    def L2C(x):
        mouth, eyes, teeth, familiar = x
      
        happy = 0
        angry = 0
        sad = 0
        calm = 0
                                                            
        if teeth < 0.8:
            angry = 1
        elif mouth < 0.8:
            angry = 1
        else:
            calm = 0.2

        return happy, angry, sad, calm

    B2C = np.array([
                    [1, 1], # happy
                    [-1, 1], # angry
                    [-1, -1], # sad
                    [1, -1], # calm
                   ])


    '''
    amygdala = amy.Amygdala(l2b=L2B, l2c=L2C, b2c=B2C)


    amygdala = amy.Amygdala(lateral_dim=4, basal_dim=2, central_dim=4)
    nengo.Connection(amygdala.lateral, amygdala.central, function=L2C)
    '''

    amygdala = amy.Amygdala(lateral=4, basal=2, central=4)
    amygdala.make_l2c(function=L2C)
    eval_points=np.array(list(L2B.keys()))
    function=np.array(list(L2B.values()))
    amygdala.make_l2b(eval_points=eval_points, function=function)
    amygdala.make_b2c(transform=B2C)
    
    
    


