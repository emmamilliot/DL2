import utils
import principal_RBM_alpha as RBM
import numpy as np
import matplotlib.pyplot as plt
from principal_RBM_alpha import train_RBM, init_RBM, entree_sortie_RBM


class DBN:

    def __init__(self, p, hidden_layers_units, nbr_classes):
        """
        Parameters:
            - p = dimension input
            - hidden_layers_units = [100, 200 ,100] <- 3 hiddens layers 
            - nbr_classes for output (final q)
        """
        if len(hidden_layers_units) == 0:
            raise ValueError('Pas de hidden layers...')
        self.p = p
        self.nbr_classes = nbr_classes
        layer_p = p
        self.DBN = []
        for l, q in enumerate(hidden_layers_units):
            self.DBN.append(init_RBM(p=layer_p, q=q))
            layer_p = q
        # Ajouter dernier RBM pour matcher le numéro de classes
        # La classification se fera par l'argmax de l'output de cette couche
        self.classification_layer = init_RBM(p=layer_p, q=nbr_classes)
        self.nbr_hidden_layers = len(self.DBN)

def init_DNN(p, hidden_layers_units, nbr_classes):
    """ construit et d’initialise les poids et les biais d’un DNN"""
    return DBN(p, hidden_layers_units, nbr_classes)


def pretrain_DNN(self, epochs, lr, taille_batch, data):
    err_layers = []
    x = data.copy()

    for i in range(self.p):
        self.p[i], err_eqm = RBM.train_RBM(self.p[i], epochs, lr, taille_batch, x)
        err_layers.append(err_eqm)
        x = RBM.entree_sortie_RBM(self.p[i], x)

    return self, err_layers


def generer_image_DBN(dbn, nb_images, iter_gibbs, visualize = True):

  p, q = dbn.layers[0].a.shape[1], dbn.layers[-1].b.shape[1]
  imgs = []

  for i in range(0, nb_images):
    v = 1* np.random.rand(1,dbn.layers[-1].W.shape[0])<0.5

    for j in range(0, iter_gibbs):
      p_h = RBM.entree_sortie_RBM(v, dbn.layers[-1])
      h = 1* np.random.rand(p_h.shape[0],p_h.shape[1])<p_h
      p_v = RBM.sortie_entree_RBM(h, dbn.layers[-1])
      v = 1* np.random.rand(p_v.shape[0],p_v.shape[1])<p_v

    for l in range(dbn.num_layers-2, -1, -1):
      proba = RBM.sortie_entree_RBM(v, dbn.layers[l])
      v = 1* np.random.rand(proba.shape[0], proba.shape[1])<proba


    #fin generation
    imgs.append(1 * v.reshape(20, 16))
    if visualize:
        plt.figure()
        plt.imshow(imgs[-1], cmap='gray') # AlphaDigits
        plt.title("Generated image after {0} iterations".format(iter_gibbs))
        plt.show()

  return np.array(imgs)

