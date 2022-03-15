import utils
import numpy as np
import matplotlib.pyplot as plt


class RBM:
  """
    """
  def __init__(self, p, q):
    self.W = np.random.randn(p,q)*0.1 # centered with variance 10^-2
    self.b = np.zeros(q)
    self.a = np.zeros(p)
    self.p = p
    self.q = q

def init_RBM(p, q):
  return RBM(p,q)


def entree_sortie_RBM(self, V):
  """
  data: matrice de taille mxp
  self : RBM object du type RBM.
  """

  H = V @ self.W + self.b

  return utils.sigmoid(H)


def sortie_entree_RBM(self, H):
  """
  data: matrice de taille mxq
  rbm: object du type RBM.
  """

  V = H @ self.W.T + self.a

  return utils.sigmoid(V)


""" def train_RBM(self, epochs, lr, taille_batch, X):
   

    n = X.shape[0]
    p, q = self.a.shape[1], self.b.shape[1]
    shuffled_index = np.arange(n)
    err_eqm = []

    for i in range(0, epochs):
        np.random.shuffle(shuffled_index)
        x = X[shuffled_index]
        for batch in range(0, n, taille_batch):
            data_batch = x[batch:min(batch + taille_batch, n), :]
            taille_batch = data_batch.shape[0]

            v0 = data_batch
            p_h_v0 = entree_sortie_RBM(self, v0)
            h_0 = 1 * (np.random.rand(taille_batch, q) < p_h_v0)

            p_v_h0 = sortie_entree_RBM(self, h_0)
            v1 = 1 * (np.random.rand(taille_batch, p) < p_v_h0)
            p_h_v1 = entree_sortie_RBM(self, v1)

            da = np.sum(v0 - v1, axis=0)
            db = np.sum(p_h_v0 - p_h_v1, axis=0)
            dW = v0.T @ p_h_v0 - v1.T @ p_h_v1

            self.W += lr * dW / taille_batch
            self.a += lr * da / taille_batch
            self.b += lr * db / taille_batch

            # fin du batch
        h = entree_sortie_RBM(self, X)
        x_recovered = sortie_entree_RBM(self, h)
        err = np.mean(np.sum((X - x_recovered) ** 2, axis=1))
        err_eqm.append(err)

    return self, err_eqm """


def train_RBM(self, X, nb_iter=50, batch_size=200, eps=0.01):
    n = X.shape[0]
    loss = []
    for i in range(nb_iter):
        X_shuffled = X.copy()
        np.random.shuffle(X_shuffled) 
        
        for batch_index in range(0,n,batch_size):
                
            X_batch = X_shuffled[batch_index:min(batch_index + batch_size, n)]
            current_batch_size = X_batch.shape[0]
            V0 = X_batch
            P_H0 = entree_sortie_RBM(self, V0)
            probs = np.random.rand(current_batch_size,self.q)
            H0 = (probs < P_H0) * 1.
            P_V1 = sortie_entree_RBM(self, H0)
            probs = np.random.rand(current_batch_size,self.p)
            V1 = (probs < P_V1) * 1.
            P_H1 = entree_sortie_RBM(self, V1)
                
            grad_a = np.sum(X_batch - V1,axis=0)
            grad_b = np.sum(P_H0 - P_H1,axis=0)
            grad_W = V0.T @ P_H0 - V1.T @ P_H1
                
            self.a += eps * (grad_a/current_batch_size)
            self.b += eps * (grad_b/current_batch_size)
            self.W += eps * (grad_W/current_batch_size)
            
            # affichage loss
        current_loss = np.mean((V1 - V0)**2)
        loss.append(current_loss)
        print(f'Epoch: {i} ------ Reconstruction error: {current_loss}')
    return self, loss


def generer_images_RBM(self,nb_images,nb_iter_gibbs,image_shape=(16,20)):
    for i in range(nb_images):
        #V = (np.random.rand(self.p) < 0.5) * 1
        V = np.random.rand(1, self.p)
        for j in range(nb_iter_gibbs):
            H = (np.random.rand(self.q) < entree_sortie_RBM(self,V)) * 1
            V = (np.random.rand(self.p) < sortie_entree_RBM(self,H)) * 1
                
            # affichage de l'image
        image = V.reshape(image_shape)
        plt.imshow(image, cmap='gray',  interpolation='nearest')
        plt.show()

"""def generer_image_RBM(rbm, nb_images, iter_gibbs, visualize = True):

  p, q = rbm.a.shape[1], rbm.b.shape[1]
  imgs = []
  for i in range(0, nb_images):
    v = 1* np.random.rand(1,p)<0.5
    for j in range(0, iter_gibbs):
      p_h = entree_sortie_RBM(rbm, v)
      h = 1* np.random.rand(1,q)<p_h
      p_v = sortie_entree_RBM(rbm, h)
      v = 1* np.random.rand(1,p)<p_v

    #fin generation
    imgs.append(1 * v.reshape(20, 16))
    if visualize:
        plt.figure()
        plt.imshow(imgs[-1], cmap='gray') # AlphaDigits
        plt.title("Generated image after {0} iterations".format(iter_gibbs))
        plt.show()

  return np.array(imgs)"""
