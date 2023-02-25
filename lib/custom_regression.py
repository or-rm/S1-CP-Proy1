import numpy as np

class linear_regression:

    @property
    def y_pred(self):
        y=np.matmul(self.X_,self.coef_)
        self.y_pred_=y
        return y

    @property
    def grad(self):
        grad_=(1./self.n_obs)*np.matmul(np.transpose(self.X_),(self.y_pred-self.y_))
        self.grad_=grad_
        return(grad_)

    def fit(self):
        for i in range(self.n_epochs):
            self.coef_=self.coef_-self.alpha_*self.grad
            self.iteracion_+=1
            self.sqr_err()
        return self.coef_

    def sqr_err(self):
        y_err=(1/self.n_obs)*np.matmul(np.transpose(self.y_pred_-self.y_), (self.y_pred_-self.y_))
        self.err_iter_.append(y_err)

    def __init__(self, X, y, fit_intercept=True, n_epochs=2000, alpha=0.01):

        self.n_epochs=n_epochs
        self.n_obs=y.shape[0]

        self.X_=np.array(X)
        self.y_=np.array(y)

        self.alpha_=alpha

        self.x_dim=X.shape[1]
        self.y_dim=y.shape[1]

        self.iteracion_=0
        self.coef_=np.zeros((self.x_dim, self.y_dim))
        self.grad_=np.zeros((self.x_dim, self.y_dim))
        self.y_pred_=np.zeros((self.n_obs, self.y_dim))
        self.err_iter_=[]

        if fit_intercept:
            x_mat_shape=((self.n_obs,1))
            intercept_vec=np.ones(x_mat_shape)
            self.X_=np.concatenate((intercept_vec, self.X_), axis=1)
            self.x_dim=self.X_.shape[1]
            self.coef_=np.zeros((self.x_dim,self.y_dim))