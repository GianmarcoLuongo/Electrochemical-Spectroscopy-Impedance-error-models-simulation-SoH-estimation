from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
import numpy as np

class CompositeKernel(Kernel):
    """
    k(x,x') = k_EIS(x_EIS,x'_EIS)
            * k_temp(T, T')
            * k_soc(SOC, SOC')

    Components:
    - k_EIS: RBF (squared-exponential) on EIS features
    - k_temp: Arrhenius
    - k_soc: polynomial on SOC

    Hyperparameters (optimizable):
    - variance_eis, lengthscale_eis
    - variance_temp, lengthscale_temp
    - variance_soc, constant_soc, degree_soc (degree can be fixed or small integer)
    """
    # Define hyperparameter objects (name, type, default_bounds)
    hyperparameter_variance_eis = Hyperparameter(name="variance_eis", value_type="numeric", bounds=[1e-5, 1e5])
    hyperparameter_lengthscale_eis = Hyperparameter(name="lengthscale_eis", value_type="numeric", bounds=[1e-5, 1e5])
    hyperparameter_variance_temp = Hyperparameter(name="variance_temp", value_type="numeric", bounds=[1e-5, 1e5])
    hyperparameter_lengthscale_temp = Hyperparameter(name="lengthscale_temp", value_type="numeric", bounds=[1e-5, 1e5])
    hyperparameter_variance_soc = Hyperparameter(name="variance_soc", value_type="numeric", bounds=[1e-5, 1e5])
    hyperparameter_constant_soc = Hyperparameter(name="constant_soc", value_type="numeric", bounds="fixed")
    hyperparameter_degree_soc = Hyperparameter(name="degree_soc", value_type="numeric", bounds="fixed")

    def __init__(self,
                 variance_eis=1.0, lengthscale_eis=1.0,
                 variance_temp=1.0, lengthscale_temp=1.0,
                 variance_soc=1.0, constant_soc=1.0, degree_soc=2):
        self.variance_eis = variance_eis
        self.lengthscale_eis = lengthscale_eis
        self.variance_temp = variance_temp
        self.lengthscale_temp = lengthscale_temp
        self.variance_soc = variance_soc
        self.constant_soc = constant_soc
        self.degree_soc = degree_soc

    @property
    def theta(self):
        # log-transformed parameters (except fixed)
        return np.log([self.variance_eis,
                       self.lengthscale_eis,
                       self.variance_temp,
                       self.lengthscale_temp,
                       self.variance_soc])

    @property
    def bounds(self):
        return np.log([(1e-5, 1e5),  # variance_eis
                       (1e-5, 1e5),  # lengthscale_eis
                       (1e-5, 1e5),  # variance_temp
                       (1e-5, 1e5),  # lengthscale_temp
                       (1e-5, 1e5)])  # variance_soc

    def clone_with_theta(self, theta):
        """Return a new kernel instance with hyperparameters set to exp(theta)."""        
        var_eis, ls_eis, var_temp, ls_temp, var_soc = np.exp(theta)
        return CompositeKernel(
            variance_eis=var_eis,
            lengthscale_eis=ls_eis,
            variance_temp=var_temp,
            lengthscale_temp=ls_temp,
            variance_soc=var_soc,
            constant_soc=self.constant_soc,
            degree_soc=self.degree_soc
        )

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        m = X.shape[1] - 2
        X_eis, X_temp, X_soc = X[:, :m], X[:, m], X[:, m+1]
        Y_eis, Y_temp, Y_soc = Y[:, :m], Y[:, m], Y[:, m+1]

        # EIS RBF
        d2_eis = np.sum((X_eis[:, None] - Y_eis[None, :])**2, axis=2)
        K_eis = self.variance_eis * np.exp(-0.5 * d2_eis / self.lengthscale_eis**2)
        # Temp Arrhenius
        inv_Xt = 1.0 / X_temp[:, None]
        inv_Yt = 1.0 / Y_temp[None, :]
        d_temp = np.abs(inv_Xt - inv_Yt)
        K_temp = self.variance_temp * np.exp(-d_temp / self.lengthscale_temp)
        # SOC polynomial
        prod_soc = np.outer(X_soc, Y_soc)
        K_soc = (self.variance_soc * prod_soc + self.constant_soc)**self.degree_soc

        K = K_eis * K_temp * K_soc

        if eval_gradient:
            # gradients w.r.t log-hyperparameters theta
            # dK/dvariance_eis = K / variance_eis
            grad_var_eis = K / self.variance_eis
            # dK/dlengthscale_eis = K_eis * (d2_eis/lengthscale_eis^3) * K_temp * K_soc
            grad_ls_eis = K * (d2_eis / (self.lengthscale_eis**2)) / self.lengthscale_eis
            # dK/dvariance_temp = K / variance_temp
            grad_var_temp = K / self.variance_temp
            # dK/dlengthscale_temp = K * (d_temp/lengthscale_temp)
            grad_ls_temp = K * (d_temp / self.lengthscale_temp)
            # dK/dvariance_soc = K * (degree_soc * prod_soc)/(variance_soc * prod_soc + constant_soc)
            denom = (self.variance_soc * prod_soc + self.constant_soc)
            grad_var_soc = K * (self.degree_soc * prod_soc) / denom

            # chain rule for log-space: dK/d(log p) = p * dK/dp
            grads = np.stack([
                self.variance_eis * grad_var_eis,
                self.lengthscale_eis * grad_ls_eis,
                self.variance_temp * grad_var_temp,
                self.lengthscale_temp * grad_ls_temp,
                self.variance_soc * grad_var_soc
            ], axis=2)
            return K, grads

        return K

    def diag(self, X):
        soc = X[:, -1]
        return np.full(X.shape[0],
                       self.variance_eis *
                       self.variance_temp *
                       (self.variance_soc * soc**2 + self.constant_soc)**self.degree_soc)

    def is_stationary(self):
        return False

    def __repr__(self):
        return (f"CompositeKernel(var_eis={self.variance_eis}, ls_eis={self.lengthscale_eis}, "
                f"var_temp={self.variance_temp}, ls_temp={self.lengthscale_temp}, "
                f"var_soc={self.variance_soc}, c_soc={self.constant_soc}, d_soc={self.degree_soc})")
    
    @theta.setter
    def theta(self, theta):
        self.variance_eis, self.lengthscale_eis, \
        self.variance_temp, self.lengthscale_temp, \
        self.variance_soc = np.exp(theta)

