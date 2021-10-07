import numpy as np
import scipy.optimize as opt
import Jancode as dft


class Correlations:
    def __init__(self, T, T_eval, L_ch, Kelvin=True):
        """
        Parameters
        ----------
            T: float, array like
                Surface temperature
            
            T_eval: float, array like
                Temperature at which properties will be evaluated

            L_ch: float (optional)
                Characteristic length for determining heat transfer coefficients
            
            Kelvin: bool (optional)
                Whether or not the temperatures are in Celsius or Kelvin. If in Celsius, the data is converted to Kelvin
        """

        if not Kelvin:
            self.T_eval = T_eval + 273
            self.T = T + 273

        else:
            self.T_eval = T_eval
            self.T = T
        self.L_ch = L_ch
        self.nu = dft.air_props(self.T_eval, Kelvin=True).nu
        self.alpha = dft.air_props(self.T_eval, Kelvin=True).alpha
        self.k = dft.air_props(self.T_eval, Kelvin=True).k
        self.Pr = self.nu / self.alpha
        self.rho = self.density(self.T_eval)
        self.nu_s = dft.air_props(self.T, Kelvin=True).nu
        self.rho_s = self.density(self.T)
        self.mu_s = self.nu_s * self.rho_s
        self.mu = self.nu * self.rho
        self.mu_ratio = self.mu / self.mu_s

    @staticmethod
    def density(T):
        """
        Parameters
        ----------
            T: scalar or vector
                Temperature at which to find density
                
        Returns
        ----------
            rho: scalar or vector
                Density
        """

        R_air = 287.05
        P_atm = 101325
        rho = P_atm / (R_air * T)
        return rho

    def plate_correlation(self, u):
        """
        Parameters
        ----------
            u: float or array like
                Flow velocity
        Returns
        ----------
            h: float or array like
                Heat transfer coefficient
        """
        Re = u * self.L_ch / self.nu
        Nu = 0.664 * Re ** 0.5 * self.Pr ** (1 / 3)
        h = Nu * self.k / self.L_ch
        return h

    def sphere_correlation(self, u):
        """
        Parameters
        ----------
            u: float or array like
                Flow velocity

        Returns
        ----------
            h: scalar or vector
                Heat transfer coefficient
        """
        Re = u * self.L_ch / self.nu
        Nu = 2 + (0.4 * Re ** 0.5 + 0.06 * Re ** (2 / 3)) * self.Pr ** 0.4 * self.mu_ratio ** 0.25
        h = Nu * self.k / self.L_ch
        return h

    def inverse_sphere_correlation(self, h):
        """
        Parameters
        ----------
            h: scalar or vector
                Heat transfer coefficient
                
        Returns
        ----------
            u: scalar or vector
                Flow velocity
        """

        Nu = h * self.L_ch / self.k
        Re_recovered = np.zeros(self.T.shape[0])

        def Re(h):
            Re = 2 + (0.4 * h ** 0.5 + 0.06 * h ** (2 / 3)) * (Pr_eq ** 0.4) * mu_ratio_eq ** 0.25 - Nu_eq
            return Re

        for idx in range(self.T.shape[0]):
            Nu_eq = Nu[idx]
            mu_ratio_eq = self.mu_ratio[idx]
            Pr_eq = self.Pr[idx]
            solution = opt.root(Re, np.array(3))
            Re_recovered[idx] = solution.x

        u = Re_recovered * self.nu / self.L_ch
        return u
