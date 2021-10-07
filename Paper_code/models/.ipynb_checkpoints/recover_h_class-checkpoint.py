import numpy as np
import Jancode as dft



class RecoverH:
    def __init__(self, it_number, t, u_guess, T_DFT_front, T_DFT_back, L_ch_DFT, T_TC, A_TC, D_TC, T_inf, T_sur,
                 rho_TC, eps_TC, C_TC, V_TC, eps_DFT, idx_low, idx_high):
        """
        Parameters
        ----------
            it_number: float
                Number of iterations
            t: like array
                Time vector
            u_guess: float
                Initial guess for velocity
            T_DFT_front: like array
                DFT front temperature
            T_DFT_back: like array
                DFT back temperature
            L_ch_DFT: float
                DFT characteristic length
            T_TC: like array
                TC temperature
            A_TC: float
                TC area
            D_TC: float
                TC diameter
            T_inf: like array
                Air temperature
            T_sur: like array
                Surroundings temperature
            rho_TC: float
                TC density
            eps_TC: float
                TC emissivity
            C_TC: float
                TC heat capacity
            V_TC: float
                TC volume
            eps_DFT: float
                DFT emissivity
            idx_low: float
                index at which recovery should start to avoid T_TC=T_inf causing division by 0
            idx_high: float
                index at which recovery should stop
        """
        self.it_number = it_number
        self.t = t
        self.u_guess = u_guess
        self.T_DFT_front = T_DFT_front
        self.T_DFT_back = T_DFT_back
        self.L_ch_DFT = L_ch_DFT
        self.T_TC = T_TC
        self.A_TC = A_TC
        self.D_TC = D_TC
        self.T_inf = T_inf
        self.T_sur = T_sur
        self.rho_TC = rho_TC
        self.eps_TC = eps_TC
        self.C_TC = C_TC
        self.V_TC = V_TC
        self.eps_DFT = eps_DFT
        self.idx_low = idx_low
        self.idx_high = idx_high
        self.sig = 5.67e-8
        self.u_avg = None
        self.q_inc_iter = None
        self.h_f = None
        self.h_TC = None
        self.T_TC_trim = None
        self.u = None
        self.T_TC_trim = self.T_TC[self.idx_low:self.idx_high]

        self.recover_h_simple()

    def recover_h_simple(self):
        self.u_avg = []
        self.u = np.ones(self.t.shape[0]) * self.u_guess
        T_eval = (self.T_inf + self.T_DFT_front) / 2
        self.h_f = dft.Correlations(self.T_DFT_front, T_eval, self.L_ch_DFT).plate_correlation(self.u)[0]
        C = 0.65
        n = 0.25
        h_r = dft.natural_convection(self.T_DFT_back, Kelvin=True, T_infty=self.T_inf).custom(C, n)[0]


        for it in range(self.it_number):
            self.u_avg.append(np.mean(self.u[self.idx_low:self.idx_high]))
            self.q_inc_iter = dft.one_dim_conduction(self.T_DFT_front, self.T_DFT_back, self.t, self.h_f, h_r,
                                                     self.eps_DFT, model='one_d_conduction', Kelvin=True,
                                                     T_inf=self.T_inf, T_sur=self.T_sur).q_inc * 1e3


            self.h_energy_bal_TC()
            self.u = dft.Correlations(self.T_TC_trim, self.T_inf[self.idx_low:self.idx_high],
                                      self.D_TC).inverse_sphere_correlation(self.h_TC)

            self.h_f = dft.Correlations(self.T_DFT_front[self.idx_low:self.idx_high],
                                        T_eval[self.idx_low:self.idx_high],
                                        self.L_ch_DFT).plate_correlation(self.u)[0]

            before = np.ones(self.idx_low) * self.h_f[0]
            after = np.ones(len(self.t) - self.idx_high) * self.h_f[-1]
            self.h_f = np.hstack((before, self.h_f, after))

    def h_energy_bal_TC(self):
        dTdt = np.gradient(self.T_TC_trim, self.t[self.idx_low:self.idx_high])
        self.h_TC = 1 / (self.A_TC * (self.T_TC_trim - self.T_inf[self.idx_low:self.idx_high])) * (
                -self.rho_TC * self.C_TC * self.V_TC * dTdt + self.eps_TC * self.A_TC *
                self.q_inc_iter[self.idx_low:self.idx_high] - self.eps_TC * self.sig * self.A_TC * (
                    self.T_TC_trim ** 4 - self.T_sur[self.idx_low:self.idx_high] ** 4))

