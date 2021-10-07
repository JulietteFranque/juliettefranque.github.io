import numpy as np
import copy
from scipy.sparse import diags
from .dft_properties import *
from .heat_transfer_coefficients import *
from .correlations import *


class ForwardModel():
    def __init__(self, n, time, L_ins, L_DFT, q_inc, velocity, T0, T_inf, T_sur, eps_DFT, alpha_DFT, eps_TC, alpha_TC,
                 D_TC, V_TC, C_TC, rho_TC, A_TC):

        """
        Parameters
        ----------
            n: float
                Number of nodes
            
            time: array like
                Time vector in s

            L_ins: float 
                Thickness of insulation in m
            
            L_DFT: float
                Thickness of DFT in m
            
            q_inc: array like
                Prescribed heat flux in W/m^2
            
            velocity: array like
                Prescribed velocity in m/s

            T0: float
                Initial temperature

            T_inf: array like
                Air temperature in K
            
            T_sur: array like
                Surroundings temperature in K

            eps_DFT: float
                DFT emissivity
            
            alpha_DFT: float
                DFT absorptivity
            
            eps_TC: float
                Thermocouple emissivity
            
            alpha_TC: float
                Thermocouple absorptivity

            D_TC: float
                Thermocouple diameter in m
            
            V_TC: float
                Thermocouple volume in m^3
            
            C_TC: float
                Thermocouple heat capacity
            
            rho_TC: float
                Thermocouple density
            
            A_TC: float
                Thermocouple area
        """
           
        self.n = n
        self.time = time
        self.L_ins = L_ins
        self.L_DFT = L_DFT
        self.q_inc = q_inc
        self.velocity = velocity
        self.T0 = T0
        self.T_inf = T_inf
        self.T_sur = T_sur
        self.eps_DFT = eps_DFT
        self.alpha_DFT = alpha_DFT
        self.eps_TC = eps_TC
        self.alpha_TC = alpha_TC
        self.D_TC = D_TC
        self.V_TC = V_TC
        self.C_TC = C_TC
        self.rho_TC = rho_TC
        self.A_TC = A_TC
        self.sig = 5.67e-8
        self.temperature_DFT = None
        self.temperature_DFT_front = None
        self.temperature_DFT_back = None
        self.temperature_TC = None

    def run_model(self):
        self.solve_temperature()
        self.temperature_DFT_front = self.temperature_DFT[:, 1]
        self.temperature_DFT_back = self.temperature_DFT[:, -2]
        self.temperature_TC = self.temperature_TC[0,:]

    def solve_temperature(self):
        time_steps = len(self.time)
        delta_t = self.time[1] - self.time[0]
        self.temperature_DFT = np.ones((time_steps, self.n + 4)) * self.T0
        self.temperature_DFT[:, 0] = self.T0 ** 4
        self.temperature_DFT[:, -1] = self.T0 ** 4
        self.temperature_TC = np.ones([2, self.time.shape[0]])
        self.temperature_TC[0] = self.T0
        self.temperature_TC[1] = self.T0 ** 4

        k_ins = ceramic_fiber(self.T0, Kelvin=True).thermal_conductivity(self.T0)
        rCp_ins = ceramic_fiber(self.T0, Kelvin=True).volumetric_heat_capacity(self.T0)
        alpha = ceramic_fiber(self.T0, Kelvin=True).alpha
        delta_x = self.L_ins / (self.n + 1)
        self.Fo = delta_t * alpha / delta_x ** 2

        for j in range(time_steps - 1):
            T_eval = 0.5 * (self.temperature_DFT[j, 1] + self.T_inf[j])
            nu = air_props(T_eval, Kelvin=True).nu
            Gr = 9.81 * 1 / (0.5 * (self.temperature_DFT[j, 1] + self.T_inf[j])) * (
                    self.temperature_DFT[j, 1] - self.T_inf[j]) * 0.0762 ** 3 / nu ** 2
            Re = self.velocity[j] * 0.0762 / nu
            C = 0.65
            n_ = 0.25

            if Gr / Re ** 2 < 0.75:
                h_f = Correlations(self.temperature_DFT[j, 1], T_eval, 0.0762).plate_correlation(self.velocity[j])[0]

            if Gr / Re ** 2 > 1.25:
                h_f = natural_convection(self.temperature_DFT[j, 1].reshape(1, -1), Kelvin=True, T_infty=self.T_inf[j]).custom(C,n_)[0]
            else:
                h_f = Correlations(self.temperature_DFT[j, 1], T_eval, 0.0762).mixed_convection_plate(self.velocity[j],
                                                                                                 self.T_inf[j])

            h_r = natural_convection(self.temperature_DFT[j, -2].reshape(1, -1), Kelvin=True, T_infty=self.T_inf[j])
            h_r.custom(C, n_)
            h_r = h_r.h
            h_TC = Correlations(self.temperature_TC[0,j], self.T_inf[j], self.D_TC).sphere_correlation(self.velocity[j])

            rCp_f = stainless_steel(self.temperature_DFT[j, 1], Kelvin=True).rCp
            rCp_r = stainless_steel(self.temperature_DFT[j, -2], Kelvin=True).rCp
            A_matrix = self.DFT_matrix(self.eps_DFT, delta_t, rCp_f, rCp_r, self.L_DFT, k_ins, delta_x,
                                       h_f, h_r)

            dTdt_dt = A_matrix @ self.temperature_DFT[j]

            dTdt_dt[0] = dTdt_dt[0].copy() + h_f * delta_t * self.T_inf[j] / (rCp_f * self.L_DFT) + self.eps_DFT * \
                         self.q_inc[
                             j] * delta_t / (rCp_f * self.L_DFT) + self.eps_DFT * self.sig * delta_t * self.T_sur[
                             j] ** 4 / (rCp_f * self.L_DFT)
            dTdt_dt[-1] = dTdt_dt[-1].copy() + h_r * delta_t * self.T_inf[j] / (
                    rCp_r * self.L_DFT) + self.eps_DFT * self.sig * delta_t * self.T_sur[j] ** 4 / (rCp_r * self.L_DFT)
            self.temperature_DFT[j + 1, 1:self.n + 3] = self.temperature_DFT[j, 1:self.n + 3] + dTdt_dt
            self.temperature_DFT[j + 1, 0] = self.temperature_DFT[j + 1, 1] ** 4
            self.temperature_DFT[j + 1, -1] = self.temperature_DFT[j + 1, -2] ** 4

            B_matrix, D_matrix = self.create_matrix_TC(h_TC,self.q_inc[j], self.T_inf[j], self.T_sur[j])

            dTdt_TC = B_matrix @ self.temperature_TC[:, j] + D_matrix
            self.temperature_TC[0, j + 1] = self.temperature_TC[0, j] + dTdt_TC * delta_t
            self.temperature_TC[1, j + 1] = copy.copy(self.temperature_TC[0, j + 1] ** 4)

    def construct_banded_matrix(self):
        """          
        Returns
        ----------
            A: array like
                Banded matrix 
        """
        k = np.array([np.ones(self.n - 1) * (self.Fo), np.ones(self.n) * (-2 * self.Fo), np.ones(self.n - 1) * (self.Fo)])
        offset = [-1, 0, 1]
        A = diags(k, offset).toarray()
        return A

    def DFT_matrix(self, eps_DFT, delta_t, rCp_f, rCp_r, L_DFT, k_ins, delta_x, h_f, h_r):
        """          
        Returns
        ----------
            A_i: array like
                dT/dt * dt matrix
        """

        row_above = np.zeros((1, self.n))
        row_below = np.zeros((1, self.n))
        columns_right = np.zeros((self.n + 2, 2))
        columns_left = np.zeros((self.n + 2, 2))

        A_i = self.construct_banded_matrix()
        A_i = np.vstack((row_above, A_i.copy(), row_below))
        A_i = np.hstack((columns_left, A_i.copy(), columns_right))
        A_i[0, 0] = -eps_DFT * self.sig * delta_t / (rCp_f * L_DFT)
        A_i[0, 1] = -h_f * delta_t / (rCp_f * L_DFT) - k_ins * delta_t / (L_DFT * delta_x * rCp_f)
        A_i[0, 2] = k_ins * delta_t / (L_DFT * delta_x * rCp_f)
        A_i[1, 1] = self.Fo
        A_i[-1, -1] = -eps_DFT * self.sig * delta_t / (rCp_r * L_DFT)
        A_i[-1, -2] = -h_r * delta_t / (rCp_r * L_DFT) - k_ins * delta_t / (L_DFT * delta_x * rCp_r)
        A_i[-1, -3] = k_ins * delta_t / (delta_x * L_DFT * rCp_r)
        A_i[-2, -2] = self.Fo
        return A_i

    def create_matrix_TC(self, h, q_inc, T_inf, T_sur):
        """       
        Parameters
        ----------   
            h: float
                Thermocouple heat transfer coefficient
            
            q_inc: float
                Heat flux in W/m^2
            
            T_inf: float
                Air temperature in K
            
            T_sur: float
                Surroundings temperature in K

        Returns
        ----------
            D: array like
               matrix
            B: array like
                matrix
        
        where dT/dt = D@T + B
        """
        B = np.zeros(2)
        B[0] = -h * self.A_TC
        B[1] = - self.eps_TC * self.A_TC * self.sig
        D = np.array([self.alpha_TC * q_inc * self.A_TC + self.eps_TC * self.A_TC * self.sig * T_sur ** 4 + h * self.A_TC * T_inf])
        B = 1 / (self.rho_TC * self.C_TC * self.V_TC) * B
        D = 1 / (self.rho_TC * self.C_TC * self.V_TC) * D
        return B, D
