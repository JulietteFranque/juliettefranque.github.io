import numpy as np
import pandas as pd


class Resistances:
    def __init__(self, areas, height_TC_DFT, height_TC_cone, height_total, eq_r_DFT, D_cone, eps_TC, eps_DFT):
        """
        Parameters
        ----------
            areas: dict
                Dictionary containing TC, cone, DFT and room areas
            height_TC_DFT: float
                Height from TC to DFT
            height_TC_cone: float
                Height from TC to cone
            eq_r_DFT: float
                DFT equivalent diameter
            D_cone: float
                Cone diameter
            eps_TC: float
                Emissivity of TC
            eps_DFT: float
                Emissivity of DFT
        """

        self.areas = areas
        self.height_TC_DFT = height_TC_DFT
        self.height_TC_cone = height_TC_cone
        self.height_total = height_total
        self.eq_r_DFT = eq_r_DFT
        self.D_cone = D_cone
        self.eps_TC = eps_TC
        self.eps_DFT = eps_DFT
        self.eps_other = .9999
        self.view_factors = None
        self.resistances = None
        self.create_view_factor_matrix()
        self.create_resistance_matrix()
        

    @staticmethod
    def F_ij_sphere_disk(r_disk, h):
        A = r_disk / h
        return 0.5 * np.sqrt(1 - 1 / np.sqrt(1 + A ** 2))

    @staticmethod
    def F_ij_disks(r_i, r_j, h):
        R_i = r_i / h
        R_j = r_j / h
        S = 1 + (1 + R_j ** 2) / R_i ** 2
        return 0.5 * (S - (S ** 2 - 4 * (r_j / r_i) ** 2) ** 0.5)

    def create_view_factor_matrix(self):
        init_df = pd.DataFrame(np.zeros((4, 4)))
        self.view_factors = pd.DataFrame(data=init_df)

        self.view_factors.index = ['cone', 'TC', 'DFT', 'room']

        self.view_factors.columns = ['cone', 'TC', 'DFT', 'room']

        self.view_factors.loc['TC', 'DFT'] = self.F_ij_sphere_disk(self.eq_r_DFT, self.height_TC_DFT)

        self.view_factors.loc['DFT', 'TC'] = self.view_factors.loc['TC', 'DFT'] * self.areas['TC'] / self.areas['DFT']

        self.view_factors.loc['TC', 'cone'] = self.F_ij_sphere_disk(self.D_cone / 2, self.height_TC_cone)

        self.view_factors.loc['cone', 'TC'] = self.view_factors.loc['TC', 'cone'] * self.areas['TC'] / self.areas[
            'cone']

        self.view_factors.loc['TC', 'room'] = 1 - self.view_factors.loc['TC', 'cone'] - self.view_factors.loc[
            'TC', 'DFT']

        self.view_factors.loc['room', 'TC'] = self.view_factors.loc['TC', 'room'] * self.areas['TC'] / self.areas[
            'room']

        self.view_factors.loc['cone', 'DFT'] = self.F_ij_disks(self.D_cone / 2, self.eq_r_DFT, self.height_total)

        self.view_factors.loc['DFT', 'cone'] = self.view_factors.loc['cone', 'DFT'] * self.areas['cone'] / self.areas[
            'DFT']

        self.view_factors.loc['DFT', 'room'] = 1 - self.view_factors.loc['DFT', 'cone']

        self.view_factors.loc['room', 'DFT'] = self.view_factors.loc['DFT', 'room'] * self.areas['DFT'] / self.areas[
            'room']

        self.view_factors.loc['cone', 'room'] = 1 - self.view_factors.loc['cone', 'TC'] - self.view_factors.loc[
            'cone', 'DFT']

        self.view_factors.loc['room', 'cone'] = self.view_factors.loc['cone', 'room'] * self.areas['cone'] / self.areas[
            'room']

        self.view_factors.loc['room', 'room'] = 1 - self.view_factors.loc['room', 'cone'] - self.view_factors.loc[
            'room', 'DFT'] - self.view_factors.loc['room', 'TC']

    def create_resistance_matrix(self):
        init_df = pd.DataFrame(np.zeros((4, 4)))
        self.resistances = pd.DataFrame(data=init_df)
        self.resistances.index = ['cone', 'TC', 'DFT', 'room']
        self.resistances.columns = ['cone', 'TC', 'DFT', 'room']

        areas_array = np.array([self.areas[surface] for surface in self.resistances.columns])
        self.resistances = 1 / (self.view_factors * np.tile(areas_array, (4, 1)).T)
        self.resistances.loc['cone', 'cone'] = (1 - self.eps_other) / (self.eps_other * self.areas['cone'])
        self.resistances.loc['TC', 'TC'] = (1 - self.eps_TC) / (self.eps_TC * self.areas['TC'])
        self.resistances.loc['DFT', 'DFT'] = (1 - self.eps_DFT) / (self.eps_DFT * self.areas['DFT'])
        self.resistances.loc['room', 'room'] = (1 - self.eps_other) / (self.eps_other * self.areas['room'])
