# man get me into c as quickly as possible.

cdef extern from "sim.h":
    cdef struct broState:
        double* T_t;
        double* m_l;
        double* m_v;
        double* D_f;
        double* m_g;
        double* nmol_g;
        double* T_g;
        double* Cp_g;

        int upto;
        int count;

        double V_t;

        double C_w;

        double vff0_o;

        double Cd_inj;
        double A_inj;

        double L_f;
        double rho_f;
        double D0_f;

        double D_c;
        double eta_c;
        double Vempty_c;

        double Cd_nzl;
        double A_nzl;
        double eps_nzl;

        double T_a;
        double P_a;
        double rho_a;
        double Mw_a;
        double cp_a;


    void bro_sim(broState* s)

import numpy as np
cimport numpy as np
np.import_array()


cdef class State:
    cdef broState obj
    def __cinit__(State self, *,
                np.ndarray[np.float64_t, mode="c"] T_t,
                np.ndarray[np.float64_t, mode="c"] m_l,
                np.ndarray[np.float64_t, mode="c"] m_v,
                np.ndarray[np.float64_t, mode="c"] D_f,
                np.ndarray[np.float64_t, mode="c"] m_g,
                np.ndarray[np.float64_t, mode="c"] nmol_g,
                np.ndarray[np.float64_t, mode="c"] T_g,
                np.ndarray[np.float64_t, mode="c"] Cp_g,
                double V_t,
                double C_w,
                double vff0_o,
                double Cd_inj,
                double A_inj,
                double L_f,
                double D0_f,
                double D_c,
                double eta_c,
                double Vempty_c,
                double Cd_nzl,
                double A_nzl,
                double eps_nzl,
                double T_a,
                double P_a,
                double rho_a,
                double Mw_a,
                double cp_a,
            ):
        self.obj.T_t    = <double*>T_t.data
        self.obj.m_l    = <double*>m_l.data
        self.obj.m_v    = <double*>m_v.data
        self.obj.D_f    = <double*>D_f.data
        self.obj.m_g    = <double*>m_g.data
        self.obj.nmol_g = <double*>nmol_g.data
        self.obj.T_g    = <double*>T_g.data
        self.obj.Cp_g   = <double*>Cp_g.data
        self.obj.upto = <int>0
        self.obj.count = <int>min(
                T_t.size, m_l.size, m_v.size,
                D_f.size, m_g.size, nmol_g.size,
                T_g.size, Cp_g.size
            )
        self.obj.V_t = V_t
        self.obj.C_w = C_w
        self.obj.vff0_o = vff0_o
        self.obj.Cd_inj = Cd_inj
        self.obj.A_inj = A_inj
        self.obj.L_f = L_f
        self.obj.D0_f = D0_f
        self.obj.D_c = D_c
        self.obj.eta_c = eta_c
        self.obj.Vempty_c = Vempty_c
        self.obj.Cd_nzl = Cd_nzl
        self.obj.A_nzl = A_nzl
        self.obj.eps_nzl = eps_nzl
        self.obj.T_a = T_a
        self.obj.P_a = P_a
        self.obj.rho_a = rho_a
        self.obj.Mw_a = Mw_a
        self.obj.cp_a = cp_a

    def sim(State self):
        bro_sim(&self.obj)
        # Keep how many elements it used, then reset.
        count = int(self.obj.upto)
        self.obj.upto = <int>0
        return count
