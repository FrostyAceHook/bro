# man get me into c as quickly as possible.

cdef extern from "sim.h":
    cdef struct broRunning:
        int onfire;
        double t;
        double T_t;
        double m_l;
        double m_v;
        double ID_f;
        double m_g;
        double N_g;
        double T_g;
        double Cp_g;
        double Cv_g;
        double alt_r;
        double vel_r;

    cdef struct broState:
        double target_apogee;

        double m_locked;
        double L_locked;
        double com_locked;

        double D_r;
        double alt0_r;

        double T_a;

        double L_tw;
        double rho_tw;
        double Ys_tw;
        double c_tw;
        double sf_tw;
        double V_t;
        double m_tw;
        double C_tw;

        double vff0_l;

        double m_mov;
        double L_mov;
        double com_mov;

        double m_inj;
        double L_inj;
        double com_inj;
        double Cd_inj;
        double A_inj;

        double D_c;
        double rho_cw;
        double Ys_cw;
        double sf_cw;
        double PCPP;
        double L_c;
        double Vempty_c;
        double m_cw;
        double th_cw;

        double L_f;
        double th0_f;

        double Cd_nzl;
        double eps;
        double A_throat;
        double L_nzl;
        double m_nzl;

        broRunning running;

        int upto;
        int count;
        double* out_t;
        double* out_alt_r;
        double* out_vel_r;
        double* out_acc_r;
        double* out_m_r;
        double* out_com_r;
        double* out_T_t;
        double* out_T_g;
        double* out_P_t;
        double* out_P_c;
        double* out_P_a;
        double* out_m_l;
        double* out_m_v;
        double* out_m_f;
        double* out_dm_inj;
        double* out_dm_reg;
        double* out_dm_out;
        double* out_m_g;
        double* out_cp_g;
        double* out_cv_g;
        double* out_y_g;
        double* out_R_g;
        double* out_ofr;
        double* out_Isp;
        double* out_Fthrust;
        double* out_Fdrag;
        double* out_Fgravity;

    void bro_sim(broState* s)

import numpy as np
cimport numpy as np
np.import_array()


cdef class State:
    cdef broState obj
    def __cinit__(State self, *,
                np.ndarray[np.float64_t, mode="c"] t=None,
                np.ndarray[np.float64_t, mode="c"] alt_r=None,
                np.ndarray[np.float64_t, mode="c"] vel_r=None,
                np.ndarray[np.float64_t, mode="c"] acc_r=None,
                np.ndarray[np.float64_t, mode="c"] m_r=None,
                np.ndarray[np.float64_t, mode="c"] com_r=None,
                np.ndarray[np.float64_t, mode="c"] T_t=None,
                np.ndarray[np.float64_t, mode="c"] T_g=None,
                np.ndarray[np.float64_t, mode="c"] P_t=None,
                np.ndarray[np.float64_t, mode="c"] P_c=None,
                np.ndarray[np.float64_t, mode="c"] P_a=None,
                np.ndarray[np.float64_t, mode="c"] m_l=None,
                np.ndarray[np.float64_t, mode="c"] m_v=None,
                np.ndarray[np.float64_t, mode="c"] m_f=None,
                np.ndarray[np.float64_t, mode="c"] dm_inj=None,
                np.ndarray[np.float64_t, mode="c"] dm_reg=None,
                np.ndarray[np.float64_t, mode="c"] dm_out=None,
                np.ndarray[np.float64_t, mode="c"] m_g=None,
                np.ndarray[np.float64_t, mode="c"] cp_g=None,
                np.ndarray[np.float64_t, mode="c"] cv_g=None,
                np.ndarray[np.float64_t, mode="c"] y_g=None,
                np.ndarray[np.float64_t, mode="c"] R_g=None,
                np.ndarray[np.float64_t, mode="c"] ofr=None,
                np.ndarray[np.float64_t, mode="c"] Isp=None,
                np.ndarray[np.float64_t, mode="c"] Fthrust=None,
                np.ndarray[np.float64_t, mode="c"] Fdrag=None,
                np.ndarray[np.float64_t, mode="c"] Fgravity=None,
                double target_apogee,
                double m_locked,
                double L_locked,
                double com_locked,
                double D_r,
                double alt0_r,
                double T_a,
                double L_tw,
                double rho_tw,
                double Ys_tw,
                double c_tw,
                double sf_tw,
                double vff0_l,
                double m_mov,
                double L_mov,
                double com_mov,
                double m_inj,
                double L_inj,
                double com_inj,
                double Cd_inj,
                double A_inj,
                double D_c,
                double rho_cw,
                double Ys_cw,
                double sf_cw,
                double PCPP,
                double L_f,
                double th0_f,
                double Cd_nzl,
                double eps,
                double A_throat,
            ):
        min_size = None
        def new_min_size(x):
            if x is None:
                return min_size
            return x.size if min_size is None else min(min_size, x.size)
        min_size = new_min_size(t)
        min_size = new_min_size(alt_r)
        min_size = new_min_size(vel_r)
        min_size = new_min_size(acc_r)
        min_size = new_min_size(m_r)
        min_size = new_min_size(com_r)
        min_size = new_min_size(T_t)
        min_size = new_min_size(T_g)
        min_size = new_min_size(P_t)
        min_size = new_min_size(P_c)
        min_size = new_min_size(P_a)
        min_size = new_min_size(m_l)
        min_size = new_min_size(m_v)
        min_size = new_min_size(m_f)
        min_size = new_min_size(dm_inj)
        min_size = new_min_size(dm_reg)
        min_size = new_min_size(dm_out)
        min_size = new_min_size(m_g)
        min_size = new_min_size(cp_g)
        min_size = new_min_size(cv_g)
        min_size = new_min_size(y_g)
        min_size = new_min_size(R_g)
        min_size = new_min_size(ofr)
        min_size = new_min_size(Isp)
        min_size = new_min_size(Fthrust)
        min_size = new_min_size(Fdrag)
        min_size = new_min_size(Fgravity)
        if min_size is None:
            min_size = 0
        self.obj.count = <int>min_size
        self.obj.out_t        = NULL if t        is None else <double*>t.data
        self.obj.out_alt_r    = NULL if alt_r    is None else <double*>alt_r.data
        self.obj.out_vel_r    = NULL if vel_r    is None else <double*>vel_r.data
        self.obj.out_acc_r    = NULL if acc_r    is None else <double*>acc_r.data
        self.obj.out_m_r      = NULL if m_r      is None else <double*>m_r.data
        self.obj.out_com_r    = NULL if com_r    is None else <double*>com_r.data
        self.obj.out_T_t      = NULL if T_t      is None else <double*>T_t.data
        self.obj.out_T_g      = NULL if T_g      is None else <double*>T_g.data
        self.obj.out_P_t      = NULL if P_t      is None else <double*>P_t.data
        self.obj.out_P_c      = NULL if P_c      is None else <double*>P_c.data
        self.obj.out_P_a      = NULL if P_a      is None else <double*>P_a.data
        self.obj.out_m_l      = NULL if m_l      is None else <double*>m_l.data
        self.obj.out_m_v      = NULL if m_v      is None else <double*>m_v.data
        self.obj.out_m_f      = NULL if m_f      is None else <double*>m_f.data
        self.obj.out_dm_inj   = NULL if dm_inj   is None else <double*>dm_inj.data
        self.obj.out_dm_reg   = NULL if dm_reg   is None else <double*>dm_reg.data
        self.obj.out_dm_out   = NULL if dm_out   is None else <double*>dm_out.data
        self.obj.out_m_g      = NULL if m_g      is None else <double*>m_g.data
        self.obj.out_cp_g     = NULL if cp_g     is None else <double*>cp_g.data
        self.obj.out_cv_g     = NULL if cv_g     is None else <double*>cv_g.data
        self.obj.out_y_g      = NULL if y_g      is None else <double*>y_g.data
        self.obj.out_R_g      = NULL if R_g      is None else <double*>R_g.data
        self.obj.out_ofr      = NULL if ofr      is None else <double*>ofr.data
        self.obj.out_Isp      = NULL if Isp      is None else <double*>Isp.data
        self.obj.out_Fthrust  = NULL if Fthrust  is None else <double*>Fthrust.data
        self.obj.out_Fdrag    = NULL if Fdrag    is None else <double*>Fdrag.data
        self.obj.out_Fgravity = NULL if Fgravity is None else <double*>Fgravity.data
        self.obj.target_apogee = target_apogee
        self.obj.m_locked = m_locked
        self.obj.L_locked = L_locked
        self.obj.com_locked = com_locked
        self.obj.D_r = D_r
        self.obj.alt0_r = alt0_r
        self.obj.T_a = T_a
        self.obj.L_tw = L_tw
        self.obj.rho_tw = rho_tw
        self.obj.Ys_tw = Ys_tw
        self.obj.c_tw = c_tw
        self.obj.sf_tw = sf_tw
        self.obj.vff0_l = vff0_l
        self.obj.m_mov = m_mov
        self.obj.L_mov = L_mov
        self.obj.com_mov = com_mov
        self.obj.m_inj = m_inj
        self.obj.L_inj = L_inj
        self.obj.com_inj = com_inj
        self.obj.Cd_inj = Cd_inj
        self.obj.A_inj = A_inj
        self.obj.D_c = D_c
        self.obj.rho_cw = rho_cw
        self.obj.Ys_cw = Ys_cw
        self.obj.sf_cw = sf_cw
        self.obj.PCPP = PCPP
        self.obj.L_f = L_f
        self.obj.th0_f = th0_f
        self.obj.Cd_nzl = Cd_nzl
        self.obj.eps = eps
        self.obj.A_throat = A_throat

    def sim(State self):
        self.obj.upto = <int>0
        bro_sim(&self.obj)
        return int(self.obj.upto)
