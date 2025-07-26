# cython: boundscheck=False, wraparound=False, cdivision=True, cpow=True



# Nitrous oxide properties from CoolProp, approximated via rational
# polynomials by 'bro/func_approx.py'.

# Valids bounds of all inputs:
#  temperature   263.15 K .. 308.15 K
#  pressure        80 kPa .. 7.2 Mpa
#  density        1 kg/m3 .. 250 kg/m3
#  quality              0 .. 1

# man what is this required cast to double.
cdef const double nox_Mw = <double>44.013e-3 # [kg/mol]

cdef double nox_rho_satliq(double T): # max 0.68% error
    cdef double x1 = T
    cdef double x2 = T*T
    cdef double p0 = <double>156087.80354721082
    cdef double p1 = <double>810.653176062025
    cdef double q0 = <double>73.69985723964697
    cdef double q1 = <double>0.23238113977053437
    cdef double P = p0 - p1*x1 + x2
    cdef double Q = q0 - q1*x1
    return P / Q

cdef double nox_rho_satvap(double T): # max 0.09% error
    cdef double x1 = T
    cdef double x2 = T*T
    cdef double x4 = x2*x2
    cdef double p0 = <double>13017372521.157866
    cdef double p1 = <double>72262011.77066447
    cdef double q0 = <double>292245993.87807804
    cdef double q1 = <double>1190768.9555756454
    cdef double q4 = <double>0.00836425317722044
    cdef double P = p0 - p1*x1 + x4
    cdef double Q = -q0 + q1*x1 - q4*x4
    return P / Q

cdef double nox_rho_sat(double T, double Q): # idx max err but cant be that high
    cdef double rho_liq = nox_rho_satliq(T)
    cdef double rho_vap = nox_rho_satvap(T)
    return (<double>1.0 - Q) * rho_liq + Q * rho_vap

cdef double nox_P_satliq(double T): # max 0.84% error
    cdef double x1 = T
    cdef double x2 = T*T
    cdef double c0 = <double>49628367.373975985
    cdef double c1 = <double>419724.2380649199
    cdef double c2 = <double>913.1756356877407
    return c0 - c1*x1 + c2*x2

cdef double nox_P(double T, double rho): # vapour only, max 0.79% error
    cdef double x1 = T
    cdef double y2 = rho*rho
    cdef double x2y1 = T*T
    cdef double y3 = y2*rho
    cdef double p5 = <double>957.8831858912396
    cdef double p7 = <double>3.5011895479129755
    cdef double q1 = <double>0.018522546839394596
    cdef double P = -p5*y2 + p7*x2y1 + y3
    cdef double Q = q1*x1
    return P / Q

cdef double nox_s_satliq(double P): # max 0.91% error
    cdef double x1 = P
    cdef double x2 = P*P
    cdef double x3 = x2*P
    cdef double p0 = <double>359.3783151517807
    cdef double p1 = <double>103.86623173005904
    cdef double p2 = <double>26.496588420618643
    cdef double q0 = <double>1.0548444415506883
    cdef double q1 = <double>0.13306626678729863
    cdef double P = p0 + p1*x1 - p2*x2 + x3
    cdef double Q = q0 - q1*x1
    return P / Q

cdef double nox_s_satvap(double P): # max 0.47% error
    cdef double x1 = P
    cdef double x2 = P*P
    cdef double p0 = <double>246.10930235262038
    cdef double p1 = <double>40.262395529528995
    cdef double q0 = <double>0.14145003304901121
    cdef double q1 = <double>0.018736153535782087
    cdef double P = p0 - p1*x1 + x2
    cdef double Q = q0 - q1*x1
    return P / Q

cdef double nox_s_sat(double P, double Q): # idx max err but cant be that high
    cdef double s_liq = nox_s_satliq(P)
    cdef double s_vap = nox_s_satvap(P)
    return (<double>1.0 - Q) * s_liq + Q * s_vap

cdef double nox_cp(double T, double P): # vapour only, max 3.6% error
    # blows.
    cdef double x1 = T
    cdef double y1 = P
    cdef double x2 = T*T
    cdef double y2 = P*P
    cdef double p1 = <double>116.51610731426311
    cdef double p2 = <double>7300.76925443315
    cdef double q0 = <double>91.65124829819894
    cdef double q1 = <double>0.5241714826256261
    cdef double q2 = <double>14.921840347891258
    cdef double q5 = <double>0.7146233838229532
    cdef double P = -p1*x1 - p2*y1 + x2
    cdef double Q = -q0 + q1*x1 - q2*y1 + q5*y2
    return P / Q

cdef double nox_cv_satliq(double T): # max 0.24% error
    cdef double x1 = T
    cdef double x2 = T*T
    cdef double p0 = <double>1144775.8711670972
    cdef double p1 = <double>3936.9793029653038
    cdef double q0 = <double>1192.5631799152063
    cdef double q1 = <double>3.787505750839443
    cdef double P = p0 - p1*x1 + x2
    cdef double Q = q0 - q1*x1
    return P / Q

cdef double nox_cv_satvap(double T): # max 1.6% error
    cdef double x1 = T
    cdef double x2 = T*T
    cdef double q0 = <double>1.7249360631879567
    cdef double q1 = <double>0.015009299578479001
    cdef double q2 = <double>2.7648375677840006e-05
    cdef double P = x1
    cdef double Q = -q0 + q1*x1 - q2*x2
    return P / Q

cdef double nox_cv_sat(double T, double Q): # idx max err but cant be that high
    cdef double cv_liq = nox_cv_satliq(T)
    cdef double cv_vap = nox_cv_satvap(T)
    return (<double>1.0 - Q) * cv_liq + Q * cv_vap

cdef double nox_cv(double T, double P): # vapour only, max 1.2% error
    cdef double x1 = T
    cdef double y1 = P
    cdef double x1y1 = T*P
    cdef double y2 = P*P
    cdef double x3 = T*T*T
    cdef double p2 = <double>5789250.582826521
    cdef double p4 = <double>28105.916013494785
    cdef double p5 = <double>4133.783907249727
    cdef double q0 = <double>44609.46603039637
    cdef double q1 = <double>277.5261817149765
    cdef double q2 = <double>4639.100711485678
    cdef double P = p2*y1 - p4*x1y1 + p5*y2 + x3
    cdef double Q = -q0 + q1*x1 - q2*y1
    return P / Q

cdef double nox_h_satliq(double T): # max 1.9% error
    cdef double x1 = T
    cdef double p0 = <double>66.0327027959993
    cdef double q0 = <double>0.007194968322024715
    cdef double q1 = <double>1.882583428861607e-05
    cdef double P = p0 + x1
    cdef double Q = q0 - q1*x1
    return P / Q

cdef double nox_h(double T, double rho): # vapour only, max 0.89% error
    # just lovely.
    cdef double x1 = T
    cdef double y1 = rho
    cdef double p0 = <double>156.6103714457326
    cdef double q0 = <double>0.0009637606915044852
    cdef double q2 = <double>1.2531446548205385e-06
    cdef double P = p0 + x1
    cdef double Q = q0 + q2*y1
    return P / Q

cdef double nox_u_satliq(double T): # max 1.4% error
    cdef double x1 = T
    cdef double x3 = T*T*T
    cdef double x4 = x3*T
    cdef double c0 = <double>7267787.626643519
    cdef double c1 = <double>51741.50859952818
    cdef double c3 = <double>0.6502711777275986
    cdef double c4 = <double>0.0011765913963689713
    return -c0 + c1*x1 - c3*x3 + c4*x4

cdef double nox_u_satvap(double T): # max 0.19% error
    cdef double x1 = T
    cdef double x2 = T*T
    cdef double p0 = <double>39314394868.610306
    cdef double p1 = <double>124682202.67355517
    cdef double q0 = <double>106235.11214047845
    cdef double q1 = <double>335.64679750282
    cdef double P = p0 - p1*x1 + x2
    cdef double Q = q0 - q1*x1
    return P / Q

cdef double nox_u_sat(double T, double Q): # idx max err but cant be that high
    cdef double u_liq = nox_u_satliq(T)
    cdef double u_vap = nox_u_satvap(T)
    return (<double>1.0 - Q) * u_liq + Q * u_vap

cdef double nox_u(double T, double rho): # vapour only, max 0.66% error
    # once again beautiful. thank you energy.
    cdef double x1 = T
    cdef double y1 = rho
    cdef double p0 = <double>226.49887221481575
    cdef double q0 = <double>0.001264621721201892
    cdef double q2 = <double>1.2786597468309012e-06
    cdef double P = p0 + x1
    cdef double Q = q0 + q2*y1
    return P / Q

cdef double nox_Z(double T, double rho): # vapour only, max 1.0% error
    cdef double y1 = rho
    cdef double x1y1 = T*rho
    cdef double y2 = rho*rho
    cdef double c0 = <double>0.998648554728038
    cdef double c2 = <double>0.009846953884435787
    cdef double c4 = <double>2.2992669091214318e-05
    cdef double c5 = <double>2.621170870338741e-06
    return c0 - c2*y1 + c4*x1y1 + c5*y2



cdef struct State:
    double T_t
    double P_c
    double D_f
    double m_l
    double m_v
    double m_g
    double T_g
    double nmol_g
    double Cp_g


cdef double df(State* state):
    cdef double T_t = state.T_t
    cdef double P_c = state.P_c
    cdef double D_f = state.D_f
    cdef double m_l = state.m_l
    cdef double m_v = state.m_v
    cdef double m_g = state.m_g
    cdef double T_g = state.T_g
    cdef double nmol_g = state.nmol_g
    cdef double Cp_g = state.Cp_g
