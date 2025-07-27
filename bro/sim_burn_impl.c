// c my beloved.
#include "sim_burn_impl.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>


typedef signed char       i8;   // 8-bit signed integer (two's complement).
typedef signed short      i16;  // 16-bit signed integer (two's complement).
typedef signed int        i32;  // 32-bit signed integer (two's complement).
typedef signed long long  i64;  // 64-bit signed integer (two's complement).

typedef unsigned char       u8;   // 8-bit unsigned integer.
typedef unsigned short      u16;  // 16-bit unsigned integer.
typedef unsigned int        u32;  // 32-bit unsigned integer.
typedef unsigned long long  u64;  // 64-bit unsigned integer.

typedef float   f32;  // 32-bit floating-point number (IEEE-754).
typedef double  f64;  // 64-bit floating-point number (IEEE-754).


// For discussing maths, cop a legend (shoutout charle):
//
// X0 = initial
// dX = time derivative
// DX = discrete change
//
// X_l = ox liquid
// X_v = ox vapour
// X_o = ox anywhere
// X_f = fuel
// X_g = cc gases
// X_n = new cc gases
//
// X_t = tank
// X_c = cc
// X_w = tank wall (for heat sink)
// X_a = ambient
//
// X_inj = injector
// X_u = upstream (tank-side of injector)
// X_d = downstream (cc-side of injector)
//
// X_nzl = nozzle
//
// X_reg = regression (fuel erosion/vapourisation)


#define PI (3.141592653589793) // [-]
#define PI_4 (0.7853981633974483) // [-]

static f64 circ_area(f64 D) {
    return PI_4 * D*D;
}
static f64 cyl_area(f64 L, f64 D) { // only curved face.
    return L * PI * D;
}
static f64 tube_vol(f64 L, f64 ID, f64 OD) {
    return L * PI_4 * (OD*OD - ID*ID);
}



// Straight off the dome.
#define GAS_CONSTANT (8.31446261815324) // [J/mol/K]



// Nitrous oxide properties from CoolProp, approximated via rational
// polynomials by 'bro/func_approx.py'.

// Valids bounds of all inputs:
//  temperature   263.15 K .. 308.15 K
//  pressure        80 kPa .. 7.2 Mpa
//  sat. pressure    2 MPa .. 7.2 Mpa
//  vap. density   1 kg/m3 .. 325 kg/m3
//  quality              0 .. 1
#define NOX_IN_T(T) (263.15 <= (T) && (T) <= 308.15)
#define NOX_IN_P(P) (0.08 <= (P) && (P) <= 7.2)
#define NOX_IN_Psat(P) (2 <= (P) && (P) <= 7.2)
#define NOX_IN_rho(rho) (1 <= (rho) && (rho) <= 325)
#define NOX_IN_Q(Q) (0.0 <= (Q) && (Q) <= 1.0)


#define NOX_Mw (44.013e-3) // [kg/mol]
#define NOX_R (GAS_CONSTANT / NOX_Mw) // [J/kg/K]

static f64 nox_Tsat(f64 P) { // max 0.13% error
    P *= 1e-6; // Pa -> MPa
    assert(NOX_IN_Psat(P));
    double x1 = P;
    double n0 = +3.6113787984517662;
    double d0 = +0.01676607833192893;
    double d1 = +0.0025292160857822753;
    double Num = n0 + x1;
    double Den = d0 + d1*x1;
    return Num / Den;
}

static f64 nox_rho_satliq(f64 T) { // max 0.68% error
    assert(NOX_IN_T(T));
    f64 x1 = T;
    f64 x2 = T*T;
    f64 n0 = +156087.8754291886;
    f64 n1 = -810.6533977445022;
    f64 d0 = +73.69992119127612;
    f64 d1 = -0.2323813281191897;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

static f64 nox_rho_satvap(f64 T) { // max 0.08% error
    assert(NOX_IN_T(T));
    f64 x1 = T;
    f64 x3 = T*T*T;
    f64 n0 = +33298391.761922084;
    f64 n1 = -204351.1261299974;
    f64 d0 = -686517.2650492929;
    f64 d1 = +3146.20641220234;
    f64 d3 = -0.009715066047700149;
    f64 Num = n0 + n1*x1 + x3;
    f64 Den = d0 + d1*x1 + d3*x3;
    return Num / Den;
}

static f64 nox_P_satliq(f64 T) { // max 0.84% error
    assert(NOX_IN_T(T));
    f64 x1 = T;
    f64 x2 = T*T;
    f64 c0 = +49628368.03406795;
    f64 c1 = -419724.2426924475;
    f64 c2 = +913.1756437857401;
    return c0 + c1*x1 + c2*x2;
}

static f64 nox_P(f64 T, f64 rho) { // vapour only, max 0.85% error
    assert(NOX_IN_T(T));
    assert(NOX_IN_rho(rho));
    f64 x1 = T;
    f64 y2 = rho*rho;
    f64 x2y1 = T*T;
    f64 y3 = y2*rho;
    f64 n5 = -1004.4435204505293;
    f64 n7 = +3.7264490783988635;
    f64 d1 = +0.019777885554525112;
    f64 Num = n5*y2 + n7*x2y1 + y3;
    f64 Den = d1*x1;
    return Num / Den;
}

static f64 nox_s_satliq(f64 P) { // max 0.83% error
    P *= 1e-6; // Pa -> MPa
    assert(NOX_IN_Psat(P));
    f64 x1 = P;
    f64 x2 = P*P;
    f64 x6 = x2*x2*x2;
    f64 n0 = -330735366.98577213;
    f64 d0 = -817420.0670422539;
    f64 d1 = +158470.39606104503;
    f64 d2 = -14678.975863932443;
    f64 Num = -n0;
    f64 Den = d0 + d1*x1 + d2*x2 + x6;
    return Num / Den;
}

static f64 nox_s_satvap(f64 P) { // max 0.47% error
    P *= 1e-6; // Pa -> MPa
    assert(NOX_IN_Psat(P));
    f64 x1 = P;
    f64 x2 = P*P;
    f64 n0 = +246.10928756012322;
    f64 n1 = -40.26239399448462;
    f64 d0 = +0.14145002404877624;
    f64 d1 = -0.018736152701492044;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

static f64 nox_cp(f64 T, f64 P) { // vapour only, max 3.6% error
    P *= 1e-6; // Pa -> MPa
    assert(NOX_IN_T(T));
    assert(NOX_IN_P(P));
    // blows.
    f64 x1 = T;
    f64 y1 = P;
    f64 x2 = T*T;
    f64 y2 = P*P;
    f64 n1 = -116.51699084548176;
    f64 n2 = -7300.734145907978;
    f64 d0 = -91.65103697146496;
    f64 d1 = +0.5241696296156855;
    f64 d2 = -14.921756479203275;
    f64 d5 = +0.7146187152094382;
    f64 Num = n1*x1 + n2*y1 + x2;
    f64 Den = d0 + d1*x1 + d2*y1 + d5*y2;
    return Num / Den;
}

static f64 nox_cv_satliq(f64 T) { // max 0.24% error
    assert(NOX_IN_T(T));
    f64 x1 = T;
    f64 x2 = T*T;
    f64 n0 = +1144787.6651075622;
    f64 n1 = -3937.01681226655;
    f64 d0 = +1192.576384943476;
    f64 d1 = -3.7875478741887108;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

static f64 nox_cv_satvap(f64 T) { // max 1.3% error
    assert(NOX_IN_T(T));
    f64 x1 = T;
    f64 x2 = T*T;
    f64 d0 = -1.7249439211880886;
    f64 d1 = +0.015009354557928997;
    f64 d2 = -2.7648472161050912e-05;
    f64 Num = x1;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;
}

static f64 nox_cv(f64 T, f64 P) { // vapour only, max 1.2% error
    P *= 1e-6; // Pa -> MPa
    assert(NOX_IN_T(T));
    assert(NOX_IN_P(P));
    f64 x1 = T;
    f64 y1 = P;
    f64 x1y1 = T*P;
    f64 y2 = P*P;
    f64 x3 = T*T*T;
    f64 n2 = +5789256.963839073;
    f64 n4 = -28105.935930542626;
    f64 n5 = +4134.052836043407;
    f64 d0 = -44609.45406542292;
    f64 d1 = +277.52613563434414;
    f64 d2 = -4639.098422516753;
    f64 Num = n2*y1 + n4*x1y1 + n5*y2 + x3;
    f64 Den = d0 + d1*x1 + d2*y1;
    return Num / Den;
}

static f64 nox_h_satliq(f64 T) { // max 2.0% error
    assert(NOX_IN_T(T));
    f64 x1 = T;
    f64 d0 = +0.0056002005901963905;
    f64 d1 = -1.4442735468669312e-05;
    f64 Num = x1;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

static f64 nox_h_satvap(f64 T) { // max 0.34% error
    assert(NOX_IN_T(T));
    double x1 = T;
    double x2 = T*T;
    double n0 = +44029181155.05621;
    double n1 = -138886400.2908723;
    double d0 = +106633.1549244389;
    double d1 = -334.3462973733325;
    double Num = n0 + n1*x1 + x2;
    double Den = d0 + d1*x1;
    return Num / Den;
}

static f64 nox_h(f64 T, f64 rho) { // vapour only, max 0.76% error
    assert(NOX_IN_T(T));
    assert(NOX_IN_rho(rho));
    // just lovely.
    f64 x1 = T;
    f64 y1 = rho;
    f64 d0 = +0.00038666795882217524;
    f64 d1 = +8.229213007432222e-07;
    f64 d2 = +8.20724847683098e-07;
    f64 Num = x1;
    f64 Den = d0 + d1*x1 + d2*y1;
    return Num / Den;
}

static f64 nox_u_satliq(f64 T) { // max 1.4% error
    assert(NOX_IN_T(T));
    f64 x1 = T;
    f64 x3 = T*T*T;
    f64 x4 = x3*T;
    f64 c0 = -7267788.558102235;
    f64 c1 = +51741.51506234207;
    f64 c3 = -0.6502712552305997;
    f64 c4 = +0.0011765915303324403;
    return c0 + c1*x1 + c3*x3 + c4*x4;
}

static f64 nox_u_satvap(f64 T) { // max 0.19% error
    assert(NOX_IN_T(T));
    f64 x1 = T;
    f64 x2 = T*T;
    f64 n0 = +34599947743.215485;
    f64 n1 = -109730783.13591044;
    f64 d0 = +93495.74130581485;
    f64 d1 = -295.39713418629657;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

static f64 nox_u(f64 T, f64 rho) { // vapour only, max 0.54% error
    assert(NOX_IN_T(T));
    assert(NOX_IN_rho(rho));
    // once again beautiful. thank you energy.
    f64 x1 = T;
    f64 y1 = rho;
    f64 d0 = +0.00037742028086887655;
    f64 d1 = +1.1447255525542155e-06;
    f64 d2 = +7.272303965256336e-07;
    f64 Num = x1;
    f64 Den = d0 + d1*x1 + d2*y1;
    return Num / Den;
}

static f64 nox_Z(f64 T, f64 rho) { // vapour only, max 1.0% error
    assert(NOX_IN_T(T));
    assert(NOX_IN_rho(rho));
    f64 y1 = rho;
    f64 x1y1 = T*rho;
    f64 y2 = rho*rho;
    f64 c0 = +0.9984208759508151;
    f64 c2 = -0.009865255977849555;
    f64 c4 = +2.309181429551549e-05;
    f64 c5 = +2.5526116512486693e-06;
    return c0 + c2*y1 + c4*x1y1 + c5*y2;
}



// CEA results from rocketcea, approximated via rational
// polynomials by 'bro/func_approx.py'.

// Valids bounds of all inputs:
//  pressure     80 kPa .. 7 Mpa
//  ox-fuel ratio   0.5 .. 13

static f64 cea_T__high(f64 P, f64 ofr) { // max 4.3% error
    f64 x1 = P;
    f64 y1 = ofr;
    f64 x1y1 = P*ofr;
    f64 y2 = ofr*ofr;
    f64 x3 = P*P*P;
    f64 x2y1 = x1y1*P;
    f64 n0 = -23.39955473159082;
    f64 n1 = +2.1207746242104513;
    f64 n2 = +5.921369062404011;
    f64 d1 = +0.0018229851740627402;
    f64 d4 = -0.000293794779021038;
    f64 d5 = +0.0004593559262770208;
    f64 d6 = -9.851749432640628e-06;
    f64 d7 = +2.294535003217659e-05;
    f64 Num = n0 + n1*x1 + n2*y1 + y2;
    f64 Den = d1*x1 + d4*x1y1 + d5*y2 + d6*x3 + d7*x2y1;
    return Num / Den;
}
static f64 cea_T__low(f64 P, f64 ofr) { // max 3.9% error
    f64 x1 = P;
    f64 y1 = ofr;
    f64 x1y1 = P*ofr;
    f64 y2 = ofr*ofr;
    f64 x2y1 = x1y1*P;
    f64 x1y2 = x1y1*ofr;
    f64 n0 = -0.3308903646269448;
    f64 n2 = +7.586197253157754;
    f64 n4 = +8.808819483900445;
    f64 d1 = +0.00022361880493252257;
    f64 d2 = +0.00944138717224415;
    f64 d4 = +0.007895570851432925;
    f64 d5 = -0.0011657313243630958;
    f64 d7 = -1.3745327377942885e-05;
    f64 d8 = -0.001026130067031718;
    f64 Num = n0 + n2*y1 + n4*x1y1 + y2;
    f64 Den = d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2;
    return Num / Den;
}
static f64 cea_T(f64 P, f64 ofr) { // see constituents.
    P *= 1e-6; // Pa -> MPa
    // Different approxs for different input regions.
    if (ofr >= 4)
        return cea_T__high(P, ofr);
    return cea_T__low(P, ofr);
}

static f64 cea_cp__high(f64 P, f64 ofr) { // max 4.8% error
    f64 x1 = P;
    f64 y1 = ofr;
    f64 x1y1 = P*ofr;
    f64 x3 = P*P*P;
    f64 x2y1 = x1y1*P;
    f64 x1y2 = x1y1*ofr;
    f64 y3 = ofr*ofr*ofr;
    f64 n1 = +3802.3766890029965;
    f64 n4 = -959.1089167566258;
    f64 n6 = +0.9740826587244922;
    f64 n7 = -3.5480311720693734;
    f64 n8 = +75.4934686487755;
    f64 d0 = +0.06375088655715791;
    f64 d1 = +2.0937979929620574;
    f64 d2 = -0.012683649993537984;
    f64 d4 = -0.5040092283785854;
    f64 d8 = +0.03403190253379037;
    f64 d9 = +0.00023048707705274537;
    f64 Num = n1*x1 + n4*x1y1 + n6*x3 + n7*x2y1 + n8*x1y2 + y3;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d8*x1y2 + d9*y3;
    return Num / Den;
}
static f64 cea_cp__low_right(f64 P, f64 ofr) { // max 5.8% error
    f64 x1 = P;
    f64 y1 = ofr;
    f64 x1y1 = P*ofr;
    f64 y2 = ofr*ofr;
    f64 x1y2 = x1y1*ofr;
    f64 n0 = +4.138832624431316;
    f64 n1 = +0.29817405524316326;
    f64 n2 = -3.2575841375358316;
    f64 d0 = +0.0005884213088340639;
    f64 d1 = +7.309846342943449e-05;
    f64 d2 = -0.0007528849169182324;
    f64 d4 = -4.3580339523010414e-05;
    f64 d5 = +0.0003868623137207404;
    f64 d8 = +1.9187426863527176e-05;
    f64 Num = n0 + n1*x1 + n2*y1 + y2;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d8*x1y2;
    return Num / Den;
}
static f64 cea_cp__low_left(f64 P, f64 ofr) { // max 11.83% error (skull emoji)
    f64 x1 = P;
    f64 y1 = ofr;
    f64 x2 = P*P;
    f64 x1y1 = P*ofr;
    f64 y2 = ofr*ofr;
    f64 x2y1 = x1y1*P;
    f64 x1y2 = x1y1*ofr;
    f64 y3 = y2*ofr;
    f64 n0 = +3.0551837534331043;
    f64 n1 = +1.1168085957129472;
    f64 n2 = -3.057032048465364;
    f64 n3 = -0.08801722733929103;
    f64 n4 = -0.2480662508580779;
    f64 d0 = +0.00035410768902353467;
    f64 d1 = +0.00019003775520538854;
    f64 d2 = -0.0004712179241529132;
    f64 d4 = -0.00012521491909272612;
    f64 d5 = +0.00021512281760482563;
    f64 d7 = +4.846442773758666e-07;
    f64 d8 = +3.615723753207365e-05;
    f64 d9 = +2.406864519884499e-05;
    f64 Num = n0 + n1*x1 + n2*y1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2 + d9*y3;
    return Num / Den;
}
static f64 cea_cp(f64 P, f64 ofr) { // see constituents.
    P *= 1e-6; // Pa -> MPa
    // Different approxs for different input regions.
    if (ofr >= 4)
        return cea_cp__high(P, ofr);
    if (P >= 1)
        return cea_cp__low_right(P, ofr);
    return cea_cp__low_left(P, ofr);
}

static f64 cea_Mw(f64 P, f64 ofr) { // max 4.9% error
    P *= 1e-6; // Pa -> MPa
    f64 x1 = P;
    f64 y1 = ofr;
    f64 x2 = P*P;
    f64 x1y1 = P*ofr;
    f64 y2 = ofr*ofr;
    f64 n0 = +0.1841818441210476;
    f64 n1 = +0.7780884860427217;
    f64 n3 = +0.005825775742224297;
    f64 n4 = +0.14145416999491983;
    f64 d1 = +64.3339970083186;
    f64 d2 = +66.80439701942177;
    f64 d3 = +0.15251463574592666;
    f64 d5 = +30.325772888462538;
    f64 Num = n0 + n1*x1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d1*x1 + d2*y1 + d3*x2 + d5*y2;
    return Num / Den;
}



// Paraffin and Nox regression rate constants.

#define RR_a0 (1.55e-4)
#define RR_n (0.5)



// Couple more constants.

#define Dt (0.001)      // [s], discrete calculus over time.
#define DT (0.02)       // [K], discrete calculus over temperature.
#define NEGLIGIBLE_MASS (0.001) // [kg], assume nothing if <= this.


static void step_state(implState* s, int i) {
    // Current state variables.
    f64 T_t = s->T_t[i - 1];
    f64 m_l = s->m_l[i - 1];
    f64 m_v = s->m_v[i - 1];
    f64 D_f = s->D_f[i - 1];
    f64 m_g = s->m_g[i - 1];
    f64 nmol_g = s->nmol_g[i - 1];
    f64 T_g = s->T_g[i - 1];
    f64 Cp_g = s->Cp_g[i - 1];

    // Reconstruct some cc/fuel state.
    f64 V_f = tube_vol(s->L_f, D_f, s->D_c);
    f64 A_f = cyl_area(s->L_f, D_f);
    f64 V_c = s->Vempty_c - V_f;
    f64 m_f = s->rho_f * V_f;

    // Reconstruct some cc gas state.
    f64 cp_g = Cp_g / m_g;
    f64 Mw_g = m_g / nmol_g;
    f64 R_g = GAS_CONSTANT / Mw_g;
    f64 y_g = cp_g / (cp_g - R_g);
    f64 rho_g = m_g / V_c;

    // Assuming combustion gases are ideal:
    //  P*V = m*R*T
    //  P = R*T * m/V
    f64 P_c = R_g * T_g * rho_g;


    // Properties determined by injector flow:
    f64 dm_l;
    f64 dm_v;
    f64 dm_inj;
    f64 dT_t;

    // Liquid draining while there's any liquid in the tank.
    if (m_l > NEGLIGIBLE_MASS) {

        // Find injector flow rate.

        f64 P_u = nox_P_satliq(T_t); // tank at saturated pressure.
        f64 P_d = P_c;
        printf("P_u %f\n", P_u);

        if (P_u <= P_d)
            goto NO_INJECTOR_FLOW;
            // id love to see python try to handle a goto.

        // Single-phase incompressible model (with Beta = 0):
        // (assuming upstream density as the "incompressible" density)
        f64 rho_u = nox_rho_satliq(T_t);
        f64 mdot_SPI = s->Cd_inj * s->A_inj * sqrt(2 * rho_u * (P_u - P_d));

        // Homogenous equilibrium model:
        // (assuming only saturated liquid leaving from upstream)
        f64 s_u = nox_s_satliq(P_u);
        f64 s_d_l = nox_s_satliq(P_d);
        f64 s_d_v = nox_s_satvap(P_d);
        f64 x_d = (s_u - s_d_l) / (s_d_v - s_d_l);
        f64 h_u = nox_h_satliq(T_t);
        f64 T_d = nox_Tsat(P_d); // since our prop funcs expect temp.
        f64 h_d_l = nox_h_satliq(T_d);
        f64 h_d_v = nox_h_satvap(T_d);
        f64 h_d = (1.0 - x_d)*h_d_l + x_d*h_d_v;
        f64 rho_d_l = nox_rho_satliq(T_d);
        f64 rho_d_v = nox_rho_satvap(T_d);
        f64 rho_d = (1.0 - x_d)*rho_d_l + x_d*rho_d_v;
        f64 mdot_HEM = s->Cd_inj * s->A_inj * rho_d * sqrt(2 * (h_u - h_d));

        // Generalised non-homogenous non-equilibrium model:
        // (assuming that P_sat is upstream saturation, and so is alaways
        //  =P_u since its saturated?????? this means that the dyer model
        //  is always just an arithmetic mean of spi and hem when the tank
        //  is saturated but hey maybe thats what we're looking for).
        //
        //  kappa = sqrt((P_u - P_d) / (P_sat - P_d))
        //  kappa = sqrt((P_u - P_d) / (P_u - P_d))
        //  kappa = sqrt(1) = 1
        f64 kappa = 1;
        f64 k_NHNE = 1 / (1 + kappa);
        f64 dm_inj = mdot_SPI * (1 - k_NHNE) + mdot_HEM * k_NHNE;


        // To determine temperature and vapourised mass derivatives,
        // we're going to have to use: our brain.
        //  V = const.
        //  m_l / rho_l + m_v / rho_v = const.
        //  d/dt (m_l / rho_l + m_v / rho_v) = 0
        //  d/dt (m_l / rho_l) + d/dt (m_v / rho_v) = 0
        //  0 = (dm_l * rho_l - m_l * drho_l) / rho_l**2  [quotient rule]
        //    + (dm_v * rho_v - m_v * drho_v) / rho_v**2
        // dm_l = -dm_inj - dm_v  [injector and vapourisation]
        //  0 = ((-dm_inj - dm_v) * rho_l - m_l * drho_l) / rho_l**2
        //    + (dm_v * rho_v - m_v * drho_v) / rho_v**2
        //  0 = -dm_inj / rho_l
        //    - dm_v / rho_l
        //    - m_l * drho_l / rho_l**2
        //    + dm_v / rho_v
        //    - m_v * drho_v / rho_v**2
        //  0 = dm_v * (1/rho_v - 1/rho_l)
        //    - dm_inj / rho_l
        //    - m_l * drho_l / rho_l**2
        //    - m_v * drho_v / rho_v**2
        // drho = d/dt (rho) = d/dT (rho) * dT/dt  [chain rule]
        // drhodT = d/dT (rho)
        //  0 = dm_v * (1/rho_v - 1/rho_l)
        //    - dm_inj / rho_l
        //    - m_l * dT * drhodT_l / rho_l**2
        //    - m_v * dT * drhodT_v / rho_v**2
        //  dm_v = (dm_inj / rho_l
        //         + m_l * dT * drhodT_l / rho_l**2
        //         + m_v * dT * drhodT_v / rho_v**2
        //         ) / (1/rho_v - 1/rho_l)
        //  dm_v = dm_inj / rho_l / (1/rho_v - 1/rho_l)
        //       + dT / (1/rho_v - 1/rho_l) * (m_l * drhodT_l / rho_l**2
        //                                   + m_v * drhodT_v / rho_v**2)
        // let:
        //   foo = dm_inj / rho_l / (1/rho_v - 1/rho_l)
        //   bar = (m_l * drhodT_l / rho_l**2
        //        + m_v * drhodT_v / rho_v**2) / (1/rho_v - 1/rho_l)
        //  dm_v = foo + dT * bar
        // So, dm_v depends on dT, but also vice versa:
        //  d/dt (U) = -dm_inj * h_l  [first law of thermodynamics, adiabatic]
        //  d/dt (U_w + U_l + U_v) = -dm_inj * h_l
        //  d/dt (m_w*u_w) + d/dt (m_l*u_l) + d/dt (m_v*u_v) = -dm_inj * h_l
        //  -dm_inj * h_l = dm_w*u_w + m_w*du_w
        //                + dm_l*u_l + m_l*du_l
        //                + dm_v*u_v + m_v*du_v
        // dm_w = 0  [wall aint going anywhere]
        // dm_l = -dm_v - dm_inj  [same as earlier]
        //  -dm_inj * h_l = m_w*du_w + m_l*du_l + m_v*du_v
        //                + (-dm_v - dm_inj) * u_l
        //                + dm_v*u_v
        //  dm_inj * (u_l - h_l) = m_w*du_w + m_l*du_l + m_v*du_v
        //                       - dm_v*u_l
        //                       + dm_v*u_v
        //  dm_inj * (u_l - h_l) = m_w*du_w + m_l*du_l + m_v*du_v
        //                       + dm_v * (u_v - u_l)
        // du = d/dt (u) = d/dT (u) * dT/dt
        // also note:
        //   u = int (cv) dT
        //   d/dT (u) = cv
        // therefore:
        //   du = dT * cv
        //  dm_inj * (u_l - h_l) = dT * (m_w*cv_w + m_l*cv_l + m_v*cv_v)
        //                       + dm_v * (u_v - u_l)
        // let: Cv = m_w*cv_w + m_l*cv_l + m_v*cv_v
        //  dm_inj * (u_l - h_l) = dT * Cv + dm_v * (u_v - u_l)
        //  dT * Cv = dm_inj * (u_l - h_l) + dm_v * (u_l - u_v)
        // i think conceptually this makes sense as:
        //  internal energy change = boundary work + phase change energy
        // which checks out, so: bitta simul lets substitute
        //  dT * Cv = dm_inj * (u_l - h_l) + (foo + dT * bar) * (u_l - u_v)
        //  dT * Cv - dT * bar * (u_l - u_v) = dm_inj * (u_l - h_l) + foo * (u_l - u_v)
        //  dT = (dm_inj * (u_l - h_l) + foo * (u_l - u_v))
        //     / (Cv - bar * (u_l - u_v))
        // dandy.

        f64 rho_l = nox_rho_satliq(T_t);
        f64 rho_v = nox_rho_satvap(T_t);
        f64 drhodT_l = (nox_rho_satliq(T_t + DT) - rho_l) / DT;
        f64 drhodT_v = (nox_rho_satvap(T_t + DT) - rho_v) / DT;

        f64 Cv_l = m_l * nox_cv_satliq(T_t);
        f64 Cv_v = m_v * nox_cv_satvap(T_t);
        f64 Cv = Cv_l + Cv_v + s->C_w;

        f64 h_l = nox_h_satliq(T_t);
        f64 u_l = nox_u_satliq(T_t);
        f64 u_v = nox_u_satvap(T_t);

        f64 foo = dm_inj / rho_l / (1/rho_v - 1/rho_l);
        f64 bar = (m_l * drhodT_l / (rho_l*rho_l)
                 + m_v * drhodT_v / (rho_v*rho_v))
                / (1/rho_v - 1/rho_l);

        dT_t = (dm_inj * (u_l - h_l) + foo * (u_l - u_v))
             / (Cv - bar * (u_l - u_v));

        dm_v = foo + dT_t * bar;
        dm_l = -dm_inj - dm_v;


    // Otherwise vapour draining.
    } else if (m_v > NEGLIGIBLE_MASS) {
        dm_l = 0.0; // liquid mass is ignored hence fourth (big word init).

        // During this period, temperature and density are used to fully
        // define the state (density is simple due to fixed volume).
        f64 rho_v = m_v / s->V_t;

        // Due to numerical inaccuracy, might technically have the properties
        // of a saturated mixture so just pretend its a saturated vapour.
        f64 rhosat_v = nox_rho_satvap(T_t);
        if (rho_v >= rhosat_v)
            rho_v = rhosat_v;


        // Find injector flow rate.

        f64 P_u = nox_P(T_t, rho_v);
        f64 P_d = P_c;

        if (P_u <= P_d)
            goto NO_INJECTOR_FLOW;

        // Technically gamma but use 'y' for file size reduction.
        f64 y_u = nox_cp(T_t, P_u) / nox_cv(T_t, P_u);
        // Use compressibility factor to account for non-ideal gas.
        f64 Z_u = nox_Z(T_t, rho_v);

        // Real compressible flow through an injector, with both
        // choked and unchoked possibilities:
        f64 Pr_crit = pow(2 / (y_u + 1), y_u / (y_u - 1));
        f64 Pr_rec = P_d / P_u;
        f64 Pterm;
        if (Pr_rec <= Pr_crit) { // choked.
            Pterm = pow(2 / (y_u + 1), (y_u + 1) / (y_u - 1));
        } else { // unchoked.
            Pterm = pow(Pr_rec, 2 / y_u) - pow(Pr_rec, (y_u + 1) / y_u);
            Pterm *= 2 / (y_u - 1);
        }
        f64 dm_inj = s->Cd_inj * s->A_inj * P_u * sqrt(y_u / Z_u / NOX_R / T_t * Pterm);

        // Mass only leaves through injector, and no state change.
        dm_v = -dm_inj;


        // Back to the well.
        //  d/dt (U) = -dm_inj * h  [first law of thermodynamics, adiabatic]
        //  d/dt (U_w + U) = -dm_inj * h  [no suffix is the non-saturated vapour in the tank]
        //  d/dt (m_w*u_w) + d/dt (m*u) = -dm_inj * h
        //  -dm_inj * h = dm_w*u_w + m_w*du_w
        //              + dm*u + m*du
        // dm_w = 0  [wall aint going anywhere]
        // dm = -dm_inj  [only mass change is from injector]
        //  -dm_inj * h = m_w * du_w
        //              - dm_inj * u
        //              + m * du
        //  dm_inj * (u - h) = m_w * du_w + m * du
        // du = dT * cv  [previously derived]
        //  dm_inj * (u - h) = dT * (m_w * cv_w + m * cv)
        // let: Cv = m_w * cv_w + m * cv
        //  dm_inj * (u - h) = dT * Cv
        // => dT = dm_inj * (u - h) / Cv
        // which makes sense, since only energy change is due to lost flow work.

        f64 u_u = nox_u(T_t, rho_v);
        f64 h_u = nox_h(T_t, rho_v);

        f64 Cv = s->C_w + m_v * nox_cv(T_t, P_u);

        dT_t = dm_inj * (u_u - h_u) / Cv;


    // No oxidiser left.
    } else {
      NO_INJECTOR_FLOW:;
        dm_l = 0.0;
        dm_v = 0.0;
        dm_inj = 0.0;
        dT_t = 0.0;
    }


    // Do fuel regression.
    f64 dD_f;
    f64 dV_f;
    f64 dm_reg;
    // Gotta be fuel left.
    if (m_f > NEGLIGIBLE_MASS) {

        // Get oxidiser mass flux through the fuel grain.
        f64 Gox = dm_inj / circ_area(D_f);
        // Find regression rate from empirical parameters (and ox mass flux).
        f64 rr_rdot = RR_a0 * pow(Gox, RR_n);

        // Fuel mass and diameter change from rdot:
        dD_f = 2 * rr_rdot;
        dV_f = A_f * rr_rdot;
        dm_reg = s->rho_f * dV_f;

    // No fuel.
    } else {
        dD_f = 0.0;
        dV_f = 0.0;
        dm_reg = 0.0;
    }


    // Do nozzle flow.
    f64 dm_out;
    if (P_c <= s->P_a) {
        // Model the nozzle as an injector, using ideal compressible
        // flow and both choked and unchoked possibilities:
        f64 Pr_crit = pow(2 / (y_g + 1), y_g / (y_g - 1));
        f64 Pr_rec = s->P_a / P_c;
        f64 Pterm;
        if (Pr_rec <= Pr_crit) { // choked.
            Pterm = pow(2 / (y_g + 1), (y_g + 1) / (y_g - 1));
        } else { // unchoked.
            Pterm = pow(Pr_rec, 2 / y_g) - pow(Pr_rec, (y_g + 1) / y_g);
            Pterm *= 2 / (y_g - 1);
        }
        dm_out = s->Cd_nzl * s->A_nzl * P_c * sqrt(y_g / R_g / T_g * Pterm);
    } else {
        // Assume no flow through nozzle if no pressure ratio.
        dm_out = 0.0;
    }


    // Gases in the chamber is just entering - exiting.
    f64 dm_g = dm_inj + dm_reg - dm_out;


    // Change in cc gas properties due to added gas.
    f64 T_n;
    f64 Mw_n;
    f64 cp_n;
    f64 dm_n = dm_reg + dm_inj; // new gases is just fuel+ox.

    // Combustion occurs if there is both fuel and oxidiser.
    if (dm_reg != 0 && dm_inj != 0) {
        // Instantaneous oxidiser-fuel ratio.
        f64 ofr = dm_inj / dm_reg;

        // If ofr too low, our cea approxes dont work and there would be
        // very little comb anyway, so assume none.
        if (ofr < 0.5)
            goto NO_COMBUSTION;

        // Do cea to find combustion properties.
        T_n = cea_T(P_c, ofr);
        Mw_n = cea_Mw(P_c, ofr);
        cp_n = cea_cp(P_c, ofr);
    // Otherwise non-combusting oxidiser.
    } else if (dm_inj != 0) {
      NO_COMBUSTION:;

        // No combustion but chamber gas changes due to oxidiser. Note this is
        // assuming isothermal mass transfer, so using tank temperature but
        // with current chamber pressure.
        T_n = T_t;
        Mw_n = NOX_Mw;
        cp_n = nox_cp(T_t, P_c);

    } else {
        T_n = 0.0;
        Mw_n = 0.0;
        cp_n = 0.0;
    }

    // Change in any mass-specific property for a reservoir with
    // flow in and out:
    //  d/dt (m*p) = dm_in * p_in - dm_out * p

    // Change in moles:
    //  dn = d/dt (n_n) - d/dt (n_out)
    //  dn = dm_n / Mw_n - dm_out / Mw
    f64 dnmol_g = dm_n / Mw_n - dm_out / Mw_g;

    // Change in specific heat:
    f64 dCp_g = dm_n * cp_n - dm_out * cp_g;

    // Change in temperature:
    //  d/dt (m * cp * T) = dm_n * cp_n * T_n - dm_out * cp * T
    //  d/dt (m * cp * T) = dm_n * cp_n * T_n - dm_out * cp * T
    //  d/dt (m * cp) * T + m*cp * dT = dm_n * cp_n * T_n - dm_out * cp * T
    //  dCp * T + Cp * dT = dm_n * cp_n * T_n - dm_out * cp * T
    //  Cp * dT = dm_n * cp_n * T_n - dm_out * cp * T - dCp * T
    //  dT = (dm_n * cp_n * T_n - dm_out * cp * T - dCp * T) / Cp
    f64 dT_g = (dm_n * cp_n * T_n - dm_out * cp_g * T_g - dCp_g * T_g) / Cp_g;


    // While I generally make a point not to use explicit euler
    // (its just not it), this system performs poorly under other
    // methods since it is not smooth. So, explicit euler it is.
    s->T_t[i]    = s->T_t[i - 1]    + Dt * dT_t;
    s->m_l[i]    = s->m_l[i - 1]    + Dt * dm_l;
    s->m_v[i]    = s->m_v[i - 1]    + Dt * dm_v;
    s->D_f[i]    = s->D_f[i - 1]    + Dt * dD_f;
    s->m_g[i]    = s->m_g[i - 1]    + Dt * dm_g;
    s->nmol_g[i] = s->nmol_g[i - 1] + Dt * dnmol_g;
    s->T_g[i]    = s->T_g[i - 1]    + Dt * dT_g;
    s->Cp_g[i]   = s->Cp_g[i - 1]   + Dt * dCp_g;
}



i32 sim_burn_impl(implState* s) {
    // Calculate initial state.

    // Assuming ox tank at ambient temperature and a saturated mixture.
    f64 T0_t = s->T_a;
    f64 V0_l = s->V_t * s->vff0_o;
    f64 V0_v = s->V_t - V0_l;
    f64 m0_l = V0_l * nox_rho_satliq(T0_t);
    f64 m0_v = V0_v * nox_rho_satvap(T0_t);

    // Fuel inner diameter explicit param.
    f64 D0_f = s->D0_f;

    // Combustion chamber initially filled with ambient properties.
    f64 V0_c = s->Vempty_c - tube_vol(s->L_f, D0_f, s->D_c);
    f64 m0_g = s->rho_a * V0_c;
    f64 nmol0_g = m0_g / s->Mw_a;
    f64 T0_g = s->T_a;
    f64 Cp0_g = m0_g * s->cp_a;

    // Chuck into the state.
    if (s->max_count < 1)
        return 0;
    s->T_t[0] = T0_t;
    s->m_l[0] = m0_l;
    s->m_v[0] = m0_v;
    s->D_f[0] = D0_f;
    s->m_g[0] = m0_g;
    s->nmol_g[0] = nmol0_g;
    s->T_g[0] = T0_g;
    s->Cp_g[0] = Cp0_g;


    // TODO: figure out what is considered the termination of the burn.
    i64 count = 1; // init already set.
    while (count < s->max_count) { // dont buffer overflow.
        step_state(s, count);
        count += 1;

        // TODO: remove lmao.
        i64 i = count - 1;
        if (s->m_l[i] <= NEGLIGIBLE_MASS && (s->m_v[i] - s->m_v[i - 1] <= NEGLIGIBLE_MASS))
            break;
    }
    return count;
}
