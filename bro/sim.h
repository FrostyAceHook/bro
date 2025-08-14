#ifndef SIM_H_
#define SIM_H_


// For discussing maths, cop a legend (shoutout charle):
//
// X0 = initial value.
// dX = time derivative.
// dXdY = derivative of X w.r.t Y.
// DX = discrete change over time.
//
// X_r = rocket property.
// X_a = ambient property.
// X_t = tank property.
// X_tw = tank wall property.
// X_mov = main oxidiser valve property.
// X_inj = injector property.
// X_c = combustion chamber property.
// X_cw = combustion chamber wall property.
// X_nzl = nozzle property.
// X_throat = nozzle throat property.
// X_exit = nozzle exit property.
// X_locked = locked section property.
//
// X_l = liquid oxidiser property.
// X_v = vapour oxidiser property.
// X_o = general oxidiser property.
// X_u = orifice upstream property.
// X_d = orifice downstream property.
// X_f = fuel property.
// X_reg = fuel regression property.
// X_g = cc gases property.
// X_n = added cc gases property.
//
// Legend for 'X':
// A = area.
// a = speed of sound.
// acc = acceleration.
// alt = altitude relative to sea-level.
// C = heat capacity of a solid.
// c = mass-specific heat capacity of a solid.
// CD = drag coefficient.
// Cd = discharge coefficient.
// com = centre of mass relative to top, where positive is fin-wards.
// Cp = constant pressure heat capacity.
// cp = constant pressure mass-specific heat capacity.
// Cv = constant volume heat capacity.
// cv = constant volume mass-specific heat capacity.
// D = diameter.
// eps = nozzle exit area to throat area ratio.
// F = force.
// G = mass flux.
// g = (downwards) acceleration due to gravity.
// H = enthalpy.
// h = mass-specific enthalpy.
// ID = inner diameter.
// Ivac = specific impulse in vaccuum.
// L = length.
// m = mass.
// mach = mach number (velocity as a multiple of the current speed of sound).
// Mw = molar mass.
// N = number of moles.
// OD = outer diameter.
// ofr = oxidiser-fuel ratio.
// P = pressure.
// Pr = pressure ratio.
// PCPP = pre-combustion chamber pressure proportion.
// Pcrit = critical point pressure.
// Psat = saturated pressure.
// Ptrip = triple point pressure.
// R = specific gas constant.
// rho = mass density.
// rhosat = saturated mass density.
// S = entropy.
// s = mass-specific entropy.
// sf = safety factor.
// T = temperature.
// t = time.
// th = thickness.
// Tcrit = critical point temperature.
// Tsat = saturated temperature.
// Ttrip = triple point temperature.
// U = internal energy.
// u = mass-specific internal energy.
// V = volume.
// v = mass-specific volume.
// vel = velocity.
// vff = volumetric fill fraction.
// x = saturated mixture vapour quality.
// y = ratio of specific heats (gamma).
// Ys = yield strength.
// Z = compressibility factor.


// Private types:
typedef struct broRunning { // time-dependant constantly-updating parameters.
    // NOTE: any changes to this struct must be mirrored in the cython bridging
    //       struct.
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
} broRunning;


// Public types:
typedef double Input;   // independant input parameter
typedef double Output;  // generated output parameter.

typedef struct broState {
    // NOTE: any changes to this struct must be mirrored in the cython bridging
    //       struct.

    Input target_apogee;

    Input m_locked;
    Input L_locked;
    Input com_locked;

    Input D_r;
    Input alt0_r;
    Output A_r;

    Input T_a;

    Input L_tw;
    Input rho_tw;
    Input Ys_tw;
    Input c_tw;
    Input sf_tw;
    Output V_t;
    Output m_tw;
    Output C_tw;

    Input vff0_l;

    Input m_mov;
    Input L_mov;
    Input com_mov;

    Input m_inj;
    Input L_inj;
    Input com_inj;
    Input Cd_inj;
    Input A_inj;

    Input D_c;
    Input rho_cw;
    Input Ys_cw;
    Input sf_cw;
    Input PCPP;
    Output L_c;
    Output Vempty_c;
    Output m_cw;
    Output th_cw;

    Input L_f;
    Input th0_f;

    Input Cd_nzl;
    Input eps;
    Input A_throat;
    Output L_nzl;
    Output m_nzl;


    broRunning running;


    // Optional output parameters:
    // These buffers must be supplied by the user, and each must be either null
    // or contain at-least `count` elements. If `count` is 0, all buffers are
    // assumed to be null. `upto` contains the index of the next write into the
    // buffers, aka the number of elements already written provided it was set to
    // 0 before simming.

    int upto; // input+output.
    int count; // input.
    Output* restrict out_t; // pointer=input, data=output.
    Output* restrict out_alt_r;
    Output* restrict out_vel_r;
    Output* restrict out_acc_r;
    Output* restrict out_m_r;
    Output* restrict out_com_r;
    Output* restrict out_T_t;
    Output* restrict out_T_g;
    Output* restrict out_P_t;
    Output* restrict out_P_c;
    Output* restrict out_P_a;
    Output* restrict out_m_l; // always saturated.
    Output* restrict out_m_v; // only saturated if m_l > negligible.
    Output* restrict out_m_f;
    Output* restrict out_dm_inj; // oxidiser injector mass flow rate.
    Output* restrict out_dm_reg; // fuel regression mass flow rate.
    Output* restrict out_dm_out; // exhaust mass flow rate.
    Output* restrict out_m_g;
    Output* restrict out_cp_g;
    Output* restrict out_cv_g;
    Output* restrict out_y_g;
    Output* restrict out_R_g;
    Output* restrict out_ofr;
    Output* restrict out_Isp;
    Output* restrict out_Fthrust;
    Output* restrict out_Fdrag;
    Output* restrict out_Fgravity;
} broState;


// Simulates the given state, writing all its outputs.
__declspec(dllexport) void bro_sim(broState* s);


#endif
