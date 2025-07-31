#ifndef SIM_H_
#define SIM_H_


typedef struct broState {
    // Tracked state (all of which changes over time and are described by
    // differentials) is:
    double* T_t;       // tank temperature
    double* m_l;       // tank liquid mass (happens to always be saturated)
    double* m_v;       // tank vapour mass (only saturated if m_l > negligible)
    double* D_f;       // fuel grain inner diameter
    double* m_g;       // cc gas mass
    double* nmol_g;    // cc gas number of moles
    double* T_g;       // cc gas temperature
    double* Cp_g;      // cc gas constant pressure heat capacity
    // This is enough to fully define the system at all times (when combined with
    // other constant parameters).

    int max_count;  // array length of ^ those time dep vars.

    double V_t;        // tank volume.

    double C_w;        // tank wall heat capacity (note Cv ~= Cp for solids).

    double vff0_o;     // ox initial volumetric fill fraction.

    double Cd_inj;     // injector discharge coeff.
    double A_inj;      // injector orifice area.

    double L_f;        // fuel length.
    double D0_f;       // initial fuel inner diameter.

    double D_c;        // cc diameter.
    double eta_c;      // cc combustion efficiency.
    double Vempty_c;   // empty (no fuel) cc volume.

    double Cd_nzl;     // nozzle discharge coeff.
    double A_nzl;      // nozzle throat area.
    double eps_nzl;    // nozzle exit area to throat area ratio.

    double T_a;        // ambient temperature.
    double P_a;        // ambient pressure.
    double rho_a;      // ambient density.
    double Mw_a;       // ambient molar mass.
    double cp_a;       // ambient constant pressure specific heat capacity.

    // NOTE: any changes to this must be reflected in the cython wrapping class.
} broState;


// Simulates the burn and returns how many elements of the state arrays were
// used. Expected all members of `s` to be set (but the array data will be
// overwritten).
__declspec(dllexport) int bro_sim(broState* s);


#endif
