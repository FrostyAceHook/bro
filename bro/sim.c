// c my beloved.

#include "sim.h"

#include <stdio.h>
#include <setjmp.h>

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


#if (defined(BR_NO_ASSERT) && BR_NO_ASSERT)

#define assertion_failed() (0)
#define assertx(x, extra, ...) do { (void)(x); } while (0)
#define assert(x) do { (void)(x); } while (0)

#else

// Cheeky jumping-assert.
static jmp_buf _assert_jump;
#define assertion_failed() (setjmp(_assert_jump))
#define assertx(x, extra, ...) do {                                             \
        if (__builtin_expect_with_probability(!(x), 0, 1.0)) {                  \
            fprintf(stderr, "Assertion failed: %s, file %s, line %d, extra: "   \
                    extra "\n", #x, __FILE__, __LINE__, __VA_ARGS__);           \
            longjmp(_assert_jump, 1);                                           \
        }                                                                       \
    } while (0)
#define assert(x) do {                                                      \
        if (__builtin_expect_with_probability(!(x), 0, 1.0)) {              \
            fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", #x, \
                    __FILE__, __LINE__);                                    \
            longjmp(_assert_jump, 1);                                       \
        }                                                                   \
    } while (0)

#endif


#if (defined(BR_HAVEALOOK) && BR_HAVEALOOK)
  #define local __attribute((__used__)) static
#else
  #define local static
#endif




#define PI (3.141592653589793) // pi
#define PI_4 (0.7853981633974483) // pi/4

local f64 circ_area(f64 D) {
    return PI_4 * D*D;
}
local f64 cyl_area(f64 L, f64 D) { // only curved face.
    return L * PI * D;
}
local f64 tube_vol(f64 L, f64 ID, f64 OD) {
    return L * PI_4 * (OD*OD - ID*ID);
}


// Returns `2^exp`, requiring `exp` to be an integer.
local f64 br_with_exp(i32 exp) {
    assertx(-1022 <= exp && exp <= 1023, "exp=%d", exp);
    // Bias exponent and place it in the ieee754 double exponent bits.
    u64 bits = ((u64)(exp + 1023) << 52);
    // Interpret bits as float.
    f64 f;
    __builtin_memcpy(&f, &bits, 8);
    return f;
}

// Finds `fraction` and `exponent` s.t. `pos_norm_f = fraction * 2^exponent`.
local f64 br_fsplit(f64 pos_norm_f, i32* exp) {
    assertx(pos_norm_f == pos_norm_f && pos_norm_f >= 2.2250738585072014e-308 &&
            pos_norm_f != __builtin_inf(), "f=%f", pos_norm_f);

    // Get the exponent from the bits.
    u64 bits;
    __builtin_memcpy(&bits, &pos_norm_f, 8);
    *exp = ((bits & 0x7FF0000000000000U) >> 52) - 1023;

    // Hard-replace the exponent with 0 (2^0 = 1), scaling `f` to be in [1,2).
    // Try our best to do it in the xmm reggie.
    typedef f64 br_v2df __attribute((__vector_size__(16), __may_alias__));
    br_v2df fvec;
    fvec[0] = pos_norm_f;
    // for some reason, a `movq xmm0 xmm0` is generated here which i dont think
    // has any effect (it clears the hi 64b but like is that important?).
    u64 not_exp_mask = 0x800FFFFFFFFFFFFFU;
    f64 f_not_exp_mask;
    __builtin_memcpy(&f_not_exp_mask, &not_exp_mask, 8);
    br_v2df not_exp = { f_not_exp_mask, 0 };
    br_v2df zero_exp = { 1.0, 0 };
    fvec = __builtin_ia32_andpd(fvec, not_exp);
    fvec = __builtin_ia32_orpd(fvec, zero_exp);
    assertx(1.0 <= fvec[0] && fvec[0] < 2.0, "f=%f", fvec[0]);
    return fvec[0];
}

// Returns `2^x`, requiring `-1022 < x < 1024`.
local f64 br_exp2(f64 x) {
    assertx(-1022.0 < x && x < 1024.0, "x=%f", x);

    // Basic idea:
    //  x = i + f
    //  2^x = 2^(i + f)
    //  2^x = 2^i * 2^f

    // Split `x` into two parts, an integer and a fraction.
    i32 i = (i32)x; // integer part (truncating).
    f64 f = x - i;  // fractional part.
    // Now: `x = i + f`.

    // Now `f` is in -1..1, but if we move it to 0..1, our approximation becomes
    // ~100x more accurate for the same number of terms. so lets do that.
    i32 take = (f < 0.0);
    i -= take;
    f += take;
    // Note that `x` still ~=`i + f`.

    // Rational polynomial approximation of `2^f`, for `f` in 0..1. See:
    // https://www.desmos.com/calculator/eyj8pnvkdc
    f64 n0 = +3.67657762666000;
    f64 n1 = +1.31942423003000;
    f64 n2 = +0.19282487450200;
    f64 n3 = +0.01220740233640;
    f64 d0 = +3.67657762721000;
    f64 d1 = -1.22898523359000;
    f64 d2 = +0.16148178495000;
    f64 d3 = -0.00855711197218;
    f64 exp2f = (
        (((n3 * f + n2) * f + n1) * f + n0)
        /
        (((d3 * f + d2) * f + d1) * f + d0)
    );

    // Get `2^i` by directly placing the bits of `i` into the exponent bits.
    f64 exp2i = br_with_exp(i);

    // Now calculate the final result: `2^x = 2^i * 2^f`.
    return exp2i * exp2f;
}

// Returns `log_2(x)`, requiring `x >= 2.2250738585072014e-308`.
local f64 br_log2(f64 x) {
    // No negatives, zeros, or subnormals.
    assertx(x >= 2.2250738585072014e-308, "x=%f", x);
    assert(x != __builtin_inf()); // no infinities.

    // It's useful enough to ensure an exactly 0 return on 1.
    if (x == 1.0)
        return 0.0;

    // Basic idea:
    //  x = m * 2^e
    //  log2(x) = log2(m * 2^e)
    //  log2(x) = log2(m) + log2(2^e)
    //  log2(x) = log2(m) + e

    // hey we got a function for that first step.
    i32 e;
    f64 m = br_fsplit(x, &e);

    // Rational polynomial approximation of `log2(m)`, for `m` in 1..2. See:
    // https://www.desmos.com/calculator/he0wfmt2dt
    // Note that the %error explodes as the output goes to zero, what can you do.
    // Also the constant generation is super random and i managed to lose the
    // exact desmos i used for these contants (oops).
    f64 n0 = -4.958898909080;
    f64 n1 = -6.612108365090;
    f64 n2 = +9.333280025010;
    f64 n3 = +2.237727255500;
    f64 d0 = +1.023735966000;
    f64 d1 = +6.730341730150;
    f64 d2 = +4.867463372800;
    f64 d3 = +0.387193700112;
    f64 log2m = (
        (((n3 * m + n2) * m + n1) * m + n0)
        /
        (((d3 * m + d2) * m + d1) * m + d0)
    );

    // Easy final result: `log2(x) = log2(m) + e`
    return log2m + e;
}

// Returns `x^y`, requiring `x > 0`.
local f64 br_pow(f64 x, f64 y) {
    // Exp/log properties.
    //  x   = 2^log2(x)
    //  x^y = (2^log2(x))^y
    //      = 2^(log2(x) * y)
    assertx(x >= 0.0, "x=%f", x);
    return br_exp2(br_log2(x) * y);
}

// Returns `x^0.5`, requiring the compiler to be using an instruction set that
// has a sqrt instrinsic bc i cant be bothered to impl it myself.
local f64 br_sqrt(f64 x) {
    return __builtin_sqrt(x);
}

// Returns `|x|`.
local f64 br_fabs(f64 x) {
    return (x < 0) ? -x : x;
}



// Straight off the dome.
#define GAS_CONSTANT (8.31446261815324) // [J/mol/K]



// Nitrous oxide properties from CoolProp, approximated via rational polynomials
// by 'bro/approximator.py'.

// All nox properties are only valid for saturated liquid-vapour mixtures
// (including qualities of exactly 0 or 1), and vapour. Note the whats up
// everybody points:
//   nox triple point:     182.34 K  0.08785 MPa
//   nox critical point:   309.55 K  7.245 MPa
// Therefore we define our bounds as:
//  temperature      183 K .. 309 K
//  pressure        90 kPa .. 7.2 MPa
//  vap. density   1 kg/m3 .. 325 kg/m3
// Note that all inputs and outputs are base si.

#define NOX_IN_T(T) (183 <= (T) && (T) <= 309)
#define NOX_IN_P(P) (0.09 <= (P) && (P) <= 7.2)
#define NOX_IN_rho(rho) (1 <= (rho) && (rho) <= 325)

#define NOX_ASSERT_IN_T(T) assertx(NOX_IN_T((T)), "T=%fK", (T))
#define NOX_ASSERT_IN_P(P) assertx(NOX_IN_P((P)), "P=%fMPa", (P))
#define NOX_ASSERT_IN_Trho(T, rho) do {                                 \
        assertx(NOX_IN_T((T)), "T=%fK, rho=%fkg/m3", (T), (rho));       \
        assertx(NOX_IN_rho((rho)), "T=%fK, rho=%fkg/m3", (T), (rho));   \
    } while (0)
#define NOX_ASSERT_IN_TP(T, P) do {                         \
        assertx(NOX_IN_T((T)), "T=%fK, P=%fMPa", (T), (P)); \
        assertx(NOX_IN_P((P)), "T=%fK, P=%fMPa", (T), (P)); \
    } while (0)



#define NOX_Mw (44.013e-3) // [kg/mol]
#define NOX_R (GAS_CONSTANT / NOX_Mw) // [J/kg/K]

local f64 nox_Tsat(f64 P) {
    P *= 1e-6; // Pa -> MPa
    NOX_ASSERT_IN_P(P);
    f64 x1 = P;
    f64 x2 = P*P;
    f64 n0 = +1.053682149480346;
    f64 n1 = +5.000029230803551;
    f64 d0 = +0.0063155487831695655;
    f64 d1 = +0.021208661457396805;
    f64 d2 = +0.002480330679198678;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;
}

local f64 nox_rho_satliq(f64 T) {
    NOX_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = T*T;
    f64 n0 = +122309.80087089837;
    f64 n1 = -703.8266867155928;
    f64 d0 = +78.84320429264461;
    f64 d1 = -0.3947682772066382;
    f64 d2 = +0.0004576329029363305;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;
}

local f64 nox_rho_satvap(f64 T) {
    NOX_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = T*T;
    f64 x4 = x2*x2;
    f64 n0 = -5616563626.488913;
    f64 n1 = +75961789.44613385;
    f64 n2 = -283334.26225138275;
    f64 d0 = -102348183.29992932;
    f64 d1 = +412063.5030797808;
    f64 d4 = -0.002764818539542846;
    f64 Num = n0 + n1*x1 + n2*x2 + x4;
    f64 Den = d0 + d1*x1 + d4*x4;
    return Num / Den;
}

local f64 nox_Psat(f64 T) {
    NOX_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = T*T;
    f64 x3 = x2*T;
    f64 x5 = x3*x2;
    f64 c1 = +3496.03701234121;
    f64 c3 = -0.248418557185819;
    f64 c5 = +4.752823512296013e-06;
    return c1*x1 + c3*x3 + c5*x5;
}

local f64 nox_P(f64 T, f64 rho) {
    NOX_ASSERT_IN_Trho(T, rho);
    f64 x1 = T;
    f64 y2 = rho*rho;
    f64 x2y1 = T*T*rho;
    f64 y3 = y2*rho;
    f64 n5 = -1002.3823520461443;
    f64 n7 = +3.7137717864729005;
    f64 d1 = +0.019689134939176935;
    f64 Num = n5*y2 + n7*x2y1 + y3;
    f64 Den = d1*x1;
    return Num / Den;
}

local f64 nox_s_satliq(f64 P) {
    P *= 1e-6; // Pa -> MPa
    NOX_ASSERT_IN_P(P);
    f64 x1 = P;
    f64 x2 = P*P;
    f64 x4 = x2*x2;
    f64 x5 = x4*P;
    f64 n0 = -355.8755188448979;
    f64 n1 = +3779.467529165695;
    f64 n4 = -13.18529083615938;
    f64 d0 = +2.6822755391518727;
    f64 d1 = +6.034067534443412;
    f64 d2 = -0.7009834045855325;
    f64 Num = n0 + n1*x1 + n4*x4 + x5;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;
}

local f64 nox_s_satvap(f64 P) {
    P *= 1e-6; // Pa -> MPa
    NOX_ASSERT_IN_P(P);
    f64 x1 = P;
    f64 x2 = P*P;
    f64 x3 = x2*P;
    f64 n0 = +106.29753009318405;
    f64 n1 = +275.8845939008316;
    f64 n2 = -45.72111883807674;
    f64 d0 = +0.04935480191796069;
    f64 d1 = +0.16833373623393996;
    f64 d2 = -0.022833528970010734;
    f64 Num = n0 + n1*x1 + n2*x2 + x3;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;
}

local f64 nox_cp(f64 T, f64 P) {
    P *= 1e-6; // Pa -> MPa
    NOX_ASSERT_IN_TP(T, P);
    f64 x1 = T;
    f64 y1 = P;
    f64 x2 = T*T;
    f64 y2 = P*P;
    f64 n1 = -159.82009434320338;
    f64 n2 = -4999.721561359963;
    f64 d1 = -0.16874605625391056;
    f64 d2 = -9.78607781596831;
    f64 d3 = +0.001104532552674854;
    f64 d5 = +0.3326972587207335;
    f64 Num = n1*x1 + n2*y1 + x2;
    f64 Den = d1*x1 + d2*y1 + d3*x2 + d5*y2;
    return Num / Den;
}

local f64 nox_cv_satliq(f64 T) {
    NOX_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = T*T;
    f64 x3 = x2*T;
    f64 n0 = +99998624.02836193;
    f64 n1 = -417287.50625049777;
    f64 d0 = +67167.41562085839;
    f64 d2 = -1.6299051411280265;
    f64 d3 = +0.0030157641836329814;
    f64 Num = n0 + n1*x1 + x3;
    f64 Den = d0 + d2*x2 + d3*x3;
    return Num / Den;
}

local f64 nox_cv_satvap(f64 T) {
    NOX_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = T*T;
    f64 x3 = x2*T;
    f64 n1 = +1053181.5139038756;
    f64 n2 = -3536.041105280067;
    f64 d0 = +319981.2745750608;
    f64 d1 = -986.8229772784155;
    f64 Num = n1*x1 + n2*x2 + x3;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

local f64 nox_cv(f64 T, f64 P) {
    P *= 1e-6; // Pa -> MPa
    NOX_ASSERT_IN_TP(T, P);
    f64 x1 = T;
    f64 y1 = P;
    f64 x2 = T*T;
    f64 x1y1 = T*P;
    f64 y2 = P*P;
    f64 n2 = +404.17179888559093;
    f64 n3 = +0.0116226190297697;
    f64 n4 = -1.6302745906659892;
    f64 d0 = -0.6247719294046308;
    f64 d1 = +0.007139106757436227;
    f64 d4 = -0.0005336079270844953;
    f64 Num = n2*y1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d0 + d1*x1 + d4*x1y1;
    return Num / Den;
}

local f64 nox_h_satliq(f64 T) {
    NOX_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x3 = T*T*T;
    f64 x4 = x3*T;
    f64 n0 = -1554844073854.8723;
    f64 n1 = +10248538986.948492;
    f64 n3 = -53363.27024534557;
    f64 d0 = +7175993.87640348;
    f64 d1 = -22685.089801817525;
    f64 Num = n0 + n1*x1 + n3*x3 + x4;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

local f64 nox_h_satvap(f64 T) {
    NOX_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = T*T;
    f64 x4 = x2*x2;
    f64 n0 = +36006860815.96613;
    f64 n1 = -145833527.21804488;
    f64 d0 = +133090.43360716323;
    f64 d1 = -784.7985028166469;
    f64 d2 = +1.1478421530010148;
    f64 Num = n0 + n1*x1 + x4;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;
}

local f64 nox_h(f64 T, f64 rho) {
    NOX_ASSERT_IN_Trho(T, rho);
    f64 x1 = T;
    f64 y1 = rho;
    f64 x2 = T*T;
    f64 n1 = +72.32632139879598;
    f64 d0 = -0.051735087047641605;
    f64 d1 = +0.000958993783893223;
    f64 d2 = +0.0003057500026811124;
    f64 Num = n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*y1;
    return Num / Den;
}

local f64 nox_u_satliq(f64 T) {
    NOX_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = T*T;
    f64 x3 = x2*T;
    f64 n0 = -5712868562742.414;
    f64 n1 = +48367409598.93219;
    f64 n2 = -94710942.69677454;
    f64 d0 = +17576470.543787282;
    f64 d1 = -54575.615844086264;
    f64 Num = n0 + n1*x1 + n2*x2 + x3;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

local f64 nox_u_satvap(f64 T) {
    NOX_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = T*T;
    f64 x4 = x2*x2;
    f64 n0 = +40973028771.259094;
    f64 n1 = -161630532.82151473;
    f64 d0 = +163516.09730925123;
    f64 d1 = -930.7444896485703;
    f64 d2 = +1.3044812440250384;
    f64 Num = n0 + n1*x1 + x4;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;
}

local f64 nox_u(f64 T, f64 rho) {
    NOX_ASSERT_IN_Trho(T, rho);
    f64 x1 = T;
    f64 y1 = rho;
    f64 x2 = T*T;
    f64 n1 = +133.30350610563823;
    f64 n2 = -47.121524818758765;
    f64 d0 = -0.05688971632749395;
    f64 d1 = +0.0012334671429795158;
    f64 d2 = +0.00017800897800483042;
    f64 Num = n1*x1 + n2*y1 + x2;
    f64 Den = d0 + d1*x1 + d2*y1;
    return Num / Den;
}

local f64 nox_Z(f64 T, f64 rho) {
    NOX_ASSERT_IN_Trho(T, rho);
    f64 y1 = rho;
    f64 x1y1 = T*rho;
    f64 y2 = rho*rho;
    f64 c0 = +0.9962244155069389;
    f64 c2 = -0.010033321729222494;
    f64 c4 = +2.3783799563418678e-05;
    f64 c5 = +2.383200854104837e-06;
    return c0 + c2*y1 + c4*x1y1 + c5*y2;
}



// Paraffin+NOx CEA results from rocketcea, approximated via rational polynomials
// by 'bro/approximator.py'.

// For the bounds of the inputs, we use:
//  chamber pressure  90 kPa .. 7.2 Mpa
//  ox-fuel ratio        0.5 .. 13
// Note that all inputs and outputs are base si.

#define CEA_IN_P(P) (0.09 <= (P) && (P) <= 7.2)
#define CEA_IN_ofr(ofr) (0.5 <= (ofr) && (ofr) <= 13)

#define CEA_ASSERT_IN_P(P) assertx(CEA_IN_P((P)), "P=%fMPa", (P))
#define CEA_ASSERT_IN_ofr(ofr) assertx(CEA_IN_ofr((ofr)), "ofr=%f", (ofr))
#define CEA_ASSERT_IN_Prho(P, rho) do {                             \
        assertx(NOX_IN_P((P)), "P=%fMPa, ofr=%f", (P), (ofr));      \
        assertx(NOX_IN_rho((rho)), "P=%fMPa, ofr=%f", (P), (ofr));  \
    } while (0)


local f64 cea_T(f64 P, f64 ofr) {
    P *= 1e-6; // Pa -> MPa
    CEA_ASSERT_IN_Prho(P, ofr);
    // Different approxs for different input regions.
    if (ofr >= 4) {
        f64 x1 = P;
        f64 y1 = ofr;
        f64 x1y1 = P*ofr;
        f64 y2 = ofr*ofr;
        f64 x2y1 = x1y1*P;
        f64 x1y2 = x1y1*ofr;
        f64 n0 = -14.286699845712628;
        f64 n1 = -32.4062851919402;
        f64 n2 = +3.4742076078850297;
        f64 n4 = +18.309776531751787;
        f64 d1 = +0.008300237024458222;
        f64 d4 = +0.001385352578317833;
        f64 d5 = +0.00041962485459847375;
        f64 d7 = -4.215277067475584e-06;
        f64 d8 = +0.00023245123290958759;
        f64 Num = n0 + n1*x1 + n2*y1 + n4*x1y1 + y2;
        f64 Den = d1*x1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2;
        return Num / Den;
    }
    f64 x1 = P;
    f64 y1 = ofr;
    f64 x1y1 = P*ofr;
    f64 n0 = +4.625880807530353;
    f64 n1 = +4.240967496861169;
    f64 d0 = +0.0062645868710667785;
    f64 d1 = +0.003833484131106042;
    f64 d2 = -0.0006506284584737187;
    f64 d4 = -0.0005047873316636391;
    f64 Num = n0 + n1*x1 + y1;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1;
    return Num / Den;
}

local f64 cea_cp(f64 P, f64 ofr) {
    P *= 1e-6; // Pa -> MPa
    CEA_ASSERT_IN_Prho(P, ofr);
    // Different approxs for different input regions.
    if (ofr >= 4) {
        f64 x1 = P;
        f64 y1 = ofr;
        f64 x1y1 = P*ofr;
        f64 x3 = P*P*P;
        f64 x2y1 = x1y1*P;
        f64 x1y2 = x1y1*ofr;
        f64 y3 = ofr*ofr*ofr;
        f64 n1 = +3582.944641587491;
        f64 n4 = -907.659602547744;
        f64 n6 = +0.8638673972095523;
        f64 n7 = -3.09295057735193;
        f64 n8 = +70.73570153378617;
        f64 d0 = +0.06115388267778262;
        f64 d1 = +1.9517349532470427;
        f64 d2 = -0.012085905000346658;
        f64 d4 = -0.47145131429742476;
        f64 d8 = +0.03175824988676035;
        f64 d9 = +0.0002303751013953221;
        f64 Num = n1*x1 + n4*x1y1 + n6*x3 + n7*x2y1 + n8*x1y2 + y3;
        f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d8*x1y2 + d9*y3;
        return Num / Den;
    }
    if (P >= 1) {
        f64 x1 = P;
        f64 y1 = ofr;
        f64 x1y1 = P*ofr;
        f64 y2 = ofr*ofr;
        f64 x1y2 = x1y1*ofr;
        f64 n0 = +4.295929211019145;
        f64 n1 = +0.28907793370196055;
        f64 n2 = -3.367483987949862;
        f64 d0 = +0.0005978622331052173;
        f64 d1 = +7.182771715581137e-05;
        f64 d2 = -0.0007415570025024972;
        f64 d4 = -4.289368017656845e-05;
        f64 d5 = +0.00037331217091959564;
        f64 d8 = +1.8653015095784888e-05;
        f64 Num = n0 + n1*x1 + n2*y1 + y2;
        f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d8*x1y2;
        return Num / Den;
    }
    f64 x1 = P;
    f64 y1 = ofr;
    f64 x2 = P*P;
    f64 x1y1 = P*ofr;
    f64 y2 = ofr*ofr;
    f64 x2y1 = x1y1*P;
    f64 x1y2 = x1y1*ofr;
    f64 y3 = y2*ofr;
    f64 n0 = +2.8700843792143695;
    f64 n1 = +1.4321259855091084;
    f64 n2 = -2.951652427295875;
    f64 n3 = -0.2420415412480568;
    f64 n4 = -0.35193895657900276;
    f64 d0 = +0.0003557287035334117;
    f64 d1 = +0.00018934292736992974;
    f64 d2 = -0.0005034451842726294;
    f64 d4 = -7.726545211497027e-05;
    f64 d5 = +0.00023583511725994315;
    f64 d7 = -2.1317881064428632e-05;
    f64 d8 = +1.85195099174797e-05;
    f64 d9 = +2.336730752102996e-05;
    f64 Num = n0 + n1*x1 + n2*y1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2 + d9*y3;
    return Num / Den;
}

local f64 cea_Mw(f64 P, f64 ofr) {
    P *= 1e-6; // Pa -> MPa
    CEA_ASSERT_IN_Prho(P, ofr);
    f64 x1 = P;
    f64 y1 = ofr;
    f64 x2 = P*P;
    f64 x1y1 = P*ofr;
    f64 y2 = ofr*ofr;
    f64 n0 = +0.21024398321396112;
    f64 n1 = +0.7486891502942208;
    f64 n3 = +0.0037434970654549094;
    f64 n4 = +0.1346327815444957;
    f64 d1 = +61.5856277715907;
    f64 d2 = +68.42132631506553;
    f64 d3 = +0.03177433731792165;
    f64 d5 = +30.142213791419902;
    f64 Num = n0 + n1*x1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d1*x1 + d2*y1 + d3*x2 + d5*y2;
    return Num / Den;
}



// Paraffin properties from Hybrid Rocket Propulsion Handbook, Karp & Jens.

#define PARAFFIN_rho (924.5) // [kg/m^3]



// Paraffin+NOx regression rate constants from Hybrid Rocket Propulsion Handbook,
// Karp & Jens.

#define RR_a0 (1.55e-4) // [-]
#define RR_n (0.5) // [-]



// Other (tweakable) system constants.

#define Dt (0.001) // [s], discrete calculus over time.
#define DT (0.05) // [K], discrete calculus over temperature.
#define GOTIME_STEPS (8) // [-], number of steps to take when gotiming.
#define MAX_TIME (35) // [s], stop sim after this time.
#define NEGLIGIBLE_MASS (0.001) // [kg], assume nothing if <= this.
#define STARTUP_TIME (0.2) // [s], worst-case longest startup time.
#define CUTOFF_Pr (1.02) // [-], maximum pressure ratio with no flow.
#define KEEP_SIMMING_AFTER_COMBUSTION 0 // {0,1}, keeping simming when no comb.



// Sim stopping mechanism/contraption.
static jmp_buf _stop_jmp;
#define sim_stopped() (setjmp(_stop_jmp))
#define stop_sim() (longjmp(_stop_jmp, 1))



// Sets the initial state of the system. Must have space left in the time-dep
// buffers and must not have set any element of them.
local void initial_state(broState* s) {
    assertx(s->upto == 0, "upto=%d", s->upto);
    assertx(s->count >= 1, "count=%d", s->count);

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
    s->t[0] = 0.0;
    s->T_t[0] = T0_t;
    s->m_l[0] = m0_l;
    s->m_v[0] = m0_v;
    s->D_f[0] = D0_f;
    s->m_g[0] = m0_g;
    s->nmol_g[0] = nmol0_g;
    s->T_g[0] = T0_g;
    s->Cp_g[0] = Cp0_g;
    ++s->upto;
}



// Performs some discrete integrations of the system differential. Must have
// space left in the time-dep buffers and must have some elements set in them.
local void step_state(broState* s) {
    assertx(s->upto >= 0, "upto=%d", s->upto);
    assertx(s->count - s->upto >= 1, "count=%d, upto=%d", s->count, s->upto);

    // Current state variables.
    f64 T_t = s->T_t[s->upto - 1];
    f64 m_l = s->m_l[s->upto - 1];
    f64 m_v = s->m_v[s->upto - 1];
    f64 D_f = s->D_f[s->upto - 1];
    f64 m_g = s->m_g[s->upto - 1];
    f64 nmol_g = s->nmol_g[s->upto - 1];
    f64 T_g = s->T_g[s->upto - 1];
    f64 Cp_g = s->Cp_g[s->upto - 1];

    // Reconstruct some cc/fuel state.
    f64 V_f = tube_vol(s->L_f, D_f, s->D_c);
    f64 A_f = cyl_area(s->L_f, D_f); // inner fuel grain surface area.
    f64 V_c = s->Vempty_c - V_f;
    f64 m_f = PARAFFIN_rho * V_f;

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

        f64 P_u = nox_Psat(T_t); // tank at saturated pressure.
        f64 P_d = P_c;

        if (P_u <= CUTOFF_Pr * P_d)
            goto NO_INJECTOR_FLOW;
            // id love to see python try to handle a goto.

        // Single-phase incompressible model (with Beta = 0):
        // (assuming upstream density as the "incompressible" density)
        f64 rho_u = nox_rho_satliq(T_t);
        f64 mdot_SPI = s->Cd_inj * s->A_inj * br_sqrt(2 * rho_u * (P_u - P_d));

        // Homogenous equilibrium model:
        // (assuming only saturated liquid leaving from upstream)
        f64 s_u = nox_s_satliq(P_u);
        f64 s_d_l = nox_s_satliq(P_d);
        f64 s_d_v = nox_s_satvap(P_d);
        f64 x_d = (s_u - s_d_l) / (s_d_v - s_d_l);
        assert(0 <= x_d && x_d <= 1.0);
        f64 h_u = nox_h_satliq(T_t);
        f64 T_d = nox_Tsat(P_d); // since our prop funcs expect temp.
        f64 h_d_l = nox_h_satliq(T_d);
        f64 h_d_v = nox_h_satvap(T_d);
        f64 h_d = (1 - x_d) * h_d_l + x_d * h_d_v;
        f64 rho_d_l = nox_rho_satliq(T_d);
        f64 rho_d_v = nox_rho_satvap(T_d);
        // gotta do the quality ratios in specific-volume.
        f64 v_d_l = 1 / rho_d_l;
        f64 v_d_v = 1 / rho_d_v;
        f64 v_d = (1 - x_d) * v_d_l + x_d * v_d_v;
        f64 rho_d = 1 / v_d;
        f64 mdot_HEM = s->Cd_inj * s->A_inj * rho_d * br_sqrt(2 * (h_u - h_d));

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
        dm_inj = mdot_SPI * (1 - k_NHNE) + mdot_HEM * k_NHNE;


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
        //  dT * Cv - dT * bar * (u_l - u_v) = dm_inj * (u_l - h_l)
        //                                   + foo * (u_l - u_v)
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

        if (P_u <= CUTOFF_Pr * P_d)
            goto NO_INJECTOR_FLOW;

        // Technically gamma but use 'y' for file size reduction.
        f64 y_u = nox_cp(T_t, P_u) / nox_cv(T_t, P_u);
        // Use compressibility factor to account for non-ideal gas.
        f64 Z_u = nox_Z(T_t, rho_v);

        // Real compressible flow through an injector, with both
        // choked and unchoked possibilities:
        f64 Pr_crit = br_pow(2 / (y_u + 1), y_u / (y_u - 1));
        f64 Pr_rec = P_d / P_u;
        f64 Pterm;
        if (Pr_rec <= Pr_crit) { // choked.
            Pterm = br_pow(2 / (y_u + 1), (y_u + 1) / (y_u - 1));
        } else { // unchoked.
            Pterm = br_pow(Pr_rec, 2 / y_u) - br_pow(Pr_rec, (y_u + 1) / y_u);
            Pterm *= 2 / (y_u - 1);
        }
        dm_inj = s->Cd_inj*s->A_inj*P_u * br_sqrt(y_u/Z_u/NOX_R/T_t * Pterm);

        // Mass only leaves through injector, and no state change.
        dm_v = -dm_inj;


        // Back to the well.
        //  d/dt (U) = -dm_inj * h  [first law of thermodynamics, adiabatic]
        // using no suffix is the non-saturated vapour in the tank:
        //  d/dt (U_w + U) = -dm_inj * h
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
        f64 rr_rdot = RR_a0 * br_pow(Gox, RR_n);

        // Fuel mass and diameter change from rdot:
        dD_f = 2 * rr_rdot;
        dV_f = A_f * rr_rdot;
        dm_reg = PARAFFIN_rho * dV_f;

    // No fuel.
    } else {
      #if !KEEP_SIMMING_AFTER_COMBUSTION
        stop_sim();
      #endif

        dD_f = 0.0;
        dV_f = 0.0;
        dm_reg = 0.0;
    }


    // Do nozzle flow.
    f64 dm_out;
    if (P_c > CUTOFF_Pr * s->P_a) {
        // Model the nozzle as an injector, using ideal compressible
        // flow and both choked and unchoked possibilities:
        f64 Pr_crit = br_pow(2 / (y_g + 1), y_g / (y_g - 1));
        f64 Pr_rec = s->P_a / P_c;
        f64 Pterm;
        if (Pr_rec <= Pr_crit) { // choked.
            Pterm = br_pow(2 / (y_g + 1), (y_g + 1) / (y_g - 1));
        } else { // unchoked.
            Pterm = br_pow(Pr_rec, 2 / y_g) - br_pow(Pr_rec, (y_g + 1) / y_g);
            Pterm *= 2 / (y_g - 1);
        }
        dm_out = s->Cd_nzl*s->A_nzl*P_c * br_sqrt(y_g / R_g / T_g * Pterm);
    } else {
        dm_out = 0.0;
    }


    // Gases in the chamber is just entering - exiting.
    f64 dm_g = dm_inj + dm_reg - dm_out;


    // Change in cc gas properties due to added gas.
    f64 T_n;
    f64 Mw_n;
    f64 cp_n;
    f64 dm_n = dm_inj + dm_reg; // new gases is just ox+fuel.

    // Combustion occurs if there is both oxidiser and fuel.
    if (dm_inj != 0 && dm_reg != 0) {
        // Instantaneous oxidiser-fuel ratio.
        f64 ofr = dm_inj / dm_reg;

        // If ofr too low, our cea approxes dont work and there would be very
        // little comb anyway, so assume none.
        if (ofr < 0.5)
            // Note that we should account for the properties of the vapourised
            // fuel also, but i cannor be bothered and its essentially irrelevant
            // since no combustion is happening the results arent very impactful.
            goto NO_COMBUSTION;

        // Do cea to find combustion properties.
        T_n = cea_T(P_c, ofr);
        Mw_n = cea_Mw(P_c, ofr);
        cp_n = cea_cp(P_c, ofr);
    // Otherwise non-combusting oxidiser.
    } else if (dm_inj != 0) {
      NO_COMBUSTION:;

        // No combustion but chamber gas changes due to oxidiser. Note this is
        // assuming isothermal mass transfer, so using tank temperature but with
        // current chamber pressure.
        T_n = T_t;
        Mw_n = NOX_Mw;
        cp_n = nox_cp(T_t, P_c);

    } else {
        // dm_n is 0 so any prop works.
        T_n = 1.0;
        Mw_n = 1.0;
        cp_n = 1.0;
    }

    // Change in any mass-specific property for a reservoir with flow in and out:
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



    // Now we have all state time-derivatives, we need to integrate. Note that
    // different phases of the burn has very different stability. Generally,
    // startup, shutdown, and liquid->vapour emptying are unstable and require
    // fine precision, otherwise we can safely take larger steps.

    // Also while I generally make a point not to use explicit euler (its just
    // not it), this system performs poorly under other methods since it is not
    // smooth. So, explicit euler it is. Also, we need the speed soooo.

    #define STABLE_FLOW(m, dm) (                                        \
            /* Nothing left or... */                                    \
            ((m) <= NEGLIGIBLE_MASS && (dm) == 0.0) ||                  \
            /* draining <5% each step and not close to cutoff. */       \
            ((m) > 10*NEGLIGIBLE_MASS && br_fabs(20*Dt*(dm)) <= (m))    \
        )
    #define STABLE_FLOW_NEVER_NEGLIGIBLE(m, dm) (   \
            /* just draining <5% each step. */      \
            (br_fabs(20*Dt*(dm)) <= (m))            \
        )
    i32 gotime = 1;
    gotime &= (s->upto * Dt > STARTUP_TIME); // dont risk startup being gotimed.
    gotime &= STABLE_FLOW(m_l, dm_l);
    gotime &= STABLE_FLOW(m_v, dm_v);
    gotime &= STABLE_FLOW(m_f, -dm_reg);
    gotime &= STABLE_FLOW_NEVER_NEGLIGIBLE(m_g, dm_g);

    i32 steps = 1;
    if (gotime) // GOTIME.
        steps = GOTIME_STEPS;

    // Assume constant first derivative over all steps and perform forward euler
    // integration.
    s->t[s->upto]      = s->t[s->upto - 1]      + steps * Dt;
    s->T_t[s->upto]    = s->T_t[s->upto - 1]    + steps * Dt * dT_t;
    s->m_l[s->upto]    = s->m_l[s->upto - 1]    + steps * Dt * dm_l;
    s->m_v[s->upto]    = s->m_v[s->upto - 1]    + steps * Dt * dm_v;
    s->D_f[s->upto]    = s->D_f[s->upto - 1]    + steps * Dt * dD_f;
    s->m_g[s->upto]    = s->m_g[s->upto - 1]    + steps * Dt * dm_g;
    s->nmol_g[s->upto] = s->nmol_g[s->upto - 1] + steps * Dt * dnmol_g;
    s->T_g[s->upto]    = s->T_g[s->upto - 1]    + steps * Dt * dT_g;
    s->Cp_g[s->upto]   = s->Cp_g[s->upto - 1]   + steps * Dt * dCp_g;
    ++s->upto;
}



// DLL-exposed function.
void bro_sim(broState* s) {
    // Setup the assert macro to abort back to here, and just get outta here with
    // whatever data we gened.
    if (assertion_failed())
        return;

    // Setup sim stop dropoff.
    if (sim_stopped())
        return;

    // Do the initial step.
    if (s->count < 1) // onto nothing.
        return;
    initial_state(s);

    // Integrate the system differential.
    while (s->upto < s->count) // dont buffer overflow.
        step_state(s);
}
