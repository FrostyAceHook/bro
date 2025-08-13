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

// i literally just cant be bother to type it out.
#define rstr restrict

// Expands to the size of the given starray.
#define countof(...) ((i32)(sizeof((__VA_ARGS__)) / sizeof(*(__VA_ARGS__))))



#if (defined(BR_NO_ASSERTS) && BR_NO_ASSERTS)

#define assertion_failed() (0)
#define assertx(x, extra, ...) do {         \
        if (!(x)) __builtin_unreachable();  \
    } while (0)
#define assert(x) do {                      \
        if (!(x)) __builtin_unreachable();  \
    } while (0)

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



#if (defined(BR_HAVEALOOK) && BR_HAVEALOOK) && \
    (!defined(BR_DONTLOOK) || !BR_DONTLOOK)
  #define local __attribute((__used__)) static
#else
  #define local static
#endif



// Alias for nan. When checking if a variable is "unset", use `!ISSET(...)`.
#define UNSET (__builtin_nan(""))
#define ISSET(x) ((x) == (x))


#define PI (3.141592653589793) // pi
#define PI_2 (1.5707963267948966) // pi/2
#define PI_4 (0.7853981633974483) // pi/4
#define LN2 (0.6931471805599453) // ln(2)


// Expands to the minimum of the two given numbers.
#define br_min(x, y) ((y) < (x) ? (y) : (x))
// Expands to the maximum of the two given numbers.
#define br_max(x, y) ((y) > (x) ? (y) : (x))
// Expands to non-zero if `x` if in [`lo`, `hi`].
#define br_within(x, lo, hi) ((lo) <= (x) && (x) <= (hi))
// Expands to `x`, clamped to the nearest bound of [`lo`, `hi`] if out of bounds.
#define br_clamp(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))


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

// Finds `fraction` and `exponent` s.t. `pos_norm_f = fraction * 2^exponent`,
// where `fraction` is in [1, 2).
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
    assertx(x > 0.0, "x=%f", x);
    return br_exp2(br_log2(x) * y);
}

// Returns `x^0.5`, requiring the compiler to be using an instruction set that
// has a sqrt instrinsic bc i cant be bothered to impl it myself.
local f64 br_sqrt(f64 x) {
    // Gaslight the compiler into not caring about the signbit. I think it
    // refuses to unconditionally send the intrinsic bc it respects signed zeros
    // whereas we compile without. But like, i dont care, i just want the
    // instruction. Anyway adding the unreachable-if-signedbit causes gcc to only
    // emit the instruction without a check-if-zero-then-send-to-stdlib-impl so
    // solved.
    assertx(!__builtin_signbit(x), "x=%f", x);
    if (__builtin_signbit(x))
        __builtin_unreachable();
    return __builtin_sqrt(x);
}



// Straight off the dome.
#define GAS_CONSTANT (8.31446261815324) // [J/mol/K]
#define STANDARD_GRAVITY (9.80665) // [m/s^2]



// N2O properties from CoolProp, approximated via rational polynomials by
// 'bro/approximator.py'.

// All N2O properties are only valid for saturated liquid-vapour mixtures
// (including qualities of exactly 0 or 1), and vapour. Note the whats up
// everybody points:
//   N2O triple point:     182.34 K  0.08785 MPa
//   N2O critical point:   309.55 K  7.245 MPa
// Therefore we define our bounds as:
//  temperature      183 K .. 309 K
//  pressure        90 kPa .. 7.2 MPa
//  vap. density   1 kg/m3 .. 325 kg/m3
// Note that all inputs and outputs are base si.

#define N2O_Tmin (183.0) // [K]
#define N2O_Tmax (309.0) // [K]
#define N2O_Pmin (90e3) // [Pa]
#define N2O_Pmax (7.2e6) // [Pa]
#define N2O_rhomin (1.0) // [kg/m^3]
#define N2O_rhomax (325.0) // [kg/m^3]

#define N2O_IN_T(T) (br_within((T), N2O_Tmin, N2O_Tmax))
#define N2O_IN_P(P) (br_within((P), N2O_Pmin, N2O_Pmax))
#define N2O_IN_rho(rho) (br_within((rho), N2O_rhomin, N2O_rhomax))

#define N2O_ASSERT_IN_T(T) do {                 \
        assertx(N2O_IN_T((T)), "T=%fK", (T));   \
    } while (0)
#define N2O_ASSERT_IN_P(P) do {                 \
        assertx(N2O_IN_P((P)), "P=%fPa", (P));  \
    } while (0)
#define N2O_ASSERT_IN_Trho(T, rho) do {                                 \
        assertx(N2O_IN_T((T)), "T=%fK, rho=%fkg/m3", (T), (rho));       \
        assertx(N2O_IN_rho((rho)), "T=%fK, rho=%fkg/m3", (T), (rho));   \
    } while (0)
#define N2O_ASSERT_IN_TP(T, P) do {                         \
        assertx(N2O_IN_T((T)), "T=%fK, P=%fPa", (T), (P));  \
        assertx(N2O_IN_P((P)), "T=%fK, P=%fPa", (T), (P));  \
    } while (0)


#define N2O_Mw (44.013e-3) // [kg/mol]
#define N2O_R (GAS_CONSTANT / N2O_Mw) // [J/kg/K]
#define N2O_Ttrip (182.34) // [K]
#define N2O_Ptrip (87.85e3) // [Pa]
#define N2O_Tcrit (309.55) // [K]
#define N2O_Pcrit (7.245e6) // [Pa]

local f64 N2O_Tsat(f64 P) {
    N2O_ASSERT_IN_P(P);
    f64 x1 = P * 1e-6;
    f64 x2 = x1*x1;
    f64 n0 = +1.053682149480346;
    f64 n1 = +5.000029230803551;
    f64 d0 = +0.0063155487831695655;
    f64 d1 = +0.021208661457396805;
    f64 d2 = +0.002480330679198678;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;
}

local f64 N2O_rho_satliq(f64 T) {
    N2O_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = x1*x1;
    f64 n0 = +122309.80087089837;
    f64 n1 = -703.8266867155928;
    f64 d0 = +78.84320429264461;
    f64 d1 = -0.3947682772066382;
    f64 d2 = +0.0004576329029363305;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;
}

local f64 N2O_rho_satvap(f64 T) {
    N2O_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = x1*x1;
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

local f64 N2O_Psat(f64 T) {
    N2O_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = x1*x1;
    f64 x3 = x2*x1;
    f64 x5 = x3*x2;
    f64 c1 = +3496.03701234121;
    f64 c3 = -0.248418557185819;
    f64 c5 = +4.752823512296013e-06;
    return c1*x1 + c3*x3 + c5*x5;
}

local f64 N2O_P(f64 T, f64 rho) {
    N2O_ASSERT_IN_Trho(T, rho);
    f64 x1 = T;
    f64 y1 = rho;
    f64 y2 = y1*y1;
    f64 x2y1 = x1*x1*y1;
    f64 y3 = y2*y1;
    f64 n5 = -1002.3823520461443;
    f64 n7 = +3.7137717864729005;
    f64 d1 = +0.019689134939176935;
    f64 Num = n5*y2 + n7*x2y1 + y3;
    f64 Den = d1*x1;
    return Num / Den;
}

local f64 N2O_s_satliq(f64 P) {
    N2O_ASSERT_IN_P(P);
    f64 x1 = P * 1e-6;
    f64 x2 = x1*x1;
    f64 x4 = x2*x2;
    f64 x5 = x4*x1;
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

local f64 N2O_s_satvap(f64 P) {
    N2O_ASSERT_IN_P(P);
    f64 x1 = P * 1e-6;
    f64 x2 = x1*x1;
    f64 x3 = x2*x1;
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

local f64 N2O_cp(f64 T, f64 P) {
    N2O_ASSERT_IN_TP(T, P);
    f64 x1 = T;
    f64 y1 = P * 1e-6;
    f64 x2 = x1*x1;
    f64 y2 = y1*y1;
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

local f64 N2O_cv_satliq(f64 T) {
    N2O_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = x1*x1;
    f64 x3 = x2*x1;
    f64 n0 = +99998624.02836193;
    f64 n1 = -417287.50625049777;
    f64 d0 = +67167.41562085839;
    f64 d2 = -1.6299051411280265;
    f64 d3 = +0.0030157641836329814;
    f64 Num = n0 + n1*x1 + x3;
    f64 Den = d0 + d2*x2 + d3*x3;
    return Num / Den;
}

local f64 N2O_cv_satvap(f64 T) {
    N2O_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = x1*x1;
    f64 x3 = x2*x1;
    f64 n1 = +1053181.5139038756;
    f64 n2 = -3536.041105280067;
    f64 d0 = +319981.2745750608;
    f64 d1 = -986.8229772784155;
    f64 Num = n1*x1 + n2*x2 + x3;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

local f64 N2O_cv(f64 T, f64 P) {
    N2O_ASSERT_IN_TP(T, P);
    f64 x1 = T;
    f64 y1 = P * 1e-6;
    f64 x2 = x1*x1;
    f64 x1y1 = x1*y1;
    f64 y2 = y1*y1;
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

local f64 N2O_h_satliq(f64 T) {
    N2O_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x3 = x1*x1*x1;
    f64 x4 = x3*x1;
    f64 n0 = -1554844073854.8723;
    f64 n1 = +10248538986.948492;
    f64 n3 = -53363.27024534557;
    f64 d0 = +7175993.87640348;
    f64 d1 = -22685.089801817525;
    f64 Num = n0 + n1*x1 + n3*x3 + x4;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

local f64 N2O_h_satvap(f64 T) {
    N2O_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = x1*x1;
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

local f64 N2O_h(f64 T, f64 rho) {
    N2O_ASSERT_IN_Trho(T, rho);
    f64 x1 = T;
    f64 y1 = rho;
    f64 x2 = x1*x1;
    f64 n1 = +72.32632139879598;
    f64 d0 = -0.051735087047641605;
    f64 d1 = +0.000958993783893223;
    f64 d2 = +0.0003057500026811124;
    f64 Num = n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*y1;
    return Num / Den;
}

local f64 N2O_u_satliq(f64 T) {
    N2O_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = x1*x1;
    f64 x3 = x2*x1;
    f64 n0 = -5712868562742.414;
    f64 n1 = +48367409598.93219;
    f64 n2 = -94710942.69677454;
    f64 d0 = +17576470.543787282;
    f64 d1 = -54575.615844086264;
    f64 Num = n0 + n1*x1 + n2*x2 + x3;
    f64 Den = d0 + d1*x1;
    return Num / Den;
}

local f64 N2O_u_satvap(f64 T) {
    N2O_ASSERT_IN_T(T);
    f64 x1 = T;
    f64 x2 = x1*x1;
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

local f64 N2O_u(f64 T, f64 rho) {
    N2O_ASSERT_IN_Trho(T, rho);
    f64 x1 = T;
    f64 y1 = rho;
    f64 x2 = x1*x1;
    f64 n1 = +133.30350610563823;
    f64 n2 = -47.121524818758765;
    f64 d0 = -0.05688971632749395;
    f64 d1 = +0.0012334671429795158;
    f64 d2 = +0.00017800897800483042;
    f64 Num = n1*x1 + n2*y1 + x2;
    f64 Den = d0 + d1*x1 + d2*y1;
    return Num / Den;
}

local f64 N2O_Z(f64 T, f64 rho) {
    N2O_ASSERT_IN_Trho(T, rho);
    f64 x1 = T;
    f64 y1 = rho;
    f64 x1y1 = x1*y1;
    f64 y2 = y1*y1;
    f64 c0 = +0.9962244155069389;
    f64 c2 = -0.010033321729222494;
    f64 c4 = +2.3783799563418678e-05;
    f64 c5 = +2.383200854104837e-06;
    return c0 + c2*y1 + c4*x1y1 + c5*y2;
}



// Paraffin properties from Hybrid Rocket Propulsion Handbook, Karp & Jens.

#define PARAFFIN_rho (924.5) // [kg/m^3]



// Paraffin+N2O regression rate constants from Hybrid Rocket Propulsion Handbook,
// Karp & Jens.

local f64 regression_rate_rdot(f64 G_o) {
    return 1.55e-4 * br_sqrt(G_o);
}



// Paraffin+N2O CEA results from NASA CEA2, approximated via rational
// polynomials and lookup-tables-with-biasing(tm) by 'bro/approximator.py'.

// For the bounds of the inputs, we use:
//  chamber pressure     90 kPa .. 7.2 Mpa
//  ox-fuel ratio           0.5 .. 13
//  exit-throat area ratio  1.2 .. 12
// Note that all inputs and outputs are base si.

#define CEA_Pmin (90e3) // [Pa]
#define CEA_Pmax (7.2e6) // [Pa]
#define CEA_ofrmin (0.5) // [-]
#define CEA_ofrmax (13.0) // [-]
#define CEA_epsmin (1.2) // [-]
#define CEA_epsmax (12.0) // [-]

#define CEA_IN_P(P) (br_within((P), CEA_Pmin, CEA_Pmax))
#define CEA_IN_ofr(ofr) (br_within((ofr), CEA_ofrmin, CEA_ofrmax))
#define CEA_IN_eps(eps) (br_within((eps), CEA_epsmin, CEA_epsmax))

#define CEA_ASSERT_IN_P(P) do {                 \
        assertx(CEA_IN_P((P)), "P=%fPa", (P));  \
    } while (0)
#define CEA_ASSERT_IN_ofr(ofr) do {                     \
        assertx(CEA_IN_ofr((ofr)), "ofr=%f", (ofr));    \
    } while (0)
#define CEA_ASSERT_IN_Pofr(P, ofr) do {                             \
        assertx(CEA_IN_P((P)), "P=%fPa, ofr=%f", (P), (ofr));       \
        assertx(CEA_IN_ofr((ofr)), "P=%fPa, ofr=%f", (P), (ofr));   \
    } while (0)
#define CEA_ASSERT_IN_Pofreps(P, ofr, eps) do {                                 \
        assertx(CEA_IN_P((P)), "P=%fPa, ofr=%f, eps=%f", (P), (ofr), (eps));    \
        assertx(CEA_IN_ofr((ofr)), "P=%fPa, ofr=%f, eps=%f", (P), (ofr), (eps));\
        assertx(CEA_IN_eps((eps)), "P=%fPa, ofr=%f, eps=%f", (P), (ofr), (eps));\
    } while (0)


local f64 cea_T_c(f64 P, f64 ofr) {
    CEA_ASSERT_IN_Pofr(P, ofr);
    // Different approxs for different input regions.
    if (ofr >= 4) {
        f64 x1 = P * 1e-6;
        f64 y1 = ofr;
        f64 x1y1 = x1*y1;
        f64 y2 = y1*y1;
        f64 x2y1 = x1y1*x1;
        f64 x1y2 = x1y1*y1;
        f64 n0 = -14.288432205175953;
        f64 n1 = -32.42867599262846;
        f64 n2 = +3.474751778815847;
        f64 n4 = +18.31561738023227;
        f64 d1 = +0.008295564068249973;
        f64 d4 = +0.0013866109147024212;
        f64 d5 = +0.0004196407872643308;
        f64 d7 = -4.217470292937784e-06;
        f64 d8 = +0.00023248363130312795;
        f64 Num = n0 + n1*x1 + n2*y1 + n4*x1y1 + y2;
        f64 Den = d1*x1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2;
        return Num / Den;
    }
    f64 x1 = P * 1e-6;
    f64 y1 = ofr;
    f64 x1y1 = x1*y1;
    f64 n0 = +4.626249969485239;
    f64 n1 = +4.2414906007771505;
    f64 d0 = +0.006265099088905423;
    f64 d1 = +0.0038339663921447076;
    f64 d2 = -0.0006507452327885072;
    f64 d4 = -0.0005048607265830429;
    f64 Num = n0 + n1*x1 + y1;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1;
    return Num / Den;
}

local f64 cea_cp_c(f64 P, f64 ofr) {
    CEA_ASSERT_IN_Pofr(P, ofr);
    // Different approxs for different input regions.
    if (ofr >= 4) {
        f64 x1 = P * 1e-6;
        f64 y1 = ofr;
        f64 x1y1 = x1*y1;
        f64 x3 = x1*x1*x1;
        f64 x2y1 = x1y1*x1;
        f64 x1y2 = x1y1*y1;
        f64 y3 = y1*y1*y1;
        f64 n1 = +3697.2499992698126;
        f64 n4 = -936.4542475610451;
        f64 n6 = +0.9270759227273586;
        f64 n7 = -3.2824223104827146;
        f64 n8 = +73.08080264333545;
        f64 d0 = +0.06117125463913653;
        f64 d1 = +2.017096527865246;
        f64 d2 = -0.012091433831058363;
        f64 d4 = -0.4872595016883579;
        f64 d8 = +0.03283582431629867;
        f64 d9 = +0.00022931797763498784;
        f64 Num = n1*x1 + n4*x1y1 + n6*x3 + n7*x2y1 + n8*x1y2 + y3;
        f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d8*x1y2 + d9*y3;
        return Num / Den;
    }
    if (P >= 1e6) {
        f64 x1 = P * 1e-6;
        f64 y1 = ofr;
        f64 x1y1 = x1*y1;
        f64 y2 = y1*y1;
        f64 x1y2 = x1y1*y1;
        f64 n0 = +4.2606478633549125;
        f64 n1 = +0.286474125290797;
        f64 n2 = -3.346916554110677;
        f64 d0 = +0.0005978252239739312;
        f64 d1 = +7.15329555395512e-05;
        f64 d2 = -0.0007466904302128527;
        f64 d4 = -4.2802043949574985e-05;
        f64 d5 = +0.0003764818378455317;
        f64 d8 = +1.8535400055226076e-05;
        f64 Num = n0 + n1*x1 + n2*y1 + y2;
        f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d8*x1y2;
        return Num / Den;
    }
    f64 x1 = P * 1e-6;
    f64 y1 = ofr;
    f64 x2 = x1*x1;
    f64 x1y1 = x1*y1;
    f64 y2 = y1*y1;
    f64 x2y1 = x1y1*x1;
    f64 x1y2 = x1y1*y1;
    f64 y3 = y2*y1;
    f64 n0 = +2.8961711401281796;
    f64 n1 = +1.380267785409765;
    f64 n2 = -2.959235197532152;
    f64 n3 = -0.22742870022761344;
    f64 n4 = -0.33389437020718044;
    f64 d0 = +0.00035840685034168327;
    f64 d1 = +0.0001893546685822136;
    f64 d2 = -0.0005060456120694603;
    f64 d4 = -8.374514776956192e-05;
    f64 d5 = +0.00023872791254781774;
    f64 d7 = -1.948261122047713e-05;
    f64 d8 = +2.07406108782002e-05;
    f64 d9 = +2.276707960740409e-05;
    f64 Num = n0 + n1*x1 + n2*y1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d0 + d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d7*x2y1 + d8*x1y2 + d9*y3;
    return Num / Den;
}

local f64 cea_y_throat(f64 P, f64 ofr) {
    CEA_ASSERT_IN_Pofr(P, ofr);
    if (ofr >= 3.07) {
        f64 x1 = P * 1e-6;
        f64 y1 = ofr;
        f64 x1y1 = x1*y1;
        f64 y2 = y1*y1;
        f64 x1y2 = x1y1*y1;
        f64 n0 = -2.9309092389810942;
        f64 n1 = +10.897607957141895;
        f64 n2 = -6.046792817383578;
        f64 n4 = -0.8706749287437507;
        f64 d1 = +7.685381299036463;
        f64 d2 = -5.7290230856550375;
        f64 d4 = -0.40591530228463435;
        f64 d5 = +0.8912798496664587;
        f64 d8 = -0.017711953717497203;
        f64 Num = n0 + n1*x1 + n2*y1 + n4*x1y1 + y2;
        f64 Den = d1*x1 + d2*y1 + d4*x1y1 + d5*y2 + d8*x1y2;
        return Num / Den;
    }
    f64 x1 = P * 1e-6;
    f64 y1 = ofr;
    f64 x2 = x1*x1;
    f64 x1y1 = x1*y1;
    f64 y2 = y1*y1;
    f64 x3 = x2*x1;
    f64 x2y1 = x1y1*x1;
    f64 x2y2 = x1y1*x1y1;
    f64 n0 = -0.11263466290978147;
    f64 n1 = +0.10384675996676593;
    f64 n2 = +0.02920480705193921;
    f64 n3 = +0.6036341368237339;
    f64 n4 = -1.4356038387210333;
    f64 d3 = +0.4996068573643659;
    f64 d4 = -1.069986102678251;
    f64 d5 = +0.7620770586156876;
    f64 d6 = +0.00039374249322294956;
    f64 d7 = -0.019399944085301078;
    f64 d12 = +0.0019976937469849713;
    f64 Num = n0 + n1*x1 + n2*y1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d3*x2 + d4*x1y1 + d5*y2 + d6*x3 + d7*x2y1 + d12*x2y2;
    return Num / Den;
}

local f64 cea_Mw_c(f64 P, f64 ofr) {
    CEA_ASSERT_IN_Pofr(P, ofr);
    f64 x1 = P * 1e-6;
    f64 y1 = ofr;
    f64 x2 = x1*x1;
    f64 x1y1 = x1*y1;
    f64 y2 = y1*y1;
    f64 n0 = +0.21182839008473092;
    f64 n1 = +0.7393450040280863;
    f64 n3 = +0.004477743899937738;
    f64 n4 = +0.13398511532142252;
    f64 d1 = +60.84686567413254;
    f64 d2 = +68.52163215322909;
    f64 d3 = +0.08672790956340412;
    f64 d5 = +30.13870865656756;
    f64 Num = n0 + n1*x1 + n3*x2 + n4*x1y1 + y2;
    f64 Den = d1*x1 + d2*y1 + d3*x2 + d5*y2;
    return Num / Den;
}

static f32 cea_Ivac_table[10][2][2] =
    { { { 126.1773681640625,  174.53619384765625 },
        { 131.6615753173828,  182.00814819335938 } },

      { { 157.9001007080078,  213.9551544189453  },
        { 164.42405700683594, 223.80224609375    } },

      { { 182.354736328125,   235.49440002441406 },
        { 182.57899475097656, 240.7645263671875  } },

      { { 201.67176818847656, 262.7115173339844  },
        { 202.70132446289062, 263.05810546875    } },

      { { 206.6666717529297,  275.6676940917969  },
        { 211.09072875976562, 277.2680969238281  } },

      { { 205.1580047607422,  280.71356201171875 },
        { 213.28236389160156, 284.69927978515625 } },

      { { 202.00814819335938, 280.4179382324219  },
        { 211.08053588867188, 287.3496398925781  } },

      { { 198.87869262695312, 275.9123229980469  },
        { 207.4923553466797,  282.7420959472656  } },

      { { 196.01426696777344, 270.4587097167969  },
        { 203.761474609375,   275.45361328125    } },

      { { 193.40469360351562, 265.0764465332031  },
        { 200.18348693847656, 268.7767639160156  } } };
    // Possibly the hardest working 40 element lookup table ever constructed.
local f64 cea_Ivac(f64 P, f64 ofr, f64 eps) {
    CEA_ASSERT_IN_Pofreps(P, ofr, eps);

    // P in 0.09..7.2, 2 points.
    // ofr in 0.5..13, 10 points.
    // eps in 1.2..12, 2 points.
    // table in ofr,P,eps order (to allow loads which are always 32 contiguous
    // bytes).

    // ofr needs typical lookup.
    i32 i = (i32)((ofr - CEA_ofrmin) / (CEA_ofrmax - CEA_ofrmin) *
                  countof(cea_Ivac_table));
    i = br_clamp(i, 0, countof(cea_Ivac_table) - 1); // dont index oob.
    f64 ii = ofr - i;

    // P and eps are biased before being looked up using some fuckass formula
    // (https://www.desmos.com/calculator/okiztovx6y). Also note only two
    // elements along P and eps, so no need to find table index.
    f64 jj = -1.4 / (P * 1e-6 + 1.1081069) + 1.1685101;
    f64 kk = -3.3 / (eps + 1.4498447) + 1.245356;

    f64 v000 = cea_Ivac_table[i][0][0];
    f64 v001 = cea_Ivac_table[i][0][1];
    f64 v010 = cea_Ivac_table[i][1][0];
    f64 v011 = cea_Ivac_table[i][1][1];
    f64 v100 = cea_Ivac_table[i + 1][0][0];
    f64 v101 = cea_Ivac_table[i + 1][0][1];
    f64 v110 = cea_Ivac_table[i + 1][1][0];
    f64 v111 = cea_Ivac_table[i + 1][1][1];
    return v000 * (1.0 - ii) * (1.0 - jj) * (1.0 - kk)
         + v001 * (1.0 - ii) * (1.0 - jj) *        kk
         + v010 * (1.0 - ii) *        jj  * (1.0 - kk)
         + v011 * (1.0 - ii) *        jj  *        kk
         + v100 *        ii  * (1.0 - jj) * (1.0 - kk)
         + v101 *        ii  * (1.0 - jj) *        kk
         + v110 *        ii  *        jj  * (1.0 - kk)
         + v111 *        ii  *        jj  *        kk;
}



// Ambient properties from the International Standard Atmosphere, approximated
// via rational polynomials by 'bro/approximator.py'.

// These functions accept any altitude, but default to assuming no atmosphere
// above 60km and a return as-if at 0km when given below 0km. Note that all
// inputs and outputs are base si.


#define AMB_STANDARD_CONDITIONS_cp (1005.0) // [J/kg/K]
#define AMB_Mw (28.9647e-3) // [kg/mol]
#define AMB_R (GAS_CONSTANT / AMB_Mw) // [J/kg/K]

local f64 amb_P(f64 alt) {
    // Just say properties below sea level are constant.
    alt = br_max(alt, 0.0);
    // Above 60k, its about 0 pressure (21.5 Pa at 60k, 0.021% of sea level).
    if (alt > 60e3)
        return 0.0;
    // Different approxs for different regions.
    if (alt > 20.063e3) {
        f64 x1 = alt;
        f64 x2 = x1*x1;
        f64 n0 = +2839481483.3687034;
        f64 n1 = -103143.5518901285;
        f64 d0 = +239165.4892657625;
        f64 d1 = -24.24291511989479;
        f64 d2 = +0.00114675186987331;
        f64 Num = n0 + n1*x1 + x2;
        f64 Den = d0 + d1*x1 + d2*x2;
        return Num / Den;
    }
    f64 x1 = alt + 10e3; // need some offset to avoid /0.
    f64 x2 = x1*x1;
    f64 n0 = +1326889135.6347075;
    f64 n1 = -70527.4205624988;
    f64 d0 = +5522.836249663364;
    f64 d2 = +1.6121842727811546e-05;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d2*x2;
    return Num / Den;
}

local f64 amb_rho(f64 alt) {
    // Just say properties below sea level are constant.
    alt = br_max(alt, 0.0);
    // Above 60k, its about 0 density (0.303 g/m^3 at 60k, 0.025% of sea level).
    if (alt > 60e3)
        return 0.0;
    // Different approxs for different regions.
    if (alt > 20.063e3) {
        f64 x1 = alt;
        f64 x2 = x1*x1;
        f64 n0 = +2239506917.973228;
        f64 n1 = -92441.9508707563;
        f64 d0 = -65681468.84943648;
        f64 d2 = +22.324464921748664;
        f64 Num = n0 + n1*x1 + x2;
        f64 Den = d0 + d2*x2;
        return Num / Den;
    }
    f64 x1 = alt + 10e3; // offset to avoid /0.
    f64 x2 = x1*x1;
    f64 n0 = +1080314808.6263719;
    f64 n1 = -64847.90278878072;
    f64 d0 = +345190107.69729966;
    f64 d1 = +12653.90131058989;
    f64 d2 = -0.37409509100171673;
    f64 Num = n0 + n1*x1 + x2;
    f64 Den = d0 + d1*x1 + d2*x2;
    return Num / Den;
}

local f64 amb_a(f64 alt) {
    // Just say properties below sea level are constant.
    alt = br_max(alt, 0.0);
    // Above 60k, the speed of sound keeps changing but we dont keep calcing it,
    // since we'd only ever use it together with density and that goes to zero so
    // it becomes irrelevant.
    if (alt > 60e3)
        alt = 60e3;
    f64 x1 = alt + 10e3; // offset to avoid /0.
    f64 x2 = x1*x1;
    f64 x3 = x2*x1;
    f64 x4 = x3*x1;
    f64 n1 = +2453911045.412046;
    f64 d0 = +27961612000.005264;
    f64 d1 = -46762.970478368705;
    f64 d2 = +565.8014318466305;
    f64 d3 = -0.00988835027584721;
    f64 d4 = +9.180518407728973e-08;
    f64 Num = n1*x1 + x3;
    f64 Den = d0 + d1*x1 + d2*x2 + d3*x3 + d4*x4;
    return Num / Den;
}

// oh yeah we modelling gravity dropoff in this bitch.
local f64 amb_g(f64 alt) {
    // Newtons law of universal gravitation:
    //  F = G*M*m / r^2
    //  g = G*M / r^2
    // let:
    // R = 6.3568e6  [radius of earth]
    // g0 = G*M / R^2  [standard gravity]
    //  g = G*M / r^2 * R^2 / R^2
    //  g = R^2 / r^2 * G*M / R^2
    //  g = R^2 / r^2 * g0
    //  g = R^2 / (R + alt)^2 * g0
    f64 R = 6.3568e6;
    f64 factor = R / (R + alt);
    return STANDARD_GRAVITY * factor*factor;
}




// Rocket properties from Lemaire aerodynamic data, approximated via rational
// polynomials.
// TODO: use real rocket cd

local f64 rocket_CD(f64 mach) {
    // https://www.desmos.com/calculator/lbmxixzaiu
    f64 n0 = +0.259939613952;
    f64 n1 = -0.0252166618307;
    f64 n2 = -0.809172430601;
    f64 n3 = +0.641098370634;
    f64 d0 = +0.731852733359;
    f64 d1 = +0.630499692542;
    f64 d2 = -3.3620559129;
    f64 d3 = +2.12978762124;
    return (
        (n0 + mach*(n1 + mach*(n2 + mach*n3)))
        /
        (d0 + mach*(d1 + mach*(d2 + mach*d3)))
    );
}



// Circles and circle-adjacent functions.

local f64 circ_area(f64 D) {
    return PI_4 * D*D;
}
local f64 cyl_area(f64 L, f64 D) { // only curved face.
    return L * PI * D;
}
local f64 tube_vol(f64 L, f64 ID, f64 OD) {
    return L * PI_4 * (OD*OD - ID*ID);
}


typedef struct Pipe {
    f64 L;  // cannot be `UNSET`.
    f64 ID; // may be `UNSET`.
    f64 OD; // may be `UNSET`.
} Pipe;
local f64 pipe_th(const Pipe* pipe) {
    assert(ISSET(pipe->ID));
    assert(ISSET(pipe->OD));
    return 0.5 * (pipe->OD - pipe->ID);
}
local f64 pipe_V(Pipe* pipe) {
    assert(ISSET(pipe->ID));
    assert(ISSET(pipe->OD));
    return tube_vol(pipe->L, pipe->ID, pipe->OD);
}
local void pipe_set_th(Pipe* pipe, f64 th) {
    assert(ISSET(pipe->ID) + ISSET(pipe->OD) != 2);
    assert(ISSET(pipe->ID) + ISSET(pipe->OD) != 0);
    if (ISSET(pipe->ID)) {
        assertx(th >= 0.0, "ID=%f, th=%f", pipe->ID, th);
        pipe->OD = pipe->ID + 2.0*th;
    } else {
        assertx(2.0*th <= pipe->OD, "OD=%f, th=%f", pipe->OD, th);
        pipe->ID = pipe->OD - 2.0*th;
    }
}
local void pipe_set_th_for_P(Pipe* pipe, f64 P, f64 Ys, f64 sf) {
    // Finding min thickness for hoop stress, using thin-walled approximation:
    //  th = sf * P * ((ID + OD) / 2) / (2 * Ys)
    // Consider:
    // OD = ID + 2*th  [constrained inner diameter]
    //  th = sf * P * (2*ID + 2*th) / 4 / Ys
    //  th = sf * P * (ID + th) / 2 / Ys
    //  th = sf * P/2/Ys * ID + sf * P/2/Ys * th
    //  th - sf * P/2/Ys * th = sf * P/2/Ys * ID
    //  th * (1 - sf * P/2/Ys) = sf * P/2/Ys * ID
    //  th * (2 * Ys - sf * P) = sf * P * ID
    // => th = sf * P * ID / (2 * Ys - sf * P)
    // Consider:
    // ID = OD - 2*th  [constrained outer diameter]
    //  th = sf * P * (2*OD - 2*th) / 4 / Ys
    //  th = sf * P * (OD - th) / 2 / Ys
    //  th = sf * P/2/Ys * OD - sf * P/2/Ys * th
    //  th + sf * P/2/Ys * th = sf * P/2/Ys * OD
    //  th * (1 + sf * P/2/Ys) = sf * P/2/Ys * OD
    //  th * (2 * Ys + sf * P) = sf * P * OD
    // => th = sf * P * OD / (2 * Ys + sf * P)
    // Note that these relations mean it is not always possible to construct a
    // thick enough tank (in either sitation).
    assert(ISSET(pipe->ID) + ISSET(pipe->OD) != 2);
    assert(ISSET(pipe->ID) + ISSET(pipe->OD) != 0);
    if (ISSET(pipe->ID)) {
        f64 th = sf * P * pipe->ID / (2.0 * Ys - sf * P);
        assertx(th > 0.0, "ID=%f, th=%f, P=%f, Ys=%f, sf=%f", pipe->ID, th, P,
                Ys, sf);
        pipe->OD = pipe->ID + 2.0*th;
    } else {
        f64 th = sf * P * pipe->OD / (2.0 * Ys + sf * P);
        assertx(2.0*th < pipe->OD, "OD=%f, th=%f, P=%f, Ys=%f, sf=%f", pipe->OD,
                th, P, Ys, sf);
        pipe->ID = pipe->OD - 2.0*th;
    }
}



// Injector/orifice flow.

typedef struct Hole {
    f64 Cd;
    f64 A;
    f64 P_u;
    f64 P_d;
    f64 y;
    f64 R;
    f64 T;
    f64 Z;
} Hole;
// Real compressible isentropic flow through an orifice, considering both choked
// and unchoked possibilities.
local f64 hole_dm(const Hole* h) {
    // Let:
    //  Pr_critical = (2 / (y + 1)) ** (y / (y - 1))
    // Choked when:
    //  (P_u / P_d) >= Pr_critical
    // Equivalently:
    // Let: Pr = P_d / P_u
    //  Pr <= Pr_critical
    // Otherwise unchoked.
    // Choked flow rate:
    //  alpha = (2 / (y + 1)) ** ((y + 1) / (y - 1))
    // Unchoked flow rate:
    //  alpha = 2 / (y - 1) * (Pr**(2 / y) - Pr**((y + 1) / y))
    // Mass flow rate:
    //  dm = P_u * Cd * A * (alpha * y / (Z * R * T))**(1/2)
    //
    // However, can save some pow calls by doing some thinking:
    // Consider:
    //  Pr_critical = (2 / (y + 1)) ** (y / (y - 1))
    //  Pr_critical = (2 / (y + 1)) ** ((y - 1 + 1) / (y - 1))
    //  Pr_critical = (2 / (y + 1)) ** ((y - 1) / (y - 1) + 1 / (y - 1))
    //  Pr_critical = (2 / (y + 1)) ** (1 + 1 / (y - 1))
    //  Pr_critical = (2 / (y + 1)) * (2 / (y + 1)) ** (1 / (y - 1))
    // Let:
    // beta = (2 / (y + 1)) ** (1 / (y - 1))
    //  Pr_critical = (2 / (y + 1)) * beta
    // Consider choked alpha:
    //  alpha = (2 / (y + 1)) ** ((y + 1) / (y - 1))
    //  alpha = (2 / (y + 1)) ** ((y - 1 + 2) / (y - 1))
    //  alpha = (2 / (y + 1)) ** ((y - 1) / (y - 1) + 2 / (y - 1))
    //  alpha = (2 / (y + 1)) ** (1 + 2 / (y - 1))
    //  alpha = (2 / (y + 1)) * (2 / (y + 1)) ** (2 / (y - 1))
    //  alpha = (2 / (y + 1)) * ((2 / (y + 1)) ** (1 / (y - 1))) ** 2
    //  alpha = (2 / (y + 1)) * beta**2
    //  alpha = Pr_critical * beta
    // Now consider unchoked alpha:
    //  alpha = 2 / (y - 1) * (Pr**(2 / y) - Pr**((y + 1) / y))
    //  alpha = 2 / (y - 1) * (Pr**(2 / y) - Pr**(1 + 1/y))
    //  alpha = 2 / (y - 1) * (Pr**(1 / y)**2 - Pr * Pr**(1 / y))
    //  alpha = 2 / (y - 1) * Pr**(1 / y) * (Pr**(1 / y) - Pr)
    // Let: gamma = Pr**(1/y)
    //  alpha = 2 / (y - 1) * gamma * (gamma - Pr)
    // Easy.
    f64 Cd = h->Cd;
    f64 A = h->A;
    f64 P_u = h->P_u;
    f64 P_d = h->P_d;
    f64 y = h->y;
    f64 R = h->R;
    f64 T = h->T;
    f64 Z = h->Z;
    // No upstream pressure no flow (and we dont do backflow).
    if (P_u == 0.0)
        return 0.0;
    f64 Pr = P_d / P_u;
    // If downstream has a larger pressure than upstream, we assume no flow. If
    // back-flow would be a problem, it must be handled by the caller.
    if (Pr >= 1.0)
        return 0.0;
    f64 beta = br_pow(2 / (y + 1), 1 / (y - 1));
    f64 Pr_critical = 2 / (y + 1) * beta;
    f64 alpha;
    if (Pr <= Pr_critical) {
        alpha = Pr_critical * beta;
    } else {
        f64 gamma = br_pow(Pr, 1 / y);
        alpha = 2 / (y - 1) * gamma * (gamma - Pr);
    }
    return P_u * Cd * A * br_sqrt(alpha * y / Z / R / T);
}




// Some (tweakable) system constants.

#define TYPICAL_Dt (0.001) // [s], discrete calculus over time.
#define GOTIME_Dt (0.1) // [-], discrete calculus over time when gotiming.
#define TYPICAL_DT (0.05) // [K], discrete calculus over temperature.
#define MAX_SIM_t (1000.0) // [s], fail if still simming after this long.
#define MAX_STARTUP_t (0.25) // [s], worst-case longest startup time.
#define NEGLIGIBLE_m (0.002) // [kg], assume nothing if <= this.
#define CUTOFF_Pr (1.2) // [-], minimum injector pressure ratio for no back-flow.




// Now the simulation functions.


local i32 sim_has_optionals(broState* s) {
    return (s->count > 0);
}
typedef struct Optionals {
    f64 t;
    f64 alt_r;
    f64 vel_r;
    f64 acc_r;
    f64 m_r;
    f64 com_r;
    f64 T_t;
    f64 T_g;
    f64 P_t;
    f64 P_c;
    f64 P_a;
    f64 m_l;
    f64 m_v;
    f64 m_f;
    f64 dm_inj;
    f64 dm_reg;
    f64 dm_out;
    f64 m_g;
    f64 cp_g;
    f64 cv_g;
    f64 y_g;
    f64 R_g;
    f64 ofr;
    f64 Fthrust;
    f64 Fdrag;
    f64 Fgravity;
} Optionals;
local void sim_set_optionals(broState* s, Optionals* opt) {
    assert(sim_has_optionals(s));
    assertx(s->upto >= 0, "upto=%d, count=%d", s->upto, s->count);
    assertx(s->upto < s->count, "upto=%d, count=%d", s->upto, s->count);
    if (s->out_t)        s->out_t[s->upto]        = opt->t;
    if (s->out_alt_r)    s->out_alt_r[s->upto]    = opt->alt_r;
    if (s->out_vel_r)    s->out_vel_r[s->upto]    = opt->vel_r;
    if (s->out_acc_r)    s->out_acc_r[s->upto]    = opt->acc_r;
    if (s->out_m_r)      s->out_m_r[s->upto]      = opt->m_r;
    if (s->out_com_r)    s->out_com_r[s->upto]    = opt->com_r;
    if (s->out_T_t)      s->out_T_t[s->upto]      = opt->T_t;
    if (s->out_T_g)      s->out_T_g[s->upto]      = opt->T_g;
    if (s->out_P_t)      s->out_P_t[s->upto]      = opt->P_t;
    if (s->out_P_c)      s->out_P_c[s->upto]      = opt->P_c;
    if (s->out_P_a)      s->out_P_a[s->upto]      = opt->P_a;
    if (s->out_m_l)      s->out_m_l[s->upto]      = opt->m_l;
    if (s->out_m_v)      s->out_m_v[s->upto]      = opt->m_v;
    if (s->out_m_f)      s->out_m_f[s->upto]      = opt->m_f;
    if (s->out_dm_inj)   s->out_dm_inj[s->upto]   = opt->dm_inj;
    if (s->out_dm_reg)   s->out_dm_reg[s->upto]   = opt->dm_reg;
    if (s->out_dm_out)   s->out_dm_out[s->upto]   = opt->dm_out;
    if (s->out_m_g)      s->out_m_g[s->upto]      = opt->m_g;
    if (s->out_cp_g)     s->out_cp_g[s->upto]     = opt->cp_g;
    if (s->out_cv_g)     s->out_cv_g[s->upto]     = opt->cv_g;
    if (s->out_y_g)      s->out_y_g[s->upto]      = opt->y_g;
    if (s->out_R_g)      s->out_R_g[s->upto]      = opt->R_g;
    if (s->out_ofr)      s->out_ofr[s->upto]      = opt->ofr;
    if (s->out_Fthrust)  s->out_Fthrust[s->upto]  = opt->Fthrust;
    if (s->out_Fdrag)    s->out_Fdrag[s->upto]    = opt->Fdrag;
    if (s->out_Fgravity) s->out_Fgravity[s->upto] = opt->Fgravity;
    ++s->upto;
}



// Sets the initial state of the system.
local void sim_initial(broState* s) {

    // Rocket cross-sectional area.
    s->A_r = circ_area(s->D_r);

    // Using rule-of-thumb pre- and post-cc lengths:
    f64 cc_pre_length = s->D_c;
    f64 cc_post_length = 1.5 * s->D_c;
    s->L_c = cc_pre_length + s->L_f + cc_post_length;

    // Determine tank specs by assuming the largest pressure it needs to handle
    // is the N2O critical pressure.
    Pipe tank_wall = { .L=s->L_tw, .ID=UNSET, .OD=s->D_r };
    pipe_set_th_for_P(&tank_wall, N2O_Pcrit, s->Ys_tw, s->sf_tw);
    // TODO: establish exact tank mass, for now just assume thick ends.
    Pipe tank_wall_end = { .L=2*pipe_th(&tank_wall), .ID=0, .OD=s->D_r };
    Pipe tank = { .L=s->L_tw, .ID=0.0, .OD=tank_wall.ID };
    Pipe cc = { .L=s->L_c, .ID=0.0, .OD=s->D_c };
    Pipe fuel0 = { .L=s->L_f, .ID=UNSET, .OD=s->D_c };
    pipe_set_th(&fuel0, s->th0_f);

    s->V_t = pipe_V(&tank);
    s->m_tw = s->rho_tw * pipe_V(&tank_wall)
            + 2 * s->rho_tw * pipe_V(&tank_wall_end);
    s->C_tw = s->m_tw * s->c_tw;

    s->Vempty_c = pipe_V(&cc);

    // Fully specing the cc walls requires knowing the max pressure, which we
    // wont until we sim the combustion so initially we just use a ballpark
    // thickness and once we know Now do cc walls for cc max pressure.
    Pipe cc_wall = { .L=s->L_c, .ID=s->D_c, .OD=UNSET };
    f64 ballpark_max_P_c = 3e6; // =3MPa
    pipe_set_th_for_P(&cc_wall, ballpark_max_P_c, s->Ys_cw, s->sf_cw);
    s->th_cw = UNSET; // flag that walls still need to be speced.
    s->m_cw = s->rho_cw * pipe_V(&cc_wall);

    // TODO: nozzle specs
    s->L_nzl = 0.10;
    s->m_nzl = 2;


    // Setup the running variables with their initial values.
    broRunning i;

    // Already ignited.
    i.onfire = 1;

    // Start at the start.
    i.t = 0.0;

    // Assuming ox tank at ambient temperature and a saturated mixture.
    i.T_t = s->T_a;
    f64 V0_l = s->V_t * s->vff0_l;
    f64 V0_v = s->V_t - V0_l;
    i.m_l = V0_l * N2O_rho_satliq(i.T_t);
    i.m_v = V0_v * N2O_rho_satvap(i.T_t);

    i.ID_f = fuel0.ID;

    // Combustion chamber initially filled with ambient properties.
    f64 V0_c = s->Vempty_c - tube_vol(s->L_f, i.ID_f, s->D_c);
    //  P*V = m*R*T  [ideal gas law]
    // => m = P*V / (R*T)
    i.m_g = amb_P(s->alt0_r) * V0_c / AMB_R / s->T_a;
    i.N_g = i.m_g / AMB_Mw;
    i.T_g = s->T_a;
    i.Cp_g = i.m_g * AMB_STANDARD_CONDITIONS_cp;
    //  cp - cv = R  [ideal gas law]
    // => cv = cp - R
    i.Cv_g = i.m_g * (AMB_STANDARD_CONDITIONS_cp - AMB_R);

    // Initially at rest at the given altitude.
    i.alt_r = s->alt0_r;
    i.vel_r = 0.0;

    // Setup all the initial optional outputs.
    if (sim_has_optionals(s)) {
        // Same deduction logic as the genuine sims.

        f64 t     = i.t;
        f64 T_t   = i.T_t;
        f64 m_l   = i.m_l;
        f64 m_v   = i.m_v;
        f64 ID_f  = i.ID_f;
        f64 m_g   = i.m_g;
        f64 N_g   = i.N_g;
        f64 T_g   = i.T_g;
        f64 Cp_g  = i.Cp_g;
        f64 Cv_g  = i.Cv_g;
        f64 alt_r = i.alt_r;
        f64 vel_r = i.vel_r;

        f64 P_t = N2O_Psat(T_t);

        f64 P_a = amb_P(alt_r);

        f64 V_f = tube_vol(s->L_f, ID_f, s->D_c);
        f64 V_c = s->Vempty_c - V_f;
        f64 m_f = PARAFFIN_rho * V_f;

        f64 y_g = Cp_g / Cv_g;
        f64 cp_g = Cp_g / m_g;
        f64 cv_g = Cv_g / m_g;
        f64 Mw_g = m_g / N_g;
        f64 R_g = GAS_CONSTANT / Mw_g;
        f64 P_c = m_g * R_g * T_g / V_c;

        f64 m_r = s->m_locked
                + s->m_tw + m_l + m_v
                + s->m_inj
                + s->m_mov
                + s->m_cw + m_f
                + s->m_nzl;

        // All derivatives are zero initially.
        f64 acc_r = 0.0;
        f64 dm_inj = 0.0;
        f64 dm_reg = 0.0;
        f64 dm_out = 0.0;
        f64 ofr = 0.0;
        f64 Fthrust = 0.0;
        f64 Fdrag = 0.0;
        f64 Fgravity = 0.0;

        sim_set_optionals(s, &(Optionals){
            .t=t,
            .alt_r=alt_r,
            .vel_r=vel_r,
            .acc_r=acc_r,
            .m_r=m_r,
            .com_r=UNSET, // TODO: com.
            .T_t=T_t,
            .T_g=T_g,
            .P_t=P_t,
            .P_c=P_c,
            .P_a=P_a,
            .m_l=m_l,
            .m_v=m_v,
            .m_f=m_f,
            .dm_inj=dm_inj,
            .dm_reg=dm_reg,
            .dm_out=dm_out,
            .m_g=m_g,
            .cp_g=cp_g,
            .cv_g=cv_g,
            .y_g=y_g,
            .R_g=R_g,
            .ofr=ofr,
            .Fthrust=Fthrust,
            .Fdrag=Fdrag,
            .Fgravity=Fgravity,
        });
    }
    s->running = i;
}



// Sets the members of `derivatives` to the current time-derivative of each
// variable, except `t` is ignored and `ontime` is set to its new state, since it
// is discrete and does not have a derivative.
local void sim_diff(broState* s, broRunning* rstr derivatives) {
    // Current state variables.
    i32 onfire = s->running.onfire;
    f64 t     = s->running.t;
    f64 T_t   = s->running.T_t;
    f64 m_l   = s->running.m_l;
    f64 m_v   = s->running.m_v;
    f64 ID_f  = s->running.ID_f;
    f64 m_g   = s->running.m_g;
    f64 N_g   = s->running.N_g;
    f64 T_g   = s->running.T_g;
    f64 Cp_g  = s->running.Cp_g;
    f64 Cv_g  = s->running.Cv_g;
    f64 alt_r = s->running.alt_r;
    f64 vel_r = s->running.vel_r;

    // Ambient pressure changes with altitude.
    f64 P_a = amb_P(alt_r);

    // Reconstruct some cc/fuel state.
    f64 V_f = tube_vol(s->L_f, ID_f, s->D_c);
    f64 V_c = s->Vempty_c - V_f;
    f64 m_f = PARAFFIN_rho * V_f;

    // Reconstruct some cc gas state.
    f64 y_g = Cp_g / Cv_g;
    f64 cp_g = Cp_g / m_g;
    f64 cv_g = Cv_g / m_g;
    f64 Mw_g = m_g / N_g;
    f64 R_g = GAS_CONSTANT / Mw_g;

    // Assuming combustion gases are ideal:
    //  P*V = m*R*T
    //  P = m*R*T/V
    // Note if we arent burning, we stop simulating cc gases and assume ambient
    // pressure inside the cc chamber.
    f64 P_c = (onfire) ? (m_g * R_g * T_g / V_c) : P_a;
    // Note this is assuming an iso-thermal expansion of the cc gases into the
    // space left by the fuel grain eroding. While ideally we might assume
    // adiabatic the difference would be absolutely tiny.


    // Once the motor burn has started and we have a stable pressure, we assume
    // this is the highest pressure the cc will experience and can now properly
    // spec the walls for this pressure. Note that this does mean a tiny bit of
    // the trajectory is done using the wrong mass, but thats so fine.
    if ((t > MAX_STARTUP_t) && !ISSET(s->th_cw)) {
        Pipe cc_wall = { .L=s->L_c, .ID=s->D_c, .OD=UNSET };
        pipe_set_th_for_P(&cc_wall, P_c, s->Ys_cw, s->sf_cw);
        s->th_cw = pipe_th(&cc_wall);
        s->m_cw = s->rho_cw * pipe_V(&cc_wall);
    }


    // If its still simming, somethings gone wrong.
    assert(s->running.t < MAX_SIM_t);


    // Properties determined around injector:
    f64 P_t;
    f64 dm_l;
    f64 dm_v;
    f64 dm_inj;
    f64 dT_t;

    // Liquid draining while there's any liquid in the tank.
    if (m_l > NEGLIGIBLE_m) {
        P_t = N2O_Psat(T_t); // tank at saturated pressure.


        // Find injector flow rate.

        f64 T_u = T_t;
        f64 P_u = P_t;
        f64 P_d = P_c;
        f64 T_d = N2O_Tsat(P_d); // since our prop funcs expect temp.
        // If the pressure is outside our approximation input bounds, clamp it
        // to be inside the correct range.
        P_d = br_clamp(P_d, N2O_Pmin, N2O_Pmax);

        // Check theres no injector back-flow during a burn.
        if (onfire)
            assertx(P_u > CUTOFF_Pr * P_d, "P_u=%fPa, P_d=%fPa", P_u, P_d);
        // NHEM requires a pressure difference to not crack it.
        if (P_u <= P_d)
            goto NO_INJECTOR_FLOW;

        // Single-phase incompressible model (with Beta = 0):
        // (assuming upstream density as the "incompressible" density)
        f64 rho_u = N2O_rho_satliq(T_u);
        f64 dm_SPI = s->Cd_inj * s->A_inj * br_sqrt(2 * rho_u * (P_u - P_d));

        // Homogenous equilibrium model:
        // (assuming only saturated liquid leaving from upstream)
        f64 s_u = N2O_s_satliq(P_u);
        f64 s_d_l = N2O_s_satliq(P_d);
        f64 s_d_v = N2O_s_satvap(P_d);
        f64 x_d = (s_u - s_d_l) / (s_d_v - s_d_l);
        assert(0 <= x_d && x_d <= 1.0);
        f64 h_u = N2O_h_satliq(T_u);
        f64 h_d_l = N2O_h_satliq(T_d);
        f64 h_d_v = N2O_h_satvap(T_d);
        f64 h_d = (1 - x_d) * h_d_l + x_d * h_d_v;
        f64 rho_d_l = N2O_rho_satliq(T_d);
        f64 rho_d_v = N2O_rho_satvap(T_d);
        // gotta do the quality ratios in specific-volume.
        f64 v_d_l = 1 / rho_d_l;
        f64 v_d_v = 1 / rho_d_v;
        f64 v_d = (1 - x_d) * v_d_l + x_d * v_d_v;
        f64 rho_d = 1 / v_d;
        f64 dm_HEM = s->Cd_inj * s->A_inj * rho_d * br_sqrt(2 * (h_u - h_d));

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
        dm_inj = dm_SPI * (1 - k_NHNE) + dm_HEM * k_NHNE;


        // To determine temperature and vapourised mass derivatives, we're going
        // to have to use: our brain.
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
        //  d/dt (U_tw + U_l + U_v) = -dm_inj * h_l
        //  d/dt (m_tw*u_tw) + d/dt (m_l*u_l) + d/dt (m_v*u_v) = -dm_inj * h_l
        //  -dm_inj * h_l = dm_tw*u_tw + m_tw*du_tw
        //                + dm_l*u_l + m_l*du_l
        //                + dm_v*u_v + m_v*du_v
        // dm_tw = 0  [wall aint going anywhere]
        // dm_l = -dm_v - dm_inj  [same as earlier]
        //  -dm_inj * h_l = m_tw*du_tw + m_l*du_l + m_v*du_v
        //                + (-dm_v - dm_inj) * u_l
        //                + dm_v*u_v
        //  dm_inj * (u_l - h_l) = m_tw*du_tw + m_l*du_l + m_v*du_v
        //                       - dm_v*u_l
        //                       + dm_v*u_v
        //  dm_inj * (u_l - h_l) = m_tw*du_tw + m_l*du_l + m_v*du_v
        //                       + dm_v * (u_v - u_l)
        // du = d/dt (u) = d/dT (u) * dT/dt
        // also note:
        //   u = int (cv) dT
        //   d/dT (u) = cv
        // therefore:
        //   du = dT * cv
        //  dm_inj * (u_l - h_l) = dT * (m_tw*cv_tw + m_l*cv_l + m_v*cv_v)
        //                       + dm_v * (u_v - u_l)
        // let: Cv = m_tw*cv_tw + m_l*cv_l + m_v*cv_v
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

        f64 rho_l = N2O_rho_satliq(T_t);
        f64 rho_v = N2O_rho_satvap(T_t);
        f64 drhodT_l = (N2O_rho_satliq(T_t + TYPICAL_DT) - rho_l) / TYPICAL_DT;
        f64 drhodT_v = (N2O_rho_satvap(T_t + TYPICAL_DT) - rho_v) / TYPICAL_DT;

        f64 Cv_l = m_l * N2O_cv_satliq(T_t);
        f64 Cv_v = m_v * N2O_cv_satvap(T_t);
        f64 Cv = Cv_l + Cv_v + s->C_tw;

        f64 h_l = N2O_h_satliq(T_t);
        f64 u_l = N2O_u_satliq(T_t);
        f64 u_v = N2O_u_satvap(T_t);

        f64 foo = dm_inj / rho_l / (1/rho_v - 1/rho_l);
        f64 bar = (m_l * drhodT_l / (rho_l*rho_l)
                 + m_v * drhodT_v / (rho_v*rho_v)) / (1/rho_v - 1/rho_l);

        dT_t = (dm_inj * (u_l - h_l) + foo * (u_l - u_v))
             / (Cv - bar * (u_l - u_v));

        dm_v = foo + dT_t * bar;
        dm_l = -dm_inj - dm_v;


    // Otherwise vapour draining.
    } else if (m_v > NEGLIGIBLE_m) {
        dm_l = 0.0; // liquid mass is ignored hence fourth (big word init).

        // During this period, temperature and density are used to fully define
        // the state (due to fixed volume).
        f64 rho_v = m_v / s->V_t;

        // Due to numerical inaccuracy, might technically have the properties of
        // a saturated mixture so just pretend its a saturated vapour.
        f64 rhosat_v = N2O_rho_satvap(T_t);
        rho_v = br_clamp(rho_v, N2O_rhomin, rhosat_v);

        // Get tank pressure from the two properties.
        P_t = N2O_P(T_t, rho_v);
        P_t = br_clamp(P_t, N2O_Pmin, N2O_Pmax);


        // Find injector flow rate.

        f64 T_u = T_t;
        f64 P_u = P_t;
        f64 P_d = P_c;
        P_d = br_clamp(P_d, N2O_Pmin, N2O_Pmax);

        // Check theres no injector back-flow during a burn.
        if (onfire)
            assertx(P_u > CUTOFF_Pr * P_d, "P_u=%fPa, P_d=%fPa", P_u, P_d);

        // Technically gamma but use 'y' for file size reduction.
        f64 y_u = N2O_cp(T_u, P_u) / N2O_cv(T_u, P_u);
        // Use compressibility factor to account for non-ideal gas.
        f64 Z_u = N2O_Z(T_u, rho_v);

        // Use real compressible orifice flow rate.
        dm_inj = hole_dm(&(Hole){
            .Cd = s->Cd_inj,
            .A = s->A_inj,
            .P_u = P_u,
            .P_d = P_d,
            .y = y_u,
            .R = N2O_R,
            .T = T_u,
            .Z = Z_u,
        });

        // Mass only leaves through injector, and no state change.
        dm_v = -dm_inj;


        // Back to the well.
        //  d/dt (U) = -dm_inj * h  [first law of thermodynamics, adiabatic]
        // using no suffix as the non-saturated vapour in the tank:
        //  d/dt (U_tw + U) = -dm_inj * h
        //  d/dt (m_tw*u_tw) + d/dt (m*u) = -dm_inj * h
        //  -dm_inj * h = dm_tw*u_tw + m_tw*du_tw
        //              + dm*u + m*du
        // dm_tw = 0  [wall aint going anywhere]
        // dm = -dm_inj  [only mass change is from injector]
        //  -dm_inj * h = m_tw * du_tw
        //              - dm_inj * u
        //              + m * du
        //  dm_inj * (u - h) = m_tw * du_tw + m * du
        // du = dT * cv  [previously derived]
        //  dm_inj * (u - h) = dT * (m_tw * cv_tw + m * cv)
        // let: Cv = m_tw * cv_tw + m * cv
        //  dm_inj * (u - h) = dT * Cv
        // => dT = dm_inj * (u - h) / Cv
        // which makes sense, since only energy change is due to boundary work.

        f64 u_u = N2O_u(T_t, rho_v);
        f64 h_u = N2O_h(T_t, rho_v);

        f64 Cv = s->C_tw + m_v * N2O_cv(T_t, P_u);

        dT_t = dm_inj * (u_u - h_u) / Cv;


    // No oxidiser left.
    } else {
        P_t = 0.0;
      NO_INJECTOR_FLOW:;
        dm_l = 0.0;
        dm_v = 0.0;
        dm_inj = 0.0;
        dT_t = 0.0;
    }



    // Do fuel regression.
    f64 dID_f;
    f64 dm_reg;
    // Gotta be fire and fuel left to regress.
    if (onfire && m_f > NEGLIGIBLE_m) {

        // Get oxidiser mass flux through the fuel grain.
        f64 G_o = dm_inj / circ_area(ID_f);
        // Use empircally determined regression rate.
        f64 rr_rdot = regression_rate_rdot(G_o);

        // Fuel mass and diameter change from rdot:
        dID_f = 2 * rr_rdot;
        f64 dV_f = rr_rdot * cyl_area(s->L_f, ID_f);
        dm_reg = PARAFFIN_rho * dV_f;

    // No fuel left.
    } else {
        dID_f = 0.0;
        dm_reg = 0.0;
    }



    // Do nozzle flow.
    f64 dm_out = hole_dm(&(Hole){
        .Cd = s->Cd_nzl,
        .A = s->A_throat,
        .P_u = P_c,
        .P_d = P_a,
        .y = y_g,
        .R = R_g,
        .T = T_g,
        .Z = 1.0,
    });



    // Do combustion things.

    f64 Fthrust;
    f64 dm_g;
    f64 dN_g;
    f64 dCp_g;
    f64 dCv_g;
    f64 dT_g;

    // Instantaneous oxidiser-fuel ratio.
    f64 ofr = (dm_reg != 0.0) ? (dm_inj / dm_reg) : 0.0;

    // During startup, we always assume combustion is occuring. Otherwise, stop
    // combustion if the ofr is too low, since combustion would probably stop.
    if (onfire || (t <= MAX_STARTUP_t)) {
        // Just pretend inputs are in bounds even if they arent.
        f64 P_cea = br_clamp(P_c, CEA_Pmin, CEA_Pmax);
        f64 ofr_cea = br_clamp(ofr, CEA_ofrmin, CEA_ofrmax);

        // Do cea to find new combustion product properties.
        f64 dm_n = dm_inj + dm_reg;
        f64 T_n = cea_T_c(P_cea, ofr_cea);
        f64 Mw_n = cea_Mw_c(P_cea, ofr_cea);
        f64 cp_n = cea_cp_c(P_cea, ofr_cea);
        // y = cp / cv  ->  cv = cp / y
        f64 cv_n = cp_n / cea_y_throat(P_cea, ofr_cea);
        f64 Ivac = cea_Ivac(P_cea, ofr_cea, s->eps);

        f64 A_exit = s->eps * s->A_throat;
        Fthrust = dm_out * Ivac * STANDARD_GRAVITY - P_a * A_exit;
        // Pretend it doesnt suck the rocket downwards.
        Fthrust = br_max(Fthrust, 0.0);

        // Net cc gas property changed.
        dm_g = dm_n - dm_out;

        // Change in any extensive property for a reservoir with flow in and out:
        //  d/dt (m*p) = dm_in * p_in - dm_out * p

        // Change in moles:
        //  dn = d/dt (n_n) - d/dt (n_out)
        //  dn = dm_n / Mw_n - dm_out / Mw
        dN_g = dm_n / Mw_n - dm_out / Mw_g;

        // Change in specific heats:
        dCp_g = dm_n * cp_n - dm_out * cp_g;
        dCv_g = dm_n * cv_n - dm_out * cv_g;

        // Change in temperature:
        //  d/dt (m * cp * T) = dm_n * cp_n * T_n - dm_out * cp * T
        //  d/dt (m * cp * T) = dm_n * cp_n * T_n - dm_out * cp * T
        //  d/dt (m * cp) * T + m*cp * dT = dm_n * cp_n * T_n - dm_out * cp * T
        //  dCp * T + Cp * dT = dm_n * cp_n * T_n - dm_out * cp * T
        //  Cp * dT = dm_n * cp_n * T_n - dm_out * cp * T - dCp * T
        //  dT = (dm_n * cp_n * T_n - dm_out * cp * T - dCp * T) / Cp
        dT_g = (dm_n * cp_n * T_n - dm_out * cp_g * T_g - dCp_g * T_g) / Cp_g;

        // Assume that combustion stops at <1 ofr.
        onfire = (ofr >= 1.0);

    } else {
        // Assume no thrust without combustion (i.e. ignore any pressure-only
        // thrust).
        Fthrust = 0.0;

        // Ignore cc gas properties if not combusting.
        dm_g = 0.0;
        dN_g = 0.0;
        dCp_g = 0.0;
        dCv_g = 0.0;
        dT_g = 0.0;
    }



    // Calculate drag using the current mach number of the rocket and the drag
    // coefficient calculated from that.
    f64 mach_r = vel_r / amb_a(alt_r);
    f64 CD_r = rocket_CD(mach_r);
    f64 A_r = circ_area(s->D_r);
    f64 Fdrag = -0.5 * amb_rho(alt_r) * vel_r*vel_r * CD_r * A_r;
    // TODO: add fin drag

    // Calculate weight force.
    f64 m_r = s->m_locked
            + s->m_tw + m_l + m_v
            + s->m_inj
            + s->m_mov
            + s->m_cw + m_f
            + s->m_nzl;
    f64 Fgravity = -m_r * amb_g(alt_r);

    // Calculate acceleration.
    //  F = m*a  [a classic, newton i believe]
    // => a = F/m
    f64 dvel_r = (Fthrust + Fdrag + Fgravity) / m_r;



    // Got all state time-dependant derivatives, lets pack it into the return
    // struct.
    derivatives->onfire = onfire;
    // derivatives->t ignored.
    derivatives->T_t    = dT_t;
    derivatives->m_l    = dm_l;
    derivatives->m_v    = dm_v;
    derivatives->ID_f   = dID_f;
    derivatives->m_g    = dm_g;
    derivatives->N_g    = dN_g;
    derivatives->T_g    = dT_g;
    derivatives->Cp_g   = dCp_g;
    derivatives->Cv_g   = dCv_g;
    derivatives->alt_r  = vel_r;
    derivatives->vel_r  = dvel_r;
}


// Simulates the entire rocket for one step. All running vars of the system are
// updated, and one extra round of optional output is set. Returns non-zero if
// the simulation is complete and should stop.
local i32 sim_ulate(broState* s) {
    // If not combusting, can step much much larger.
    f64 Dt = TYPICAL_Dt;
    if (!s->running.onfire) // gotime.
        Dt = GOTIME_Dt;

    // Do the four state derivatives that are needed for runge-kutta 4
    // integration.
    broRunning derivatives[4]; // all members are derivatives.
    broRunning cur = s->running;
    for (i32 i=0; i<4; ++i) {
        if (i > 0) {
            // Setup new input, based on rk4 formula.
            f64 rk4_Dt = (i == 3) ? Dt : 0.5 * Dt;
            broRunning* d = derivatives + i - 1;
            s->running = (broRunning){
                .onfire = d->onfire,
                .t     = cur.t     + rk4_Dt,
                .T_t   = cur.T_t   + rk4_Dt * d->T_t,
                .m_l   = cur.m_l   + rk4_Dt * d->m_l,
                .m_v   = cur.m_v   + rk4_Dt * d->m_v,
                .ID_f  = cur.ID_f  + rk4_Dt * d->ID_f,
                .m_g   = cur.m_g   + rk4_Dt * d->m_g,
                .N_g   = cur.N_g   + rk4_Dt * d->N_g,
                .T_g   = cur.T_g   + rk4_Dt * d->T_g,
                .Cp_g  = cur.Cp_g  + rk4_Dt * d->Cp_g,
                .Cv_g  = cur.Cv_g  + rk4_Dt * d->Cv_g,
                .alt_r = cur.alt_r + rk4_Dt * d->alt_r,
                .vel_r = cur.vel_r + rk4_Dt * d->vel_r,
            };
        }
        sim_diff(s, derivatives + i);
    }

    // Apply rk4 formula to find a weighted average of the four derivatives we
    // just calculated and use that for this entire step.
    broRunning* k1 = derivatives + 0;
    broRunning* k2 = derivatives + 1;
    broRunning* k3 = derivatives + 2;
    broRunning* k4 = derivatives + 3;
    i32 Donfire = (k1->onfire | k2->onfire | k3->onfire | k4->onfire);
    f64 dT_t   = (k1->T_t   + 2*k2->T_t   + 2*k3->T_t   + k4->T_t  ) / 6;
    f64 dm_l   = (k1->m_l   + 2*k2->m_l   + 2*k3->m_l   + k4->m_l  ) / 6;
    f64 dm_v   = (k1->m_v   + 2*k2->m_v   + 2*k3->m_v   + k4->m_v  ) / 6;
    f64 dID_f  = (k1->ID_f  + 2*k2->ID_f  + 2*k3->ID_f  + k4->ID_f ) / 6;
    f64 dm_g   = (k1->m_g   + 2*k2->m_g   + 2*k3->m_g   + k4->m_g  ) / 6;
    f64 dN_g   = (k1->N_g   + 2*k2->N_g   + 2*k3->N_g   + k4->N_g  ) / 6;
    f64 dT_g   = (k1->T_g   + 2*k2->T_g   + 2*k3->T_g   + k4->T_g  ) / 6;
    f64 dCp_g  = (k1->Cp_g  + 2*k2->Cp_g  + 2*k3->Cp_g  + k4->Cp_g ) / 6;
    f64 dCv_g  = (k1->Cv_g  + 2*k2->Cv_g  + 2*k3->Cv_g  + k4->Cv_g ) / 6;
    f64 dalt_r = (k1->alt_r + 2*k2->alt_r + 2*k3->alt_r + k4->alt_r) / 6;
    f64 dvel_r = (k1->vel_r + 2*k2->vel_r + 2*k3->vel_r + k4->vel_r) / 6;

    s->running.onfire = Donfire;
    s->running.t     += Dt;
    s->running.T_t   += Dt * dT_t;
    s->running.m_l   += Dt * dm_l;
    s->running.m_v   += Dt * dm_v;
    s->running.ID_f  += Dt * dID_f;
    s->running.m_g   += Dt * dm_g;
    s->running.N_g   += Dt * dN_g;
    s->running.T_g   += Dt * dT_g;
    s->running.Cp_g  += Dt * dCp_g;
    s->running.Cv_g  += Dt * dCv_g;
    s->running.alt_r += Dt * dalt_r;
    s->running.vel_r += Dt * dvel_r;


    // Send the optional outputs.
    if (sim_has_optionals(s)) {
        // Same deduction logic as the genuine sims.

        i32 onfire = s->running.onfire;
        f64 t     = s->running.t;
        f64 T_t   = s->running.T_t;
        f64 m_l   = s->running.m_l;
        f64 m_v   = s->running.m_v;
        f64 ID_f  = s->running.ID_f;
        f64 m_g   = s->running.m_g;
        f64 N_g   = s->running.N_g;
        f64 T_g   = s->running.T_g;
        f64 Cp_g  = s->running.Cp_g;
        f64 Cv_g  = s->running.Cv_g;
        f64 alt_r = s->running.alt_r;
        f64 vel_r = s->running.vel_r;

        f64 P_t = 0.0;
        if (m_l > NEGLIGIBLE_m) {
            P_t = N2O_Psat(T_t);
        } else if (m_v > NEGLIGIBLE_m) {
            f64 rho_v = m_v / s->V_t;
            f64 rhosat_v = N2O_rho_satvap(T_t);
            rho_v = br_clamp(rho_v, N2O_rhomin, rhosat_v);
            P_t = N2O_P(T_t, rho_v);
        }

        f64 P_a = amb_P(alt_r);

        f64 V_f = tube_vol(s->L_f, ID_f, s->D_c);
        f64 V_c = s->Vempty_c - V_f;
        f64 m_f = PARAFFIN_rho * V_f;

        f64 y_g = Cp_g / Cv_g;
        f64 cp_g = Cp_g / m_g;
        f64 cv_g = Cv_g / m_g;
        f64 Mw_g = m_g / N_g;
        f64 R_g = GAS_CONSTANT / Mw_g;
        f64 P_c = m_g * R_g * T_g / V_c;

        f64 m_r = s->m_locked
                + s->m_tw + m_l + m_v
                + s->m_inj
                + s->m_mov
                + s->m_cw + m_f
                + s->m_nzl;

        f64 dm_inj = -dm_l - dm_v;

        f64 dV_f = (0.5 * dID_f) * cyl_area(s->L_f, ID_f);
        f64 dm_reg = PARAFFIN_rho * dV_f;

        f64 dm_out = dm_inj + dm_reg - dm_g;

        f64 ofr = (dm_reg != 0.0) ? (dm_inj / dm_reg) : 0.0;

        f64 Fthrust = 0.0;
        if (onfire) {
            f64 P_cea = br_clamp(P_c, CEA_Pmin, CEA_Pmax);
            f64 ofr_cea = br_clamp(ofr, CEA_ofrmin, CEA_ofrmax);
            f64 Ivac = cea_Ivac(P_cea, ofr_cea, s->eps);
            f64 A_exit = s->eps * s->A_throat;
            Fthrust = dm_out * Ivac * STANDARD_GRAVITY - P_a * A_exit;
            Fthrust = br_max(Fthrust, 0.0);
        }

        f64 mach_r = vel_r / amb_a(alt_r);
        f64 CD_r = rocket_CD(mach_r);
        f64 Fdrag = -0.5 * amb_rho(alt_r) * vel_r*vel_r * CD_r * s->A_r;

        f64 Fgravity = -m_r * amb_g(alt_r);

        // cc gases + combustion not simmed after combustion stops, so set to
        // nan. Note that a lot of things are zero after cc stops, but that is
        // because they are genuinely zero whereas these properties are just
        // unknown.
        if (!onfire) {
            m_g = UNSET;
            N_g = UNSET;
            T_g = UNSET;
            Cp_g = UNSET;
            Cv_g = UNSET;
            y_g = UNSET;
            cp_g = UNSET;
            cv_g = UNSET;
            Mw_g = UNSET;
            R_g = UNSET;
            P_c = UNSET;
            dm_out = UNSET;
            Fthrust = UNSET;
        }

        sim_set_optionals(s, &(Optionals){
            .t=t,
            .alt_r=alt_r,
            .vel_r=vel_r,
            .acc_r=dvel_r,
            .m_r=m_r,
            .com_r=UNSET, // TODO: com.
            .T_t=T_t,
            .T_g=T_g,
            .P_t=P_t,
            .P_c=P_c,
            .P_a=P_a,
            .m_l=m_l,
            .m_v=m_v,
            .m_f=m_f,
            .dm_inj=dm_inj,
            .dm_reg=dm_reg,
            .dm_out=dm_out,
            .m_g=m_g,
            .cp_g=cp_g,
            .cv_g=cv_g,
            .y_g=y_g,
            .R_g=R_g,
            .ofr=ofr,
            .Fthrust=Fthrust,
            .Fdrag=Fdrag,
            .Fgravity=Fgravity,
        });
    }

    // Finished when we going downwards (and the initial bit is over, bc like we
    // actually dip into the ground immediately before rocketing outta there).
    return (s->running.t > MAX_STARTUP_t) && (s->running.vel_r <= 0.0);
}



// DLL-exposed function.
void bro_sim(broState* s) {
    // Setup the assert macro to abort back to here, and just get outta here with
    // whatever data we gened.
    if (assertion_failed())
        return;

    // Integrate the system differential.
    sim_initial(s);
    while (!sim_ulate(s));
}
