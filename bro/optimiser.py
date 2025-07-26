import traceback
import time
from math import pi

import numpy as np
import matplotlib.pyplot as plt
import rocketcea
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj_w_units import CEA_Obj

GAS_CONSTANT = 8.31446261815324 # [J/mol/K]



def singleton(cls):
    """
    Returns a single instantiation of the given class, and prevents
    further creation of the class.
    - class decorator.
    """
    instance = cls()
    def throw(cls, *args, **kwargs):
        raise TypeError(f"cannot create another instance of singleton {repr(cls.__name__)}")
    if getattr(instance, "__doc__", None) is None:
        if getattr(cls, "__doc__", None) is not None:
            instance.__doc__ = cls.__doc__
    cls.__new__ = throw
    return instance


def frozen(cls):
    """
    Disables creating or deleting new object attributes of the
    given class (outside the `__init__` method).
    - class decorator.
    """

    if hasattr(cls, "__slots__"):
        raise ValueError("cannot combine __slots__ and @frozen")

    cls_init = cls.__init__

    def __init__(self, *args, **kwargs):
        self._in_init = getattr(self, "_in_init", 0) + 1
        cls_init(self, *args, **kwargs)
        self._in_init -= 1

    def __setattr__(self, name, value):
        if not getattr(self, "_in_init", 1) and name not in self.__dict__:
            raise AttributeError(f"cannot add attribute {repr(name)} to a frozen instance")
        super(cls, self).__setattr__(name, value)

    def __delattr__(self, name):
        if not getattr(self, "_in_init", 1):
            raise AttributeError(f"cannot delete attribute {repr(name)} from a frozen instance")
        super(cls, self).__delattr__(name)

    cls.__init__ = __init__
    cls.__setattr__ = __setattr__
    cls.__delattr__ = __delattr__
    return cls




class Ingredient:
    TEMPERATURE = 293.15

    def __init__(self, kind, name, composition, hf, rho):
        self.name = name
        self.density = rho

        # use the same kind detection as nasa cea fortran code.
        if kind[:2] == "fu":
            kind = "fu"
            adder = rocketcea.cea_obj.add_new_fuel
        elif kind[:2] == "ox":
            kind = "ox"
            adder = rocketcea.cea_obj.add_new_oxidizer
        else:
            assert False, "invalid kind"
        adder(name,
            f"{kind} {name} {composition}\n"
            f"h,kc={hf:.1f}\n"
            f"t,k={self.TEMPERATURE}\n"
            f"rho,kg={rho:.1f}\n"
        )

class Fuel(Ingredient):
    def __init__(self, name, composition, hf, rho):
        """
        name ......... unique string id.
        composition .. string of chemical formula, of the form "X0 N0 X1 N1 ..." where
                       Xn is atomic symbol and Nn is number of atoms.
        hf .......... enthalpy of formation, in kcal/mol.
        rho ......... density at `Fuel.TEMPERATURE`, in kg/m^3.
        """
        super().__init__("fu", name, composition, hf, rho)

class Oxidiser(Ingredient):
    def __init__(self, name, composition, hf, rho):
        """
        name ......... unique string id.
        composition .. string of chemical formula, of the form "X0 N0 X1 N1 ..." where
                       Xn is atomic symbol and Nn is number of atoms.
        hf .......... enthalpy of formation, in kcal/mol.
        rho ......... density at `Oxidiser.TEMPERATURE`, in kg/m^3.
        """
        super().__init__("ox", name, composition, hf, rho)

    def prop(self, *args):
        """
        Forwards to PropsSI for this oxidiser.
        """
        return PropsSI(*args, self.name)

class FuOx:
    def __init__(self, fuel, ox, a0, n):
        """
        fuel ... Fuel object.
        ox ..... Oxidiser object.
        a0 ..... regression rate coefficient.
        n ...... regression rate exponent.
        """
        self.fuel = fuel
        self.ox = ox
        self.a0 = a0
        self.n = n

    def __repr__(self):
        return f"<FuOx: {self.fuel.name}+{self.ox.name}>"


# Compositions and enthalpy of formation from Hybrid Rocket Propulsion Handbook, Karp & Jens.
PARAFFIN = Fuel("PARAFFIN", "C 32 H 66", hf=-224.2, rho=924.5)
NOX = Oxidiser("N2O", "N 2 O 1", hf=15.5,
               rho=PropsSI("D", "T", Oxidiser.TEMPERATURE, "Q", 0, "N2O"))

# Regression rate data for paraffin and NOX, from Hybrid Rocket Propulsion Handbook, Karp & Jens.
PARAFFIN_NOX = FuOx(fuel=PARAFFIN, ox=NOX, a0=1.55e-4, n=0.5)




@singleton
class INPUT:
    def __repr__(self):
        return "<INPUT>"
@singleton
class OUTPUT:
    def __repr__(self):
        return "<OUTPUT>"
@frozen
class Sys:
    def __init__(s):
        r"""
Cheeky rocket ascii:

   .     .
   :     : ----- locked (by rest of team)
   |     |
   | ___ | _________ origin line (coms measured from here, +ve down)
   ||   ||
   ||   || ----- tank (contains oxidiser)
   ||___||
   |  X  | ----- mov (main oxidiser valve)
   | _#_ | ----- injector
   ||, ,||
   ||| ||| ----- cc (combustion chamber, contains fuel grain)
  /||' '||\
 / | \ / | \
/  |_/ \_|  \
|_/   |   \_| -- fins
      |
      '---------  nozzle

All dependant variables initially set to ellipsis. Note that the choice
of "basis" variables (the independant variables that are either input
or output) is arbitrary, but some make significantly more sense than
others (i.e. the relationship is difficult to inverse). Also note that
the decision between assigning an independant variable as input and
output is generally arbitrary (so long as controlled by us and not the
weather or contest for example), and the only real difference is that
output variables are swept over a range and the optimal choice is kept.
        """

        s.target_apogee = INPUT # [m]
        s.fuox = INPUT # must be `PARAFFIN_NOX`

        s.locked_mass = INPUT # [kg]
        s.locked_length = INPUT # [m]
        s.locked_local_com = INPUT # [m]
        s.locked_com = ... # [m]

        s.tank_inner_length = OUTPUT # [m]
        s.tank_length = ... # [m]
        s.tank_com = ... # [m]
        s.tank_wall_density = INPUT # [kg/m^3]
        s.tank_wall_yield_strength = INPUT # [Pa]
        s.tank_wall_specific_heat_capacity = INPUT # [J/kg/K]
        s.tank_wall_thickness = ... # [m]
        s.tank_wall_mass = ... # [kg]
        s.tank_temperature = ... # [K, over time]
        s.tank_pressure = ... # [Pa, over time]

        s.ox_volume_fill_frac = INPUT # [-]
        s.ox_worstcase_temperature = INPUT # [K]
        s.ox_mass = ... # [kg, over time]
        s.ox_mass_liquid = ... # [kg, over time]
        s.ox_mass_vapour = ... # [kg, over time]
        s.ox_com = ... # [m]

        s.mov_mass = INPUT # [kg]
        s.mov_length = INPUT # [m]
        s.mov_local_com = INPUT # [m]
        s.mov_com = ... # [m]

        s.injector_discharge_coeff = INPUT # [-]
        s.injector_orifice_area = OUTPUT # [m^2]
        s.injector_mass = INPUT # [kg]
        s.injector_length = INPUT # [m]
        s.injector_local_com = INPUT # [m]
        s.injector_com = ... # [m]

        s.cc_diameter = OUTPUT # [m]
        s.cc_combustion_efficiency = INPUT # [-]
        s.cc_temperature = ... # [K, over time]
        s.cc_pressure = ... # [Pa, over time]
        s.cc_pre_length = ... # [m]
        s.cc_post_length = ... # [m]
        s.cc_length = ... # [m]
        s.cc_wall_density = INPUT # [kg/m^3]
        s.cc_wall_yield_strength = INPUT # [Pa]
        s.cc_wall_thickness = ... # [m]
        s.cc_wall_mass = ... # [kg]
        s.cc_wall_com = ... # [m]

        s.fuel_length = OUTPUT # [m]
        s.fuel_initial_thickness = OUTPUT # [m]
        s.fuel_mass = ... # [kg, over time]
        s.fuel_com = ... # [m]

        s.nozzle_discharge_coeff = INPUT # [-]
        s.nozzle_thrust_efficiency = INPUT # [-]
        s.nozzle_throat_area = OUTPUT # [m^2]
        s.nozzle_exit_area = ... # [m^2]
        s.nozzle_length = ... # [m]
        s.nozzle_com = ... # [m]
        s.nozzle_mass = ... # [kg]

        s.rocket_diameter = INPUT # [m]
        s.rocket_length = ... # [m]
        s.rocket_mass = ... # [kg, over time]
        s.rocket_com = ... # [m, over time]
        s.rocket_drag_coeff = ... # [idk]
        s.rocket_stability = INPUT # [-]
        s.rocket_net_force = ... # [N, over time]
        s.rocket_altitude = ... # [m, over time]

        s.ambient_temperature = INPUT # [K]
        s.ambient_pressure = INPUT # [Pa]
        s.ambient_density = INPUT # [kg/m^3]
        s.ambient_molar_mass = INPUT # [kg/mol]
        s.ambient_constant_pressure_specific_heat_capacity = INPUT # [J/kg/K]
        # shortest bro variable name.

        s.thrust = ... # [N, over time]
        s.burn_time = ... # [s]

    @classmethod
    def input_names(cls):
        return [name for name, value in cls().items() if value is INPUT]
    @classmethod
    def output_names(cls):
        return [name for name, value in cls().items() if value is OUTPUT]

    def inputs(s):
        return {name: s[name] for name in s.input_names()}
    def outupts(s):
        return {name: s[name] for name in s.output_names()}
    def independants(s):
        return {name: s[name] for name in s
                if name in s.input_names()
                or name in s.output_names()}
    def dependants(s):
        return {name: s[name] for name in s
                if name not in s.input_names()
                and name not in s.output_names()}

    def has_all_inputs(s):
        return all(value is not INPUT for value in s.inputs())
    def has_all_outputs(s):
        return all(value is not INPUT for value in s.outputs())

    @property
    def asdict(s):
        return {k: v for k, v in s.__dict__.items() if not k.startswith("_")}
    def items(s):
        return s.asdict.items()
    def keys(s):
        return s.asdict.keys()
    def values(s):
        return s.asdict.values()
    def __len__(s):
        return len(s.asdict)
    def __iter__(s):
        return iter(s.asdict)
    def __getitem__(s, name):
        if not isinstance(name, str):
            raise TypeError(f"expected string name, got '{type(name).__name__}'")
        return s.asdict[name]
    def __repr__(s):
        width = 20
        lines = ["{"]
        for name, value in s.items():
            if value is INPUT:
                val = "<input>"
            elif value is OUTPUT:
                val = "<output>"
            elif value is ...:
                val = "-"
            elif isinstance(value, float):
                val = f"{value:.6g}"
            elif isinstance(value, np.ndarray):
                if len(value) < 4:
                    val = "[" + ", ".join(f"{v:.4g}" for v in value) + "]"
                else:
                    val = f"[{value[0]:.4g}, {value[1]:.4g}, ..., {value[-2]:.4g}, {value[-1]:.4g}]"
            else:
                val = repr(value)
            sep = ".." + "." * (width - 2 - len(name))
            lines.append(f"  {name} {sep} {val}")
        lines.append("}")
        return "\n".join(lines)


class Cylinder:
    @classmethod
    def solid(cls, length, diameter):
        return cls(length, 0, diameter)

    @classmethod
    def pipe(cls, length, *, inner_diameter=None, outer_diameter=None):
        return cls(length, inner_diameter, outer_diameter)

    def __init__(self, length, inner_diameter, outer_diameter):
        self.length = length
        self.inner_diameter = inner_diameter
        self.outer_diameter = outer_diameter

    @property
    def thickness(self):
        if self.inner_diameter is None:
            raise ValueError("unconstrained inner diameter")
        if self.outer_diameter is None:
            raise ValueError("unconstrained outer diameter")
        thickness = (self.outer_diameter - self.inner_diameter) / 2
        assert thickness >= 0
        return thickness
    @thickness.setter
    def thickness(self, new_thickness, keep_inner=None):
        if self.inner_diameter is None and self.outer_diameter is None:
            raise ValueError("unconstrained diameters")
        if keep_inner is None:
            keep_inner = self.inner_diameter is not None
        if keep_inner:
            if self.inner_diameter is None:
                raise ValueError("unconstrained inner diameter")
            self.outer_diameter = self.inner_diameter + 2 * new_thickness
        else:
            if self.outer_diameter is None:
                raise ValueError("unconstrained outer diameter")
            self.inner_diameter = self.outer_diameter - 2 * new_thickness

    def volume(self):
        if self.inner_diameter is None:
            raise ValueError("unconstrained inner diameter")
        if self.outer_diameter is None:
            raise ValueError("unconstrained outer diameter")
        b = self.outer_diameter
        a = self.inner_diameter
        return self.length * pi/4 * (b**2 - a**2)

    def surface_area(self, cutoff=None):
        if self.inner_diameter is None:
            raise ValueError("unconstrained inner diameter")
        if self.outer_diameter is None:
            raise ValueError("unconstrained outer diameter")
        if cutoff is None:
            cutoff = 1.0
        if cutoff < 0:
            cutoff = 0.0
        if cutoff > 1.0:
            cutoff = 1.0
        inner = cutoff * self.length * pi * self.inner_diameter
        outer = cutoff * self.length * pi * self.outer_diameter
        end = pi/4 * (self.outer_diameter**2 - self.inner_diameter**2)
        if cutoff == 0.0:
            end = 0.0
        if cutoff == 1.0:
            end *= 2
        return end + inner + outer

    def mass(self, density):
        return density * self.volume()

    def com(self, top):
        return top + self.length / 2

    def set_thickness_for_stress(self, max_pressure, yield_strength, sf=2):
        # Finding min thickness for hoop stress, using given safety factor and
        # thin walled approximation:
        #   thickness = max_pressure * inner_diameter / (2 * yield_strength)
        # If inner diameter is unconstrained:
        #   thickness = max_pressure * (outer_diameter - thickness) / (2 yield_strength)
        #                     max_pressure * outer_diameter
        #   => thickness = -----------------------------------
        #                   max_pressure + 2 * yield_strength
        if self.inner_diameter is None and self.outer_diameter is None:
            raise ValueError("unconstrained diameters")
        if self.inner_diameter is not None and self.outer_diameter is not None:
            raise ValueError("overconstrained diameters")
        two_ys = 2 * yield_strength
        max_pressure *= sf
        if self.inner_diameter is not None:
            self.thickness = max_pressure * self.inner_diameter / two_ys
        else:
            self.thickness = max_pressure * self.outer_diameter / (max_pressure + two_ys)



def simulate_burn(s):
    """
    Simulates the motor burn, with tank venting and combustion chamber
    combusting. Requires the dependant system parameters:
    - tank_wall_thickness
    - tank_wall_mass
    - cc_length
    - nozzle_exit_area
    """
    _start_time = time.time()

    # Fuel and ox objects.
    assert s.fuox is PARAFFIN_NOX
    fuel = s.fuox.fuel
    ox = s.fuox.ox

    # Coupla cylinders.
    fuel_cyl = Cylinder.pipe(s.fuel_length, outer_diameter=s.cc_diameter)
    fuel_cyl.thickness = s.fuel_initial_thickness
    tank_inner_diameter = s.rocket_diameter - s.tank_wall_thickness
    tank_cyl = Cylinder.solid(s.tank_inner_length, tank_inner_diameter)
    cc_cyl = Cylinder.solid(s.cc_length, s.cc_diameter)


    # legend (shoutout charle):
    #
    # X0 = initial
    # dX = time derivative
    # DX = discrete change
    #
    # X_l = ox liquid
    # X_v = ox vapour
    # X_o = ox anywhere
    # X_f = fuel
    # X_g = cc gases
    # X_n = new cc gases
    #
    # X_t = tank
    # X_c = cc
    # X_w = tank wall (for heat sink)
    # X_a = ambient
    #
    # X_inj = injector
    # X_u = upstream (tank-side of injector)
    # X_d = downstream (cc-side of injector)
    #
    # X_nzl = nozzle
    #
    # X_reg = regression (fuel erosion/vapourisation)

    # Various system constants.
    Dt = 0.001 # discrete calculus over time.
    DT = 0.02 # discrete calculus over temperature.
    negligible_mass = 0.001

    Mw_o = ox.prop("M")
    R_o = GAS_CONSTANT / Mw_o

    L_f = fuel_cyl.length
    rho_f = fuel.density

    V_t = tank_cyl.volume()

    D_c = cc_cyl.outer_diameter
    eta_c = s.cc_combustion_efficiency
    Vempty_c = cc_cyl.volume()

    c_w = s.tank_wall_specific_heat_capacity # note cp ~= cv for solids.
    C_w = s.tank_wall_mass * c_w

    T_a = s.ambient_temperature
    P_a = s.ambient_pressure
    rho_a = s.ambient_density
    Mw_a = s.ambient_molar_mass
    cp_a = s.ambient_constant_pressure_specific_heat_capacity

    Cd_inj = s.injector_discharge_coeff
    A_inj = s.injector_orifice_area

    Cd_nzl = s.nozzle_discharge_coeff
    A_nzl = s.nozzle_throat_area
    eps_nzl = s.nozzle_exit_area / s.nozzle_throat_area

    rr_a0 = s.fuox.a0
    rr_n = s.fuox.n


    # Cea object me.
    cea = CEA_Obj(propName="", oxName=ox.name, fuelName=fuel.name,
                  isp_units="sec",
                  cstar_units="m/s",
                  pressure_units="Pa",
                  temperature_units="K",
                  sonic_velocity_units="m/s",
                  enthalpy_units="J/kg",
                  density_units="kg/m^3",
                  specific_heat_units="J/kg-K")


    # Tracked state (all of which changes over time and is described
    # by differentials) is:
    # - T_t: tank temperature
    # - P_c: cc pressure
    # - D_f: fuel grain inner diameter
    # - m_l: tank liquid mass (happens to always be saturated)
    # - m_v: tank vapour mass (only saturated if liquid mass > negligible)
    # - m_g: cc gas mass
    # - T_g: cc gas temperature
    # - nmol_g: cc gas number of moles
    # - Cp_g: cc gas constant pressure heat capacity
    # This is enough to fully define the system at all times (when
    # combined with the constants).

    # Calculate initial state.

    # Assuming ox tank at ambient temperature and a saturated mixture.
    T0_t = T_a
    V0_l = V_t * s.ox_volume_fill_frac
    V0_v = V_t - V0_l
    m0_l = V0_l * ox.prop("D", "T", T0_t, "Q", 0)
    m0_v = V0_v * ox.prop("D", "T", T0_t, "Q", 1)

    # Inner diameter starts at full fuel grain.
    D0_f = fuel_cyl.inner_diameter

    # Combustion chamber initially filled with ambient properties.
    V0_c = Vempty_c - L_f * pi/4 * (D_c**2 - D0_f**2)
    m0_g = rho_a * V0_c
    nmol0_g = m0_g / Mw_a
    T0_g = T_a
    Cp0_g = m0_g * cp_a

    def df(T_t, m_l, m_v, D_f, m_g, nmol_g, T_g, Cp_g):
        """
        Returns the time derivatives of all input variables.
        """

        # Reconstruct some cc/fuel state.
        V_f = L_f * pi/4 * (D_c**2 - D_f**2)
        A_f = L_f * pi * D_f # inner fuel grain surface area.
        V_c = Vempty_c - V_f
        m_f = rho_f * V_f

        # Reconstruct some cc gas state.
        cp_g = Cp_g / m_g
        Mw_g = m_g / nmol_g
        R_g = GAS_CONSTANT / Mw_g
        y_g = cp_g / (cp_g - R_g)
        rho_g = m_g / V_c

        # Assuming combustion gases are ideal:
        #  P*V = m*R*T
        #  P = R*T * m/V
        P_c = R_g * T_g * rho_g



        # Properties determined by injector flow:
        dm_l = 0.0
        dm_v = 0.0
        dm_o = 0.0
        dT_t = 0.0

        # Liquid draining while there's any liquid in the tank.
        if m_l > negligible_mass:

            # Find injector flow rate.

            P_u = ox.prop("P", "T", T_t, "Q", 0) # tank at saturated pressure.
            P_d = P_c

            if P_u <= P_d:
                raise Exception("no injector pressure ratio")

            # Single-phase incompressible model (with Beta = 0):
            # (assuming upstream density as the "incompressible" density)
            rho_u = ox.prop("D", "P", P_u, "Q", 0)
            mdot_SPI = Cd_inj * A_inj * np.sqrt(2 * rho_u * (P_u - P_d))

            # Homogenous equilibrium model:
            # (assuming only saturated liquid leaving from upstream)
            s_u = ox.prop("S", "P", P_u, "Q", 0)
            s_d_l = ox.prop("S", "P", P_d, "Q", 0)
            s_d_v = ox.prop("S", "P", P_d, "Q", 1)
            x_d = (s_u - s_d_l) / (s_d_v - s_d_l)
            h_u = ox.prop("H", "P", P_u, "Q", 0)
            h_d = ox.prop("H", "P", P_d, "Q", x_d)
            rho_d = ox.prop("D", "P", P_d, "Q", x_d)
            mdot_HEM = Cd_inj * A_inj * rho_d * np.sqrt(2 * (h_u - h_d))

            # Generalised non-homogenous non-equilibrium model:
            # (assuming that P_sat is upstream saturation, and so is alaways
            #  =P_u since its saturated?????? this means that the dyer model
            #  is always just an arithmetic mean of spi and hem when the tank
            #  is saturated but hey maybe thats what we're looking for).
            #
            #  kappa = sqrt((P_u - P_d) / (P_sat - P_d))
            #  kappa = sqrt((P_u - P_d) / (P_u - P_d))
            #  kappa = sqrt(1) = 1
            kappa = 1
            k_NHNE = 1 / (1 + kappa)
            dm_inj = mdot_SPI * (1 - k_NHNE) + mdot_HEM * k_NHNE


            # To determine temperature and vapourised mass derivatives,
            # we're going to have to use: our brain.
            #  V = const.
            #  m_l / rho_l + m_v / rho_v = const.
            #  d/dt (m_l / rho_l + m_v / rho_v) = 0
            #  d/dt (m_l / rho_l) + d/dt (m_v / rho_v) = 0
            #  0 = (dm_l * rho_l - m_l * drho_l) / rho_l**2  [quotient rule]
            #    + (dm_v * rho_v - m_v * drho_v) / rho_v**2
            # dm_l = -dm_inj - dm_v  [injector and vapourisation]
            #  0 = ((-dm_inj - dm_v) * rho_l - m_l * drho_l) / rho_l**2
            #    + (dm_v * rho_v - m_v * drho_v) / rho_v**2
            #  0 = -dm_inj / rho_l
            #    - dm_v / rho_l
            #    - m_l * drho_l / rho_l**2
            #    + dm_v / rho_v
            #    - m_v * drho_v / rho_v**2
            #  0 = dm_v * (1/rho_v - 1/rho_l)
            #    - dm_inj / rho_l
            #    - m_l * drho_l / rho_l**2
            #    - m_v * drho_v / rho_v**2
            # drho = d/dt (rho) = d/dT (rho) * dT/dt  [chain rule]
            # drhodT = d/dT (rho)
            #  0 = dm_v * (1/rho_v - 1/rho_l)
            #    - dm_inj / rho_l
            #    - m_l * dT * drhodT_l / rho_l**2
            #    - m_v * dT * drhodT_v / rho_v**2
            #  dm_v = (dm_inj / rho_l
            #         + m_l * dT * drhodT_l / rho_l**2
            #         + m_v * dT * drhodT_v / rho_v**2
            #         ) / (1/rho_v - 1/rho_l)
            #  dm_v = dm_inj / rho_l / (1/rho_v - 1/rho_l)
            #       + dT / (1/rho_v - 1/rho_l) * (m_l * drhodT_l / rho_l**2
            #                                   + m_v * drhodT_v / rho_v**2)
            # let:
            #   foo = dm_inj / rho_l / (1/rho_v - 1/rho_l)
            #   bar = (m_l * drhodT_l / rho_l**2
            #        + m_v * drhodT_v / rho_v**2) / (1/rho_v - 1/rho_l)
            #  dm_v = foo + dT * bar
            # So, dm_v depends on dT, but also vice versa:
            #  d/dt (U) = -dm_inj * h_l  [first law of thermodynamics, adiabatic]
            #  d/dt (U_w + U_l + U_v) = -dm_inj * h_l
            #  d/dt (m_w*u_w) + d/dt (m_l*u_l) + d/dt (m_v*u_v) = -dm_inj * h_l
            #  -dm_inj * h_l = dm_w*u_w + m_w*du_w
            #                + dm_l*u_l + m_l*du_l
            #                + dm_v*u_v + m_v*du_v
            # dm_w = 0  [wall aint going anywhere]
            # dm_l = -dm_v - dm_inj  [same as earlier]
            #  -dm_inj * h_l = m_w*du_w + m_l*du_l + m_v*du_v
            #                + (-dm_v - dm_inj) * u_l
            #                + dm_v*u_v
            #  dm_inj * (u_l - h_l) = m_w*du_w + m_l*du_l + m_v*du_v
            #                       - dm_v*u_l
            #                       + dm_v*u_v
            #  dm_inj * (u_l - h_l) = m_w*du_w + m_l*du_l + m_v*du_v
            #                       + dm_v * (u_v - u_l)
            # du = d/dt (u) = d/dT (u) * dT/dt
            # also note:
            #   u = int (cv) dT
            #   d/dT (u) = cv
            # therefore:
            #   du = dT * cv
            #  dm_inj * (u_l - h_l) = dT * (m_w*cv_w + m_l*cv_l + m_v*cv_v)
            #                       + dm_v * (u_v - u_l)
            # let: Cv = m_w*cv_w + m_l*cv_l + m_v*cv_v
            #  dm_inj * (u_l - h_l) = dT * Cv + dm_v * (u_v - u_l)
            #  dT * Cv = dm_inj * (u_l - h_l) + dm_v * (u_l - u_v)
            # i think conceptually this makes sense as:
            #  internal energy change = boundary work + phase change energy
            # which checks out, so: bitta simul lets substitute
            #  dT * Cv = dm_inj * (u_l - h_l) + (foo + dT * bar) * (u_l - u_v)
            #  dT * Cv - dT * bar * (u_l - u_v) = dm_inj * (u_l - h_l) + foo * (u_l - u_v)
            #  dT = (dm_inj * (u_l - h_l) + foo * (u_l - u_v))
            #     / (Cv - bar * (u_l - u_v))
            # dandy.

            rho_l = ox.prop("D", "T", T_t, "Q", 0)
            rho_v = ox.prop("D", "T", T_t, "Q", 1)
            drhodT_l = (ox.prop("D", "T", T_t + DT, "Q", 0) - rho_l) / DT
            drhodT_v = (ox.prop("D", "T", T_t + DT, "Q", 1) - rho_v) / DT

            Cv_l = m_l * ox.prop("O", "T", T_t, "Q", 0)
            Cv_v = m_v * ox.prop("O", "T", T_t, "Q", 1)
            Cv = Cv_l + Cv_v + C_w

            u_l = ox.prop("U", "T", T_t, "Q", 0)
            u_v = ox.prop("U", "T", T_t, "Q", 1)
            h_l = ox.prop("H", "T", T_t, "Q", 0)

            foo = dm_inj / rho_l / (1/rho_v - 1/rho_l)
            bar = (m_l * drhodT_l / rho_l**2
                 + m_v * drhodT_v / rho_v**2) \
                / (1/rho_v - 1/rho_l)

            dT_t = (dm_inj * (u_l - h_l) + foo * (u_l - u_v)) \
                 / (Cv - bar * (u_l - u_v))

            dm_v = foo + dT_t * bar
            dm_l = -dm_inj - dm_v


        # Otherwise vapour draining.
        elif m_v > negligible_mass:
            dm_l = 0.0 # liquid mass is ignored hence fourth (big word init).

            # During this period, temperature and density are used to fully
            # define the state (density is simple due to fixed volume).
            rho_v = m_v / V_t

            # Due to numerical inaccuracy, might technically have the properties
            # of a saturated mixture so just pretend its a saturated vapour.
            rhosat_v = ox.prop("D", "T", T_t, "Q", 1)
            if rho_v >= rhosat_v:
                rho_v = rhosat_v


            # Find injector flow rate.

            P_u = ox.prop("P", "T", T_t, "D", rho_v)
            P_d = P_c

            if P_u <= P_d:
                raise Exception("no injector pressure ratio")

            # Technically gamma but use 'y' for file size reduction.
            y_u = ox.prop("C", "T", T_t, "D", rho_v) \
                / ox.prop("O", "T", T_t, "D", rho_v)
            # Use compressibility factor to account for non-ideal gas.
            Z_u = ox.prop("Z", "T", T_t, "D", rho_v)

            # Real compressible flow through an injector, with both
            # choked and unchoked possibilities:
            Pr_crit = (2 / (y_u + 1)) ** (y_u / (y_u - 1))
            Pr_rec = P_d / P_u
            if Pr_rec <= Pr_crit: # choked.
                Pterm = (2 / (y_u + 1)) ** ((y_u + 1) / (y_u - 1))
            else: # unchoked.
                Pterm = Pr_rec ** (2 / y_u) - Pr_rec ** ((y_u + 1) / y_u)
                Pterm *= 2 / (y_u - 1)
            dm_inj = Cd_inj * A_inj * P_u * np.sqrt(y_u / Z_u / R_o / T_t * Pterm)

            # Mass only leaves through injector, and no state change.
            dm_v = -dm_inj


            # Back to the well.
            #  d/dt (U) = -dm_inj * h  [first law of thermodynamics, adiabatic]
            #  d/dt (U_w + U) = -dm_inj * h  [no suffix is the non-saturated vapour in the tank]
            #  d/dt (m_w*u_w) + d/dt (m*u) = -dm_inj * h
            #  -dm_inj * h = dm_w*u_w + m_w*du_w
            #              + dm*u + m*du
            # dm_w = 0  [wall aint going anywhere]
            # dm = -dm_inj  [only mass change is from injector]
            #  -dm_inj * h = m_w * du_w
            #              - dm_inj * u
            #              + m * du
            #  dm_inj * (u - h) = m_w * du_w + m * du
            # du = dT * cv  [previously derived]
            #  dm_inj * (u - h) = dT * (m_w * cv_w + m * cv)
            # let: Cv = m_w * cv_w + m * cv
            #  dm_inj * (u - h) = dT * Cv
            # => dT = dm_inj * (u - h) / Cv
            # which makes sense, since only energy change is due to lost flow work.

            u_u = ox.prop("U", "T", T_t, "D", rho_v)
            h_u = ox.prop("H", "T", T_t, "D", rho_v)

            Cv = C_w + m_v * ox.prop("O", "T", T_t, "D", rho_v)

            dT_t = dm_inj * (u_u - h_u) / Cv


        # No oxidiser left, and nothing happens without oxidiser flow.
        else:
            raise Exception("no ox left")


        # Do fuel regression.
        dD_f = 0.0
        dV_f = 0.0
        dm_reg = 0.0
        if m_f > negligible_mass: # Gotta be fuel left.

            # Get oxidiser mass flux through the fuel grain.
            Gox = dm_inj * 4 / pi / D_f**2
            # Find regression rate from empirical parameters (and ox mass flux).
            rr_rdot = rr_a0 * Gox**rr_n

            # Fuel mass and diameter change from rdot:
            dD_f = 2 * rr_rdot
            dV_f = A_f * rr_rdot
            dm_reg = rho_f * dV_f

        # Fuel only leaves via regression.
        dm_f = -dm_reg


        # Model the nozzle as an injector, using ideal compressible
        # flow and both choked and unchoked possibilities:
        if P_c <= P_a and not (m_l > negligible_mass): # avoid exiting instantly
            raise Exception("no injector pressure ratio")
        Pr_crit = (2 / (y_g + 1)) ** (y_g / (y_g - 1))
        Pr_rec = P_a / P_c
        if Pr_rec <= Pr_crit: # choked.
            Pterm = (2 / (y_g + 1)) ** ((y_g + 1) / (y_g - 1))
        else: # unchoked.
            Pterm = Pr_rec ** (2 / y_g) - Pr_rec ** ((y_g + 1) / y_g)
            Pterm *= 2 / (y_g - 1)
        dm_out = Cd_nzl * A_nzl * P_c * np.sqrt(y_g / R_g / T_g * Pterm)


        # Gases in the chamber is just entering - exiting.
        dm_g = dm_inj + dm_reg - dm_out


        # Change in cc gas properties due to added gas.
        T_n = 0.0
        Mw_n = 0.0
        cp_n = 0.0
        dm_n = dm_reg + dm_inj # new gases is just fuel+ox.

        # Combustion occurs if there is both fuel and oxidiser.
        if dm_reg != 0 and dm_inj != 0:
            # Instantaneous oxidiser-fuel ratio.
            ofr = dm_inj / dm_reg

            # Do cea to find combustion properties.
            _, Cstar, T_n, Mw_n, y_comb = cea.get_IvacCstrTc_ChmMwGam(P_c, ofr, eps_nzl)
            Mw_n /= 1000 # stupid non-si return value.
            # Reconstruct cp from gamma, assuming ideal gas.
            #  y = cp / (cp - R)
            #  y = cp / (cp - Ru / Mw)
            #  y * (cp - Ru / Mw) = cp
            #  -y * Ru / Mw = cp * (1 - y)
            #  y * Ru / Mw = cp * (y - 1)
            # => cp = y * Ru / Mw / (y - 1)
            cp_n = y_comb * GAS_CONSTANT / Mw_n / (y_comb - 1)

        # Otherwise non-combusting oxidiser.
        elif dm_inj != 0:
            # No combustion but chamber gas changes due to oxidiser. Note this is
            # assuming isothermal mass transfer, so using tank temperature but
            # with current chamber pressure.
            T_n = T_t
            Mw_n = ox.prop("M")
            cp_n = ox.prop("C", "P", P_c, "T", T_t)

        else:
            assert dm_reg == 0.0

        # Change in any mass-specific property for a reservoir with
        # flow in and out:
        #  d/dt (m*p) = dm_in * p_in - dm_out * p

        # Change in moles:
        #  dn = d/dt (n_n) - d/dt (n_out)
        #  dn = dm_n / Mw_n - dm_out / Mw
        dnmol_g = dm_n / Mw_n - dm_out / Mw_g

        # Change in specific heat:
        dCp_g = dm_n * cp_n - dm_out * cp_g

        # Change in temperature:
        #  d/dt (m * cp * T) = dm_n * cp_n * T_n - dm_out * cp * T
        #  d/dt (m * cp * T) = dm_n * cp_n * T_n - dm_out * cp * T
        #  d/dt (m * cp) * T + m*cp * dT = dm_n * cp_n * T_n - dm_out * cp * T
        #  dCp * T + Cp * dT = dm_n * cp_n * T_n - dm_out * cp * T
        #  Cp * dT = dm_n * cp_n * T_n - dm_out * cp * T - dCp * T
        #  dT = (dm_n * cp_n * T_n - dm_out * cp * T - dCp * T) / Cp
        dT_g = (dm_n * cp_n * T_n - dm_out * cp_g * T_g - dCp_g * T_g) / Cp_g


        # Return derivatives in input order.
        return dT_t, dm_l, dm_v, dD_f, dm_g, dnmol_g, dT_g, dCp_g


    exced = False
    try:
        state = [
            [T0_t],
            [m0_l],
            [m0_v],
            [D0_f],
            [m0_g],
            [nmol0_g],
            [T0_g],
            [Cp0_g],
        ]
        # Simulate just for some time, TODO: figure out what is
        # considered the termination of the burn.
        _time = 0.0
        while _time < 80.0:
            _time += Dt

            # While i generally make a point not to use explicit
            # euler (its just not it), this system performs poorly
            # under other methods since it is not smooth. So,
            # explicit euler it is.
            dstate = df(*[v[-1] for v in state])
            assert len(dstate) == len(state)
            for dprop, prop in zip(dstate, state):
                Dprop = Dt * dprop
                prop.append(prop[-1] + Dprop)

    except Exception:
        exced = True
        traceback.print_exc()

    print(f"Finished burn sim in {time.time() - _start_time:.2f}s")


    T_t, m_l, m_v, D_f, m_g, nmol_g, T_g, Cp_g = state
    T_t = np.array(T_t)
    m_l = np.array(m_l)
    m_v = np.array(m_v)
    D_f = np.array(D_f)
    m_g = np.array(m_g)
    nmol_g = np.array(nmol_g)
    T_g = np.array(T_g)
    Cp_g = np.array(Cp_g)

    def diffarr(x):
        if len(x) == 1:
            return np.array([0.0])
        Dx = np.diff(x)
        Dx = np.append(Dx, Dx[-1])
        return Dx


    s.burn_time = (len(T_t) - 1) * Dt
    t = np.linspace(0, s.burn_time, len(T_t))
    mask = np.ones(len(t), dtype=bool)
    # mask = (t <= 0.1)


    # Reconstruct a bunch of dependant state.

    V_f = L_f * pi / 4 * (D_c**2 - D_f**2)
    V_c = Vempty_c - V_f

    m_f = rho_f * V_f

    Dm_inj = -diffarr(m_l + m_v)
    Dm_reg = -diffarr(m_f)
    Dm_g = diffarr(m_g)

    P_t = np.zeros(len(T_t), dtype=float)
    # saturated pressure:
    Pmask = (m_l > negligible_mass)
    Psat = [ox.prop("P", "T", T, "Q", 0) for T in T_t[Pmask]]
    P_t[Pmask] = Psat
    # not:
    Pmask = ~Pmask & (m_v > negligible_mass)
    Pnot = [ox.prop("P", "T", T, "D", m / V_t) for T, m in zip(T_t[Pmask], m_v[Pmask])]
    P_t[Pmask] = Pnot

    ofr = np.zeros(len(T_t), dtype=float)
    ofr_mask = (Dm_reg != 0)
    ofr[ofr_mask] = Dm_inj[ofr_mask] / Dm_reg[ofr_mask]
    ofr[~ofr_mask] = 0.0

    R_g = GAS_CONSTANT/(m_g / nmol_g)

    P_c = R_g * T_g * m_g / V_c


    dm_out = (Dm_inj + Dm_reg - Dm_g) / Dt

    cp_g = Cp_g / m_g
    y_g = cp_g / (cp_g - R_g)


    s.ox_mass_liquid = m_l
    s.ox_mass_vapour = m_v
    s.ox_mass = m_l + m_v
    s.fuel_mass = m_f
    s.tank_temperature = T_t
    s.tank_pressure = P_t
    s.cc_pressure = P_c

    plotme = [
        # data, title, ylabel, y_lower_limit_as_zero
        (s.tank_pressure, "Tank pressure", "Pressure [Pa]", False),
        (s.cc_pressure, "CC pressure", "Pressure [Pa]", False),
        (s.tank_temperature - 273.15, "Tank temperature", "Temperature [dC]", False),
        (ofr, "Oxidiser-fuel ratio", "Ratio [-]", False),
        (Dm_inj / Dt, "Injector mass flow rate", "Mass flow rate [kg/s]", True),
        (Dm_reg / Dt, "Regression mass flow rate", "Mass flow rate [kg/s]", True),
        (s.ox_mass_liquid + s.ox_mass_vapour, "Tank mass", "Mass [kg]", True),
        (s.fuel_mass, "Fuel mass", "Mass [kg]", True),
        # (s.ox_mass_liquid, "Tank liquid mass", "Mass [kg]", True),
        # (s.ox_mass_vapour, "Tank vapour mass", "Mass [kg]", True),
        (m_g, "Gas mass", "ceebs", False),
        (T_g, "Gas temp", "ceebs", False),
        (nmol_g, "Gas number of moles", "ceebs", False),
        (Cp_g / m_g, "Gas cp", "ceebs", False),
        (dm_out, "Gas exit", "ceebs", False),
        (y_g, "Gas gamma", "ceebs", False),
    ]
    def doplot(plotme):
        plt.figure()
        ynum = 2
        xnum = (len(plotme) + 1) // 2
        for i, elem in enumerate(plotme):
            if elem is ...:
                continue
            y, title, ylabel, snapzero = elem
            plt.subplot(ynum, xnum, 1 + i // ynum + xnum * (i % ynum))
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if len(y) == 1:
                if len(y) == len(t) - 1:
                    plt.plot(t[:-1], y, "o")
                else:
                    plt.plot(t, y, "o")
            elif len(y) > 0:
                if len(y) == len(t) - 1:
                    plt.plot(t[mask][:-1], y[mask[:-1]])
                else:
                    plt.plot(t[mask], y[mask])
            plt.title(title)
            plt.xlabel("Time [s]")
            plt.ylabel(ylabel)
            plt.grid()
            if snapzero and not exced:
                _, ymax = plt.ylim()
                plt.ylim(0, ymax)
        plt.subplots_adjust(left=0.05, right=0.97, wspace=0.4, hspace=0.3)
    doplot(plotme)
    plt.show()



def simulate_trajectory(s):
    """
    havent don it yet
    """
    pass



def cost(s):
    """
    Returns some float based on the given system state s.t. when the return is
    minimised, the system is optimised. Note that both inputs and outputs (aka
    all independant system parameters) must be set.
    """
    for name, value in s.dependants().items():
        if value is not ...:
            raise ValueError("expected all dependants unset, got: "
                    f"sys[{repr(name)}] = {repr(value)}")
    for name, value in s.independants().items():
        if value is INPUT or value is OUTPUT:
            raise ValueError("expected all independants set, got unset: "
                    f"sys[{repr(name)}]")

    assert s.fuox is PARAFFIN_NOX


    # Firstly get the easy masses/coms/length out of the way.

    top = -s.locked_length

    s.locked_com = top + s.locked_local_com
    top += s.locked_length


    # Find worst-case initial saturated tank pressure, which is the max tank pressure.
    tank_max_pressure = s.fuox.ox.prop("P", "T", s.ox_worstcase_temperature, "Q", 0)
    # Determine tank specs from the max pressure.
    tank_wall_cyl = Cylinder.pipe(s.tank_inner_length, outer_diameter=s.rocket_diameter)
    tank_wall_cyl.set_thickness_for_stress(tank_max_pressure,
                                           s.tank_wall_yield_strength,
                                           sf=3.5)
    # TODO: establish exact tank mass, for now just assume thick ends.
    tank_wall_end_cyl = Cylinder.solid(2 * tank_wall_cyl.thickness,
                                       tank_wall_cyl.outer_diameter)

    s.tank_length = s.tank_inner_length + 2 * tank_wall_end_cyl.length
    s.tank_com = top + s.tank_length / 2
    s.tank_wall_thickness = tank_wall_cyl.thickness
    s.tank_wall_mass = tank_wall_cyl.mass(s.tank_wall_density) \
                     + 2 * tank_wall_end_cyl.mass(s.tank_wall_density)
    top += s.tank_length

    s.mov_com = top + s.mov_local_com
    top += s.mov_length

    s.injector_com = top + s.injector_local_com
    top += s.injector_length

    # Using rule-of-thumb pre- and post-cc lengths:
    s.cc_pre_length = s.cc_diameter
    s.cc_post_length = 1.5 * s.cc_diameter
    s.cc_length = s.cc_pre_length + s.fuel_length + s.cc_post_length

    # TODO: nozzle specs
    s.nozzle_exit_area = 20 * s.nozzle_throat_area
    s.nozzle_length = 0.10
    s.nozzle_com = top + 0.05
    s.nozzle_mass = 2


    # Burn me.
    simulate_burn(s)


    # Now do cc walls for cc max pressure.
    cc_wall_cyl = Cylinder.pipe(s.cc_length, inner_diameter=s.cc_diameter)
    cc_wall_cyl.set_thickness_for_stress(s.cc_pressure.max(),
                                         s.cc_wall_yield_strength,
                                         sf=3)
    s.cc_wall_thickness = cc_wall_cyl.thickness
    s.cc_wall_mass = cc_wall_cyl.mass(s.cc_wall_density)
    s.cc_wall_com = cc_wall_cyl.com(top)

    s.fuel_com = top + s.cc_pre_length + s.fuel_length / 2
    top += s.cc_length


    # All rocket components defined, cook me the full thing.
    s.rocket_length = (
        s.locked_length +
        s.tank_length +
        s.mov_length +
        s.injector_length +
        s.cc_length +
        s.nozzle_length
    )
    s.rocket_mass = (
        s.locked_mass +
        s.tank_wall_mass +
        s.ox_mass +
        s.mov_mass +
        s.injector_mass +
        s.cc_wall_mass +
        s.fuel_mass +
        s.nozzle_mass
    )
    s.rocket_com = (
        s.locked_com * s.locked_mass +
        s.tank_com * (s.tank_wall_mass + s.ox_mass) +
        s.mov_com * s.mov_mass +
        s.injector_com * s.injector_mass +
        s.cc_wall_com * s.cc_wall_mass +
        s.fuel_com * s.fuel_mass +
        s.nozzle_com * s.nozzle_mass
    ) / s.rocket_mass


    # Launch me.
    simulate_trajectory(s)

    print(s)
