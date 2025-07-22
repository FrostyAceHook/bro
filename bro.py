import traceback
import time

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from rocketcea import cea_obj
from rocketcea.cea_obj_w_units import CEA_Obj

GAS_CONSTANT = 8.31446261815324 # [J/K/mol]



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




class Propellant:
    TEMPERATURE = 293.15

    def __init__(self, name, kind, composition, hf, rho):
        """
        name: unique string id.
        kind: startswith 'fu' or 'ox'.
        composition: string of chemical formula, of the form "X0 N0 X1 N1 ..." where
                     Xn is atomic symbol and Nn is number of atoms.
        hf: enthalpy of formation, in kcal/mol.
        rho: density at `Propellant.TEMPERATURE`, in kg/m^3.
        """
        self.name = name
        self.density = rho

        # use the same kind detection as nasa cea fortran code.
        if kind[:2] == "fu":
            kind = "fu"
            adder = cea_obj.add_new_fuel
        elif kind[:2] == "ox":
            kind = "ox"
            adder = cea_obj.add_new_oxidizer
        else:
            assert False, "invalid kind"
        adder(name,
            f"{kind} {name} {composition}\n"
            f"h,kc={hf:.1f}\n"
            f"t,k={self.TEMPERATURE}\n"
            f"rho,kg={rho:.1f}\n"
        )

    def prop(self, *args):
        """
        Forwards to PropsSI for this oxidiser.
        """
        return PropsSI(*args, self.name)

# Formulas and enthalpy of formation from Hybrid Rocket Propulsion Handbook, Karp & Jens.
PARAFFIN = Propellant("PARAFFIN", "fuel", "C 32 H 66",
        hf=-224.2, rho=924.5)
N2O = Propellant("N2O", "oxidiser", "N 2 O 1",
        hf=15.5, rho=PropsSI("D", "T", Propellant.TEMPERATURE, "Q", 0, "N2O"))




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
        # All dependant variables initially set to ellipsis. Note that the
        # choice of "basis" variables (the independant variables that are
        # either input or output) is arbitrary, but some make significantly
        # more sense than others (i.e. the relationship is difficult to
        # inverse). Also note that the decision between assigning an
        # independant variable as input and output is generally arbitrary
        # (so long as controlled by us and not the weather or contest for
        # example), and the only real difference is that output variables
        # are swept over a range and the optimal choice is kept.

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

        s.ox_type = INPUT # must be `N2O`
        s.ox_volume_fill_frac = INPUT # [-]
        s.ox_mass = ... # [kg, over time]
        s.ox_mass_liquid = ... # [kg, over time]
        s.ox_mass_vapour = ... # [kg, over time]
        s.ox_com = ... # [m]

        s.mov_mass = INPUT # [kg]
        s.mov_length = INPUT # [m]
        s.mov_local_com = INPUT # [m]
        s.mov_com = ... # [m]

        s.injector_mass = INPUT # [kg]
        s.injector_length = INPUT # [m]
        s.injector_local_com = INPUT # [m]
        s.injector_com = ... # [m]
        s.injector_discharge_coeff = INPUT # [-]
        s.injector_orifice_area = OUTPUT # [m^2]
        s.injector_initial_pressure_ratio = INPUT # [-]

        s.cc_diameter = OUTPUT # [m]
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

        s.fuel_type = INPUT # must be `PARAFFIN`
        s.fuel_length = OUTPUT # [m]
        s.fuel_initial_thickness = OUTPUT # [m]
        s.fuel_mass = ... # [kg, over time]
        s.fuel_com = ... # [m]

        s.nozzle_exit_area = OUTPUT # [m^2]
        s.nozzle_throat_area = ... # [m^2]
        s.nozzle_length = ... # [m]
        s.nozzle_local_com = ... # [m]
        s.nozzle_com = ... # [m]
        s.nozzle_mass = ... # [kg]
        s.nozzle_thrust_efficiency = INPUT # [-]
        s.nozzle_thrust = ... # [N, over time]

        s.rocket_target_apogee = INPUT # [m]
        s.rocket_diameter = INPUT # [m]
        s.rocket_length = ... # [m]
        s.rocket_dry_mass = ... # [kg]
        s.rocket_mass = ... # [kg, over time]
        s.rocket_com = ... # [m, over time]
        s.rocket_drag_coeff = ... # [idk]
        s.rocket_stability = INPUT # [-]
        s.rocket_net_force = ... # [N, over time]
        s.rocket_altitude = ... # [m, over time]

        s.ambient_temperature = INPUT # [K]
        s.ambient_pressure = INPUT # [Pa]
        s.tank_worstcase_temperature = INPUT # [K]
        s.regression_rate_coeff = INPUT # [-]
        s.regression_rate_exp = INPUT # [.]
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
        return self.length * np.pi/4 * (b**2 - a**2)

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
        inner = cutoff * self.length * np.pi * self.inner_diameter
        outer = cutoff * self.length * np.pi * self.outer_diameter
        end = np.pi/4 * (self.outer_diameter**2 - self.inner_diameter**2)
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
    combusting. Requires the tank specs to be set in the system.
    """
    _start_time = time.time()

    assert s.ox_type is N2O
    assert s.fuel_type is PARAFFIN
    ox = s.ox_type
    fuel = s.fuel_type

    # Coupla cylinders.
    fuel_cyl = Cylinder.pipe(s.fuel_length, outer_diameter=s.cc_diameter)
    fuel_cyl.thickness = s.fuel_initial_thickness
    tank_inner_diameter = s.rocket_diameter - s.tank_wall_thickness
    tank_cyl = Cylinder.solid(s.tank_inner_length, tank_inner_diameter)


    # legend (shoutout charle):
    #
    # X0 = initial
    # dX = time derivative
    # DX = discrete change
    #
    # X_l = ox liquid
    # X_v = ox vapour
    # X_o = ox (in cc)
    # X_f = fuel
    #
    # X_t = tank
    # X_c = cc
    # X_w = tank wall (for heat sink)
    #
    # X_inj = injector
    # X_u = upstream (tank-side of injector)
    # X_d = downstream (cc-side of injector)

    # Various system constants.
    Dt = 0.03 # discrete calculus over time.
    DT = 0.08 # discrete calculus over temperature.
    negligible_mass = 0.005
    Rox = GAS_CONSTANT / ox.prop("M")
    Cd_inj = s.injector_discharge_coeff
    A_inj = s.injector_orifice_area
    rho_f = fuel.density
    L_f = fuel_cyl.length
    OD_f = fuel_cyl.outer_diameter
    V_t = tank_cyl.volume()
    c_w = s.tank_wall_specific_heat_capacity # note cp ~= cv for solids.
    C_w = s.tank_wall_mass * c_w

    # Regression rate parameters.
    rr_a0 = s.regression_rate_coeff
    rr_n = s.regression_rate_exp


    # Determine initial properties.
    T0_t = s.ambient_temperature
    V0_l = V_t * s.ox_volume_fill_frac
    V0_v = V_t - V0_l
    m0_l = V0_l * ox.prop("D", "T", T0_t, "Q", 0)
    m0_v = V0_v * ox.prop("D", "T", T0_t, "Q", 1)
    D0_f = fuel_cyl.inner_diameter

    # TODO: initial cc pressure
    P0_c = s.injector_initial_pressure_ratio * ox.prop("P", "T", T0_t, "Q", 0)



    @singleton
    class debugme:
        def __init__(self):
            self.state = {
            }
        def __setitem__(self, name, value):
            self.state[name][0].append(value)
    plot_debugme = False


    # Tracked state (all of which changes over time and is described
    # by differentials) is:
    # - m_l: tank liquid mass (happens to always be saturated)
    # - m_v: tank vapour mass (only saturated if liquid mass > negligible)
    # - T_t: tank temperature
    # - P_c: cc pressure
    # - D_f: fuel grain inner diameter
    # This is enough to fully define the system at all times (when
    # combined with the constants).


    def df(m_l, m_v, T_t, P_c, D_f):
        """
        Returns the time derivatives of all input variables.
        """

        # Properties determined by injector:
        dm_l = 0.0
        dm_v = 0.0
        dm_o = 0.0
        dT_t = 0.0

        # Liquid draining while there's any liquid in the tank.
        if m_l > negligible_mass:

            # Find injector flow rate.

            P_u = ox.prop("P", "T", T_t, "Q", 0) # tank at saturated pressure.
            P_d = P_c

            if P_u / P_d <= 1:
                raise Exception("injector pressure ratio too small")

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
            dm_o = mdot_SPI * (1 - k_NHNE) + mdot_HEM * k_NHNE


            # To determine temperature and vapourised mass derivatives,
            # we're going to have to use: our brain.
            #  V = const.
            #  m_l / rho_l + m_v / rho_v = const.
            #  d/dt (m_l / rho_l + m_v / rho_v) = 0
            #  d/dt (m_l / rho_l) + d/dt (m_v / rho_v) = 0
            #  0 = (dm_l * rho_l - m_l * drho_l) / rho_l**2  [quotient rule]
            #    + (dm_v * rho_v - m_v * drho_v) / rho_v**2
            # dm_l = -dm_o - dm_v  [injector and vapourisation]
            #  0 = ((-dm_o - dm_v) * rho_l - m_l * drho_l) / rho_l**2
            #    + (dm_v * rho_v - m_v * drho_v) / rho_v**2
            #  0 = -dm_o / rho_l
            #    - dm_v / rho_l
            #    - m_l * drho_l / rho_l**2
            #    + dm_v / rho_v
            #    - m_v * drho_v / rho_v**2
            #  0 = dm_v * (1/rho_v - 1/rho_l)
            #    - dm_o / rho_l
            #    - m_l * drho_l / rho_l**2
            #    - m_v * drho_v / rho_v**2
            # drho = d/dt (rho) = d/dT (rho) * dT/dt  [chain rule]
            # drhodT = d/dT (rho)
            #  0 = dm_v * (1/rho_v - 1/rho_l)
            #    - dm_o / rho_l
            #    - m_l * dT * drhodT_l / rho_l**2
            #    - m_v * dT * drhodT_v / rho_v**2
            #  dm_v = (dm_o / rho_l
            #         + m_l * dT * drhodT_l / rho_l**2
            #         + m_v * dT * drhodT_v / rho_v**2
            #         ) / (1/rho_v - 1/rho_l)
            #  dm_v = dm_o / rho_l / (1/rho_v - 1/rho_l)
            #       + dT / (1/rho_v - 1/rho_l) * (m_l * drhodT_l / rho_l**2
            #                                   + m_v * drhodT_v / rho_v**2)
            # let:
            #   foo = dm_o / rho_l / (1/rho_v - 1/rho_l)
            #   bar = (m_l * drhodT_l / rho_l**2
            #        + m_v * drhodT_v / rho_v**2) / (1/rho_v - 1/rho_l)
            #  dm_v = foo + dT * bar
            # So, dm_v depends on dT, but also vice versa:
            #  d/dt (U) = -dm_o * h_l  [first law of thermodynamics, adiabatic]
            #  d/dt (U_w + U_l + U_v) = -dm_o * h_l
            #  d/dt (m_w*u_w) + d/dt (m_l*u_l) + d/dt (m_v*u_v) = -dm_o * h_l
            #  -dm_o * h_l = dm_w*u_w + m_w*du_w
            #              + dm_l*u_l + m_l*du_l
            #              + dm_v*u_v + m_v*du_v
            # dm_w = 0  [wall aint going anywhere]
            # dm_l = -dm_v - dm_o  [same as earlier]
            #  -dm_o * h_l = m_w*du_w + m_l*du_l + m_v*du_v
            #              + (-dm_v - dm_o) * u_l
            #              + dm_v*u_v
            #  dm_o * (u_l - h_l) = m_w*du_w + m_l*du_l + m_v*du_v
            #                     - dm_v*u_l
            #                     + dm_v*u_v
            #  dm_o * (u_l - h_l) = m_w*du_w + m_l*du_l + m_v*du_v
            #                     + dm_v * (u_v - u_l)
            # du = d/dt (u) = d/dT (u) * dT/dt
            # also note:
            #   u = int (cv) dT
            #   d/dT (u) = cv
            # therefore:
            #   du = dT * cv
            #  dm_o * (u_l - h_l) = dT * (m_w*cv_w + m_l*cv_l + m_v*cv_v)
            #                     + dm_v * (u_v - u_l)
            # let: Cv = m_w*cv_w + m_l*cv_l + m_v*cv_v
            #  dm_o * (u_l - h_l) = dT * Cv + dm_v * (u_v - u_l)
            #  dT * Cv = dm_o * (u_l - h_l) + dm_v * (u_l - u_v)
            # i think conceptually this makes sense as:
            #  internal energy change = boundary work + phase change energy
            # which checks out, so: bitta simul lets substitute
            #  dT * Cv = dm_o * (u_l - h_l) + (foo + dT * bar) * (u_l - u_v)
            #  dT * Cv - dT * bar * (u_l - u_v) = dm_o * (u_l - h_l) + foo * (u_l - u_v)
            #  dT = (dm_o * (u_l - h_l) + foo * (u_l - u_v))
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

            foo = dm_o / rho_l / (1/rho_v - 1/rho_l)
            bar = (m_l * drhodT_l / rho_l**2
                 + m_v * drhodT_v / rho_v**2) \
                / (1/rho_v - 1/rho_l)

            dT_t = (dm_o * (u_l - h_l) + foo * (u_l - u_v)) \
                 / (Cv - bar * (u_l - u_v))

            dm_v = foo + dT_t * bar
            dm_l = -dm_o - dm_v


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

            if P_u / P_d <= 1:
                raise Exception("injector pressure ratio too small")

            # Technically gamma but use 'y' for file size reduction.
            y_u = ox.prop("C", "T", T_t, "D", rho_v) \
                / ox.prop("O", "T", T_t, "D", rho_v)
            # Use compressibility factor to account for non-ideal gas.
            Z_u = ox.prop("Z", "T", T_t, "D", rho_v)

            # General compressible flow through an injector, with both
            # choked and unchoked possibilities:
            Pr_crit = (2 / (y_u + 1)) ** (y_u / (y_u - 1))
            Pr_rec = P_d / P_u
            if Pr_rec <= Pr_crit: # choked.
                Pterm = (2 / (y_u + 1)) ** ((y_u + 1) / (y_u - 1))
            else: # unchoked.
                Pterm = Pr_rec ** (2 / y_u) - Pr_rec ** ((y_u + 1) / y_u)
                Pterm *= 2 / (y_u - 1)
            dm_o = Cd_inj * A_inj * P_u * np.sqrt(y_u / Z_u / Rox / T_t * Pterm)

            # Mass only leaves through injector, and no state change.
            dm_v = -dm_o


            # Back to the well.
            #  d/dt (U) = -dm_o * h  [first law of thermodynamics, adiabatic]
            #  d/dt (U_w + U) = -dm_o * h  [no suffix is the non-saturated vapour in the tank]
            #  d/dt (m_w*u_w) + d/dt (m*u) = -dm_o * h
            #  -dm_o * h = dm_w*u_w + m_w*du_w
            #            + dm*u + m*du
            # dm_w = 0  [wall aint going anywhere]
            # dm = -dm_o  [only mass change is from injector]
            #  -dm_o * h = m_w * du_w
            #            - dm_o * u
            #            + m * du
            #  dm_o * (u - h) = m_w * du_w + m * du
            # du = dT * cv  [previously derived]
            #  dm_o * (u - h) = dT * (m_w * cv_w + m * cv)
            # let: Cv = m_w * cv_w + m * cv
            #  dm_o * (u - h) = dT * Cv
            # => dT = dm_o * (u - h) / Cv
            # which makes sense, since only energy change is due to lost flow work.

            u_u = ox.prop("U", "T", T_t, "D", rho_v)
            h_u = ox.prop("H", "T", T_t, "D", rho_v)

            Cv = C_w + m_v * ox.prop("O", "T", T_t, "D", rho_v)

            dT_t = dm_o * (u_u - h_u) / Cv


        # No oxidiser left, and nothing happens without oxidiser flow.
        else:
            raise Exception("no ox left")


        # Properties determined by fuel regression.
        dD_f = 0.0

        # Gotta be fuel left.
        m_f = rho_f * np.pi/4 * (OD_f**2 - D_f**2)
        if m_f > negligible_mass:

            # Do fuel regression calcs.

            # Get oxidiser mass flux through the fuel grain.
            Gox = dm_o * 4 / np.pi / D_f**2
            # Find regression rate from empirical parameters (and ox mass flux).
            rr_rdot = rr_a0 * Gox**rr_n

            # Fuel mass and diameter change from rdot:
            dD_f = 2 * rr_rdot
            dm_f = rr_rdot * rho_f * np.pi * D_f * L_f


            # Oxidiser-fuel ratio.
            ofr = dm_o / dm_f


        # Properties determined by combustion dynamics.
        dP_c = 0.0
        if dm_o != 0 and dm_f != 0:
            # TODO: cc pressure
            pass


        return dm_l, dm_v, dT_t, dP_c, dD_f


    try:
        state = [
            [m0_l],
            [m0_v],
            [T0_t],
            [P0_c],
            [D0_f],
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
        traceback.print_exc()

    print(f"Finished burn sim in {time.time() - _start_time:.2f}s")


    m_l, m_v, T_t, P_c, D_f = state
    m_l = np.array(m_l)
    m_v = np.array(m_v)
    T_t = np.array(T_t)
    P_c = np.array(P_c)
    D_f = np.array(D_f)

    def diffarr(x):
        if len(x) == 1:
            return np.array([0.0])
        Dx = np.diff(x)
        Dx = np.append(Dx, Dx[-1])
        return Dx


    # Reconstruct pressure over time.
    P_t = np.zeros(len(T_t), dtype=float)
    # saturated pressure:
    Pmask = (m_l > negligible_mass)
    Psat = [ox.prop("P", "T", T, "Q", 0) for T in T_t[Pmask]]
    P_t[Pmask] = Psat
    # not:
    Pmask = ~Pmask & (m_v > negligible_mass)
    Pnot = [ox.prop("P", "T", T, "D", m / V_t) for T, m in zip(T_t[Pmask], m_v[Pmask])]
    P_t[Pmask] = Pnot

    # Reconstruct fuel mass over time.
    m_f = rho_f * L_f * np.pi / 4 * (OD_f**2 - D_f**2)

    # Reconstruct ofr over time.
    Dm_o = diffarr(m_l + m_v)
    Dm_f = diffarr(m_f)
    ofr = np.zeros(len(T_t), dtype=float)
    ofr[Dm_f != 0] = Dm_o[Dm_f != 0] / Dm_f[Dm_f != 0]
    ofr[~(Dm_f != 0)] = 0.0

    s.burn_time = (len(m_l) - 1) * Dt
    t = np.linspace(0, s.burn_time, len(m_l))
    mask = np.ones(len(t), dtype=bool)
    # mask = (np.arange(len(t)) >= int(0.85 * len(t)))

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
        (-Dm_o / Dt, "Ox exit mass flow rate", "Mass flow rate [kg/s]", True),
        (-Dm_f / Dt, "Fuel exit mass flow rate", "Mass flow rate [kg/s]", True),
        (s.ox_mass_liquid + s.ox_mass_vapour, "Tank mass", "Mass [kg]", True),
        (s.fuel_mass, "Fuel mass", "Mass [kg]", True),
        (s.ox_mass_liquid, "Tank liquid mass", "Mass [kg]", True),
        (s.ox_mass_vapour, "Tank vapour mass", "Mass [kg]", True),
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
            if snapzero:
                _, ymax = plt.ylim()
                plt.ylim(0, ymax)
        plt.subplots_adjust(left=0.05, right=0.97, wspace=0.4, hspace=0.3)
    doplot(plotme)
    if plot_debugme:
        doplot([(v[0], k, v[1], False) for k, v in debugme.state.items()])
    plt.show()




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

    assert s.ox_type is N2O
    assert s.fuel_type is PARAFFIN


    # Firstly get the easy masses/coms/length out of the way.

    top = -s.locked_length

    s.locked_com = top + s.locked_local_com
    top += s.locked_length


    # Find worst-case initial saturated tank pressure, which is the max tank pressure.
    tank_max_pressure = s.ox_type.prop("P", "T", s.tank_worstcase_temperature, "Q", 0)
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

    # Burn me.
    simulate_burn(s)


    s.mov_com = top + s.mov_local_com
    top += s.mov_length

    s.injector_com = top + s.injector_local_com
    top += s.injector_length

    # Using rule-of-thumb pre- and post-cc lengths:
    s.cc_pre_length = s.cc_diameter
    s.cc_post_length = 1.5 * s.cc_diameter
    s.cc_length = s.cc_pre_length + s.fuel_length + s.cc_post_length
    cc_wall_cyl = Cylinder.pipe(s.cc_length, inner_diameter=s.cc_diameter)
    cc_wall_cyl.set_thickness_for_stress(s.cc_pressure.max(), s.cc_wall_yield_strength)
    s.cc_wall_thickness = cc_wall_cyl.thickness
    s.cc_wall_mass = cc_wall_cyl.mass(s.cc_wall_density)
    s.cc_wall_com = cc_wall_cyl.com(top)

    s.fuel_com = top + s.cc_pre_length + s.fuel_length / 2
    top += s.cc_length

    # TODO: nozzle specs
    s.nozzle_length = 0.10
    s.nozzle_com = top + 0.05
    s.nozzle_mass = 2

    s.rocket_length = (
        s.locked_length +
        s.tank_length +
        s.mov_length +
        s.injector_length +
        s.cc_length +
        s.nozzle_length
    )
    s.rocket_dry_mass = (
        s.locked_mass +
        s.tank_wall_mass +
        s.mov_mass +
        s.injector_mass +
        s.cc_wall_mass +
        s.nozzle_mass
    )
    s.rocket_mass = s.rocket_dry_mass + s.ox_mass + s.fuel_mass
    s.rocket_com = (
        s.locked_com * s.locked_mass +
        s.tank_com * (s.tank_wall_mass + s.ox_mass) +
        s.mov_com * s.mov_mass +
        s.injector_com * s.injector_mass +
        s.cc_wall_com * s.cc_wall_mass + s.fuel_com * s.fuel_mass +
        s.nozzle_com * s.nozzle_mass
    ) / s.rocket_mass

    print(s)



def main():
    sys = Sys()

    sys.locked_mass = 5.0 # [kg]
    sys.locked_length = 2.0 # [m]
    sys.locked_local_com = 1.3 # downwards from top, [m]

    sys.tank_inner_length = 0.55 # [m], OUTPUT
    sys.tank_wall_density = 2720.0 # Al6061, [kg/m^3]
    sys.tank_wall_yield_strength = 241e6 # Al6061, [Pa]
    sys.tank_wall_specific_heat_capacity = 896 # Al6061, [J/kg/K]

    sys.ox_type = N2O # required
    sys.ox_volume_fill_frac = 0.8 # [-]

    sys.mov_mass = 0.5 # [kg]
    sys.mov_length = 0.1 # [m]
    sys.mov_local_com = sys.mov_length / 2 # [m]

    sys.injector_mass = 0.5 # [kg]
    sys.injector_length = 0.02 # [m]
    sys.injector_local_com = sys.injector_length / 2 # [m]
    sys.injector_discharge_coeff = 0.9 # [-]
    sys.injector_orifice_area = 40 * np.pi/4 * 0.5e-3**2 # [m^2], OUTPUT
    sys.injector_initial_pressure_ratio = 0.3 # [-]

    sys.cc_diameter = 0.100 # [m], OUTPUT
    sys.cc_wall_density = 2720.0 # Al6061, [kg/m^3]
    sys.cc_wall_yield_strength = 241e6 # Al6061, [Pa]

    sys.fuel_type = PARAFFIN # required.
    sys.fuel_length = 0.15 # [m], OUTPUT
    sys.fuel_initial_thickness = 0.03 # [m], OUTPUT

    sys.nozzle_exit_area = 0.01 # [m^2], OUTPUT
    sys.nozzle_thrust_efficiency = 0.9 # [-]

    sys.rocket_target_apogee = 30000 / 3.281 # [m]
    sys.rocket_diameter = 0.145 # [m]
    sys.rocket_stability = 1.5 # [-?]

    sys.ambient_temperature = 25 + 273.15 # [K]
    sys.ambient_pressure = 100e3 # [Pa]
    sys.tank_worstcase_temperature = 35 + 273.15 # 36.4dC is critical temp of N2O, [K]
    # Regression rate data for paraffin and n2o, from:
    # Hybrid Rocket Propulsion Handbook, Karp & Jens.
    sys.regression_rate_coeff = 1.55e-4
    sys.regression_rate_exp = 0.5

    cost(sys)

if __name__ == "__main__":
    main()
