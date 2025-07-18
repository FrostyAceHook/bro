import traceback

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from scipy.optimize import root_scalar


def singleton(cls):
    """
    Returns a single instantiation of the given class, and prevents
    further creation of the class.
    - class decorator.
    """
    instance = cls()
    def throw(cls, *args, **kwargs):
        raise TypeError("cannot create another instance of singleton "
                f"{repr(cls.__name__)}")
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

        s.tank_length = OUTPUT # [m]
        s.tank_com = ... # [m]
        s.tank_wall_density = INPUT # [kg/m^3]
        s.tank_wall_yield_strength = INPUT # [Pa]
        s.tank_wall_specific_heat_capacity = INPUT # [J/kg/K]
        s.tank_wall_thickness = ... # [m]
        s.tank_wall_mass = ... # [kg]
        s.tank_volume = ... # [m^3]
        s.tank_temperature = ... # [K, over time]
        s.tank_pressure = ... # [Pa, over time]

        s.ox_type = INPUT # must be "N2O"
        s.ox_volume_fill_frac = INPUT # [-]
        s.ox_initial_mass = ... # [kg]
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
        s.injector_mass_flow_rate = ... # [kg/s, over time]

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

        s.fuel_type = INPUT # must be "paraffin"
        s.fuel_length = OUTPUT # [m]
        s.fuel_thickness = OUTPUT # [m]
        s.fuel_density = ... # [kg/m^3]
        s.fuel_initial_mass = ... # [kg]
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

        s.environment_temperature = INPUT # [K]
        s.burn_time = ... # [s]
        s.integration_dt = INPUT # [s]

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



def simulate_burn(s, top):
    """
    Simulates the motor burn, with tank venting and combustion chamber
    combusting. Also does the tank properties like mass and such.
    """
    assert s.ox_type == "N2O"

    # legend (shoutout charle):
    # X0 = initial
    # X_l = liquid
    # X_v = vapour
    # X_u = upstream (tank)
    # X_d = downstream (cc)
    # dX = time derivative
    # DX = discrete change

    Dt = s.integration_dt
    DT = Dt * 1.0 # use for numerical derivatives over temperature.
    Cd = s.injector_discharge_coeff
    Ainj = s.injector_orifice_area
    T0_u = s.environment_temperature
    P0_u = PropsSI("P", "T", T0_u, "Q", 0, "N2O")


    # Now that we have pressure, we can establish the tank walls.
    tank_wall_cyl = Cylinder.pipe(s.tank_length, outer_diameter=s.rocket_diameter)
    tank_wall_cyl.set_thickness_for_stress(P0_u, s.tank_wall_yield_strength)
    tank_wall_end_cyl = Cylinder.solid(tank_wall_cyl.thickness,
                                       tank_wall_cyl.outer_diameter)
    tank_cyl = Cylinder.solid(s.tank_length, tank_wall_cyl.inner_diameter)

    s.tank_com = tank_wall_cyl.thickness + tank_wall_cyl.com(top)
    s.tank_wall_thickness = tank_wall_cyl.thickness
    s.tank_wall_mass = tank_wall_cyl.mass(s.tank_wall_density) \
                     + 2 * tank_wall_end_cyl.mass(s.tank_wall_density)
    new_top = top + s.tank_length + 2 * tank_wall_cyl.thickness

    # Now that we have the tank fully defined, determine initial ox levels.
    V_u = tank_cyl.volume()
    V0_l_u = V_u * s.ox_volume_fill_frac
    V0_v_u = V_u * (1 - s.ox_volume_fill_frac)
    m0_l_u = V0_l_u * PropsSI("D", "T", T0_u, "Q", 0, "N2O")
    m0_v_u = V0_v_u * PropsSI("D", "T", T0_u, "Q", 1, "N2O")
    x0_u = m0_v_u / (m0_l_u + m0_v_u)
    # Heat capacity of tank walls.
    Cpwall = s.tank_wall_specific_heat_capacity * s.tank_wall_mass

    # Justa coupel more sys properties.
    s.tank_volume = V_u
    s.ox_initial_mass = m0_l_u + m0_v_u


    @singleton
    class debugme:
        def __init__(self):
            self.state = {
                "V_l_u": ([], "Volume [m^3]"),
                "V_v_u": ([], "Volume [m^3]"),
                "propV_u": ([], "Volume [m^3]"),
                "propP_u": ([], "Pressure [Pa]"),
            }
        def __setitem__(self, name, value):
            self.state[name][0].append(value)

    def df_liquid(m_l_u, m_v_u, T_u, P_u, P_d):
        """
        ONLY VALID WHEN LIQUID DRAINING.
        Returns the time derivatives of all input variables.
        """

        # Single-phase incompressible model (with Beta = 0):
        # (assuming tank liquid density as the "incompressible" density)
        rho_l_u = PropsSI("D", "P", P_u, "Q", 0, "N2O")
        mdot_SPI = Cd * Ainj * np.sqrt(2 * rho_l_u * (P_u - P_d))

        # Homogenous equilibrium model:
        x_u = m_v_u / (m_l_u + m_v_u)
        s_u = PropsSI("S", "P", P_u, "Q", x_u, "N2O")
        s_l_d = PropsSI("S", "P", P_d, "Q", 0, "N2O")
        s_v_d = PropsSI("S", "P", P_d, "Q", 1, "N2O")
        x_d = (s_u - s_l_d) / (s_v_d - s_l_d)
        h_u = PropsSI("H", "P", P_u, "Q", x_u, "N2O")
        h_d = PropsSI("H", "P", P_d, "Q", x_d, "N2O")
        rho_d = PropsSI("D", "P", P_d, "Q", x_d, "N2O")
        mdot_HEM = Cd * Ainj * rho_d * np.sqrt(2 * (h_u - h_d))

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
        dminj = mdot_SPI * (1 - k_NHNE) + mdot_HEM * k_NHNE


        # To determine temperature and vapourised mass derivatives,
        # we're going to have to use: our brain.
        #  V = const.
        #  m_l / rho_l + m_v / rho_v = const.
        #  d/dt (m_l / rho_l + m_v / rho_v) = 0
        #  d/dt (m_l / rho_l) + d/dt (m_v / rho_v) = 0
        #  0 = (dm_l * rho_l - m_l * drho_l) / rho_l**2  [quotient rule]
        #    + (dm_v * rho_v - m_v * drho_v) / rho_v**2
        # dm_l = -dminj - dm_v  [injector and vapourisation]
        #  0 = ((-dminj - dm_v) * rho_l - m_l * drho_l) / rho_l**2
        #    + (dm_v * rho_v - m_v * drho_v) / rho_v**2
        #  0 = -dminj / rho_l
        #    - dm_v / rho_l
        #    - m_l * drho_l / rho_l**2
        #    + dm_v / rho_v
        #    - m_v * drho_v / rho_v**2
        #  0 = dm_v * (1/rho_v - 1/rho_l)
        #    - dminj / rho_l
        #    - m_l * drho_l / rho_l**2
        #    - m_v * drho_v / rho_v**2
        # drho = d/dt (rho) = d/dT (rho) * dT/dt  [chain rule]
        # drhodT = d/dT (rho)
        #  0 = dm_v * (1/rho_v - 1/rho_l)
        #    - dminj / rho_l
        #    - m_l * dT * drhodT_l / rho_l**2
        #    - m_v * dT * drhodT_v / rho_v**2
        #  dm_v = (dminj / rho_l
        #         + m_l * dT * drhodT_l / rho_l**2
        #         + m_v * dT * drhodT_v / rho_v**2
        #         ) / (1/rho_v - 1/rho_l)
        #  dm_v = dminj / rho_l / (1/rho_v - 1/rho_l)
        #       + dT / (1/rho_v - 1/rho_l) * (m_l * drhodT_l / rho_l**2
        #                                   + m_v * drhodT_v / rho_v**2)
        # let:
        #   foo = dminj / rho_l / (1/rho_v - 1/rho_l)
        #   bar = (m_l * drhodT_l / rho_l**2
        #        + m_v * drhodT_v / rho_v**2) / (1/rho_v - 1/rho_l)
        #  dm_v = foo + dT * bar
        # So, dm_v depends on dT but dT also depends on dm_v:
        #  dT = dE / Cp  [open system energy change]
        #  dT = dEvap / Cp  [assuming adiabatic]
        #  dT = (hf - hg) * dm_v / Cp  [wrong, see "OH BLOODY HELL"]
        # let: hgf = hf - hg
        # OH BLOODY HELL hgf is also a function of time
        # lets take a step back.
        #  Evap = m_v * hgf
        #  dEvap = d/dt (m_v * hgf)
        #  dEvap = hgf * dm_v + m_v * d/dt (hgf)
        # d/dt (hgf) = d/dT (hgf) * dT/dt  [chain rule]
        # dhgfdT = d/dT (hgf)
        # so:
        #  dT = dEvap / Cp
        #  dT = (dm_v * hgf + m_v * dT * dhgfdT) / Cp
        #  Cp * dT = hgf * dm_v + m_v * dT * dhgfdT
        # Oh hey its just simultaneous equations. substitution.
        #  Cp * dT = hgf * (foo + dT * bar) + m_v * dT * dhgfdT
        #  dT * (Cp - hgf * bar - m_v * dhgfdT) = hgf * foo
        # => dT = hgf * foo / (Cp - hgf * bar - m_v * dhgfdT)
        # insane.

        rho_l_u = PropsSI("D", "T", T_u, "Q", 0, "N2O")
        rho_v_u = PropsSI("D", "T", T_u, "Q", 1, "N2O")
        drhodT_l_u = (PropsSI("D", "T", T_u + DT, "Q", 0, "N2O") - rho_l_u) / DT
        drhodT_v_u = (PropsSI("D", "T", T_u + DT, "Q", 1, "N2O") - rho_v_u) / DT

        Cp_l_u = m_l_u * PropsSI("C", "T", T_u, "Q", 0, "N2O")
        Cp_v_u = m_v_u * PropsSI("C", "T", T_u, "Q", 1, "N2O")
        Cp_u = Cp_l_u + Cp_v_u + Cpwall # including wall heat mass.

        hgf_u = (PropsSI("H", "T", T_u, "Q", 0, "N2O")
               - PropsSI("H", "T", T_u, "Q", 1, "N2O"))
        dhgfdT_u = (PropsSI("H", "T", T_u + DT, "Q", 0, "N2O")
                  - PropsSI("H", "T", T_u + DT, "Q", 1, "N2O")
                  - hgf_u) / DT

        foo = dminj / rho_l_u / (1/rho_v_u - 1/rho_l_u)
        bar = (m_l_u * drhodT_l_u / rho_l_u**2
             + m_v_u * drhodT_v_u / rho_v_u**2) / (1/rho_v_u - 1/rho_l_u)

        dT_u = hgf_u * foo / (Cp_u - hgf_u * bar - m_v_u * dhgfdT_u)

        dm_v_u = foo + dT_u * bar
        dm_l_u = -dminj - dm_v_u


        # Tank is saturated and remains saturated.
        #  P = Psat  [which is a function of T]
        #  d/dt (P) = d/dt (Psat)
        #  d/dt (P) = d/dT (Psat) * dT/dt  [chain rule]
        # note we don't use P_u when determining d/dT (Psat) since that
        # will lead to runaway error when P_u (inevitably) deviates
        # slightly from the genuine saturation pressure.
        dPdT_u = (PropsSI("P", "T", T_u + DT, "Q", 0, "N2O")
                - PropsSI("P", "T", T_u, "Q", 0, "N2O")) / DT
        dP_u = dPdT_u * dT_u


        # TODO: sim cc
        dT_d = 0.0
        dP_d = 0.0



        # Debug me:
        V_l_u = m_l_u / PropsSI("D", "T", T_u, "Q", 0, "N2O")
        V_v_u = m_v_u / PropsSI("D", "T", T_u, "Q", 1, "N2O")
        debugme["V_l_u"] = V_l_u
        debugme["V_v_u"] = V_v_u
        debugme["propV_u"] = (V_l_u + V_v_u) / V_u
        debugme["propP_u"] = P_u / PropsSI("P", "T", T_u, "Q", 0, "N2O")


        return dm_l_u, dm_v_u, dT_u, dP_u, dP_d


    def df_vapour(m_l_u, m_v_u, T_u, P_u, P_d):
        """
        ONLY VALID WHEN VAPOUR DRAINING.
        Returns the time derivatives of all input variables.
        """
        dm_l_u = dm_v_u = dT_u = dP_u = dP_d = -10.0
        return dm_l_u, dm_v_u, dT_u, dP_u, dP_d

    try:
        state = [
            [m0_l_u],
            [m0_v_u],
            [T0_u],
            [P0_u],
            [s.injector_initial_pressure_ratio * P0_u],
        ]
        dstate0 = None
        current_time = 0.0
        while current_time < 80.0:
            current_time += Dt

            # If liquid in that thang.
            if state[0][-1] > 0.01:
                dstate1 = df_liquid(*[v[-1] for v in state])
            # elif state[1][-1] > 0.01:
            #     dstate1 = df_vapour(*[v[-1] for v in state])
            else:
                raise Exception("dujj")
            dstate1 = np.array(dstate1)
            assert len(dstate1) == len(state)

            # order 2 adams-bashforth integration (higher accuracy than
            # something like explicit euler (plus i make a point not to
            # use explicit euler (its just not it))). This is why we
            # track the previous dstate.
            if dstate0 is None:
                dstate0 = dstate1
            Dstate = Dt * (1.5 * dstate1 - 0.5 * dstate0)
            for Dv, v in zip(Dstate, state):
                v.append(v[-1] + Dv)
                if v[-1] < 0.0:
                    v[-1] = 0.0

            dstate0 = dstate1

    except Exception:
        traceback.print_exc()

    m_l_u, m_v_u, T_u, P_u, P_d = state

    s.burn_time = (len(m_l_u) - 1) * Dt
    t = np.linspace(0, s.burn_time, len(m_l_u))
    mask = np.ones(len(t), dtype=bool)
    # mask = (np.arange(len(t)) >= int(0.85 * len(t)))

    s.tank_pressure = np.array(P_u)
    s.cc_pressure = np.array(P_d)
    s.tank_temperature = np.array(T_u)
    s.ox_mass_liquid = np.array(m_l_u)
    s.ox_mass_vapour = np.array(m_v_u)
    s.ox_mass = s.ox_mass_liquid + s.ox_mass_vapour
    if len(s.ox_mass) == 1:
        dminj = np.array([0.0])
    else:
        dminj = np.diff(s.ox_mass)
        dminj = np.append(dminj, dminj[-1])
    s.injector_mass_flow_rate = -dminj / Dt

    plotme = [
        (s.tank_pressure, "Tank pressure", "Pressure [Pa]"),
        (s.cc_pressure, "CC pressure", "Pressure [Pa]"),
        (s.tank_temperature, "Tank temperature", "Temperature [K]"),
        (s.injector_mass_flow_rate, "Injector mass flow rate", "Mass flow rate [kg/s]"),
        (s.ox_mass_liquid, "Tank liquid mass", "Mass [kg]"),
        (s.ox_mass_vapour, "Tank vapour mass", "Mass [kg]"),
        (s.ox_mass_vapour/s.ox_mass, "Tank quality", "Mass [kg]"),
        (s.ox_mass_liquid + s.ox_mass_vapour, "Tank mass", "Mass [kg]"),
    ]
    def doplot(plotme):
        plt.figure()
        ynum = 2
        xnum = (len(plotme) + 1) // 2
        for i, elem in enumerate(plotme):
            if elem is ...:
                continue
            y, title, ylabel = elem
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
        plt.subplots_adjust(left=0.05, right=0.97, wspace=0.4, hspace=0.3)
    doplot(plotme)
    doplot([(v[0], k, v[1]) for k, v in debugme.state.items()])
    plt.show()

    return new_top




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

    assert s.ox_type == "N2O"
    assert s.fuel_type == "paraffin"


    # Firstly get the easy masses/coms/length out of the way.

    top = -s.locked_length

    s.locked_com = top + s.locked_local_com
    top += s.locked_length

    top += simulate_burn(s, top) # does burn and tank properties.

    s.mov_com = top + s.mov_local_com
    top += s.mov_length

    s.injector_com = top + s.injector_local_com
    top += s.injector_length

    # Using rule-of-thumb pre- and post-cc lengths:
    s.cc_pre_length = s.cc_diameter
    s.cc_post_length = 1.5 * s.cc_diameter
    s.cc_length = s.cc_pre_length + s.fuel_length + s.cc_post_length
    cc_wall_cyl = Cylinder.pipe(s.cc_length, inner_diameter=s.cc_diameter)
    cc_wall_cyl.set_thickness_for_stress(max(s.cc_pressure), s.cc_wall_yield_strength)
    s.cc_wall_thickness = cc_wall_cyl.thickness
    s.cc_wall_mass = cc_wall_cyl.mass(s.cc_wall_density)
    s.cc_wall_com = cc_wall_cyl.com(top)

    s.fuel_density = 900 # paraffin wax density.
    s.fuel_com = top + s.cc_pre_length + s.fuel_length / 2
    fuel_cyl = Cylinder.pipe(s.fuel_length, outer_diameter=s.cc_diameter)
    fuel_cyl.thickness = s.fuel_thickness
    s.fuel_initial_mass = fuel_cyl.mass(s.fuel_density)
    s.fuel_mass = np.linspace(s.fuel_initial_mass, 0, len(s.ox_mass))
    top += s.cc_length

    # TEMP, nasacea does it i think.
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



sys = Sys()

sys.locked_mass = 5.0 # [kg]
sys.locked_length = 2.0 # [m]
sys.locked_local_com = 1.3 # downwards from top, [m]

sys.tank_length = 0.55 # [m] OUTPUT
sys.tank_wall_density = 2720.0 # Al6061, [kg/m^3]
sys.tank_wall_yield_strength = 241e6 # Al6061, [Pa]
sys.tank_wall_specific_heat_capacity = 896 # Al6061, [J/kg/K]

sys.ox_type = "N2O" # required
sys.ox_volume_fill_frac = 0.8 # [-]

sys.mov_mass = 0.5 # [kg]
sys.mov_length = 0.1 # [m]
sys.mov_local_com = sys.mov_length / 2 # [m]

sys.injector_mass = 0.5 # [kg]
sys.injector_length = 0.02 # [m]
sys.injector_local_com = sys.injector_length / 2 # [m]
sys.injector_discharge_coeff = 0.67 # [-]
sys.injector_orifice_area = 40 * np.pi/4 * 0.5e-3**2 # [m^2], OUTPUT
sys.injector_initial_pressure_ratio = 0.5 # [-], INPUT

sys.cc_diameter = 0.060 # [m], OUTPUT
sys.cc_wall_density = 2720.0 # Al6061, [kg/m^3]
sys.cc_wall_yield_strength = 241e6 # Al6061, [Pa]

sys.fuel_type = "paraffin" # required.
sys.fuel_length = 0.2 # [m], OUTPUT
sys.fuel_thickness = 0.017 # [m], OUTPUT

sys.nozzle_exit_area = 0.01 # [m^2], OUTPUT
sys.nozzle_thrust_efficiency = 0.9 # [-]

sys.rocket_target_apogee = 30000 / 3.281 # [m]
sys.rocket_diameter = 0.145 # [m]
sys.rocket_stability = 1.5 # [-?]

sys.environment_temperature = 20 + 273.15 # [K]
sys.integration_dt = 0.02 # [-]

cost(sys)
