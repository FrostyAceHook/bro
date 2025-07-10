import traceback

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI


def singleton(cls):
    """
    Returns a single instantiation of the given class, and prevents further creation of the class.
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
    Disables creating new object attributes of the given class (outside the `__init__` method).
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
        if getattr(self, "_in_init", 1):
            cooked = False
        elif name not in self.__dict__:
            raise AttributeError(f"cannot add attribute {repr(name)} to a frozen instance")
        super(cls, self).__setattr__(name, value)

    cls.__init__ = __init__
    cls.__setattr__ = __setattr__
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
        s.locked_mass = INPUT # [kg]
        s.locked_length = INPUT # [m]
        s.locked_local_com = INPUT # [m]
        s.locked_com = ... # [m]

        s.tank_length = OUTPUT # [m]
        s.tank_com = ... # [m]
        s.tank_wall_density = INPUT # [kg/m^3]
        s.tank_wall_yield_strength = INPUT # [Pa]
        s.tank_wall_thickness = ... # [m]
        s.tank_wall_mass = ... # [kg]
        s.tank_volume = ... # [m^3]
        s.tank_temperature = ... # [K]
        s.tank_pressure = ... # [Pa]

        s.ox_type = INPUT # must be "N2O"
        s.ox_volume_fill_frac = INPUT # [-]
        s.ox_initial_mass = ... # [kg]
        s.ox_mass = ... # [kg]
        s.ox_mass_liquid = ... # [kg]
        s.ox_mass_vapour = ... # [kg]
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
        s.injector_mass_flow_rate = ... # [kg/s]

        s.cc_diameter = OUTPUT # [m]
        s.cc_temperature = ... # [K]
        s.cc_pressure = ... # [Pa]
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
        s.fuel_mass = ... # [kg]
        s.fuel_com = ... # [m]

        s.nozzle_exit_area = OUTPUT # [m^2]
        s.nozzle_throat_area = ... # [m^2]
        s.nozzle_length = ... # [m]
        s.nozzle_local_com = ... # [m]
        s.nozzle_com = ... # [m]
        s.nozzle_mass = ... # [kg]
        s.nozzle_thrust_efficiency = INPUT # [-]
        s.nozzle_thrust = ... # [N]

        s.rocket_target_apogee = INPUT # [m]
        s.rocket_diameter = INPUT # [m]
        s.rocket_length = ... # [m]
        s.rocket_dry_mass = ... # [kg]
        s.rocket_mass = ... # [kg]
        s.rocket_com = ... # [m]
        s.rocket_drag_coeff = ... # [idk]
        s.rocket_stability = INPUT # [-]
        s.rocket_net_force = ... # [N]
        s.rocket_altitude = ... # [m]

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



def simulate_burn(s, tank_cyl):
    assert s.ox_type == "N2O"

    # 0 = initial
    # l = liquid
    # v = vapour
    # u = upstream (tank)
    # d = downstream (cc)

    dt = s.integration_dt
    Cd = s.injector_discharge_coeff
    Ainj = s.injector_orifice_area
    T0 = s.environment_temperature
    V_l = tank_cyl.volume() * s.ox_volume_fill_frac
    V_v = tank_cyl.volume() * (1 - s.ox_volume_fill_frac)
    m0_l = V_l * PropsSI("D", "T", T0, "Q", 0, s.ox_type)
    m0_v = V_v * PropsSI("D", "T", T0, "Q", 1, s.ox_type)
    s0_u = m0_v / (m0_l + m0_v)

    s.tank_volume = tank_cyl.volume()
    s.ox_initial_mass = m0_l + m0_v

    P_u = [PropsSI("P", "T", T0, "Q", s0_u, s.ox_type)]
    P_d = [s.injector_initial_pressure_ratio * P_u[0]]
    # P_d = [100e3]
    m_l_u = [m0_l]
    m_v_u = [m0_v]
    T_u = [T0]
    T_d = [T0]

    # debugging:
    mdot_s = []

    tmp = 0

    try:
        while True:
            tmp += dt
            if tmp > 120:
                break

            # Initially liquid draining:
            if m_l_u[-1] > 0.01:

                # Single-phase incompressible model (with Beta = 0):
                # (assuming tank liquid density as the "incompressible" density)
                rho_l_u = PropsSI("D", "T", T_u[-1], "Q", 0, s.ox_type)
                mdot_SPI = Cd * Ainj * np.sqrt(2 * rho_l_u * (P_u[-1] - P_d[-1]))
                # Homogenous equilibrium model:
                x_u = m_v_u[-1] / (m_l_u[-1] + m_v_u[-1])
                s_u = PropsSI("S", "P", P_d[-1], "Q", x_u, s.ox_type)
                s_l_d = PropsSI("S", "P", P_d[-1], "Q", 0, s.ox_type)
                s_v_d = PropsSI("S", "P", P_d[-1], "Q", 1, s.ox_type)
                x_d = (s_u - s_l_d) / (s_v_d - s_l_d)
                h_u = PropsSI("H", "P", P_u[-1], "Q", x_u, s.ox_type)
                h_d = PropsSI("H", "P", P_d[-1], "Q", x_d, s.ox_type)
                if h_u < h_d: # its been known to happen.
                    h_u = h_d # mdot_HEM -> 0 as m_l_u -> 0.
                rho_d = PropsSI("D", "P", P_d[-1], "Q", x_d, s.ox_type)
                mdot_HEM = Cd * Ainj * rho_d * np.sqrt(2 * (h_u - h_d))
                # Generalised non-homogenous non-equilibrium model:
                # (assuming downstream saturation pressure as the P_sat)
                P_sat = PropsSI("P", "T", T_d[-1], "Q", 1, s.ox_type)
                kappa = np.sqrt((P_u[-1] - P_d[-1]) / (P_sat - P_d[-1]))
                k_NHNE = 1 / (1 + kappa)
                mdot = mdot_SPI * k_NHNE + mdot_HEM * (1 - k_NHNE)

                # Discrete mass change this time step.
                dm_l_u = -min(mdot * dt, m_l_u[-1])

                # Mass and energy change from liquid vapourising to remain at sat pressure.
                #  dV = 0
                #  dm_l / rho_l + dm_v / rho_v = 0
                #  (dm_l_injector - dm_v) / rho_l + dm_v / rho_v = 0
                #  ((dm_l_injector - dm_v) rho_v + dm_v rho_l) / (rho_l rho_v) = 0
                # => dm_v = dm_l_injector rho_v / (rho_v - rho_l)
                rho_l_u = PropsSI("D", "T", T_u[-1], "Q", 0, s.ox_type)
                rho_v_u = PropsSI("D", "T", T_u[-1], "Q", 1, s.ox_type)
                dm_v_u = dm_l_u * rho_v_u / (rho_v_u - rho_l_u)
                dm_l_u -= dm_v_u
                h_l_u = PropsSI("H", "T", T_u[-1], "Q", 0, s.ox_type)
                h_v_u = PropsSI("H", "T", T_u[-1], "Q", 1, s.ox_type)
                dU_u = (h_v_u - h_l_u) * (-dm_v_u)

                # Energy change due to convection.
                # Convection coefficients (TODO: calc for real):
                hc_l = 200
                hc_v = 5
                # Contact area of each:
                V_l = m_l_u[-1] / PropsSI("D", "T", T_u[-1], "Q", 0, s.ox_type)
                V_v = m_v_u[-1] / PropsSI("D", "T", T_u[-1], "Q", 1, s.ox_type)
                A_l = tank_cyl.surface_area(cutoff=V_l/tank_cyl.volume())
                A_v = tank_cyl.surface_area() - A_l
                # Energy transferred:
                dU_u += hc_l * A_l * (s.environment_temperature - T_u[-1])
                dU_u += hc_v * A_v * (s.environment_temperature - T_u[-1])

                # Tank temperature change due to energy change.
                # TODO: check whether dU is in actual fact dH?
                c_l_u = PropsSI("C", "T", T_u[-1], "Q", 0, s.ox_type)
                c_v_u = PropsSI("C", "T", T_u[-1], "Q", 1, s.ox_type)
                C_u = m_l_u[-1] * c_l_u + m_v_u[-1] * c_v_u
                dT_u = dU_u / C_u

                # Tank pressure re-stablise to saturation pressure of new temperature.
                dP_u = PropsSI("P", "T", T_u[-1], "Q", 0, s.ox_type) - P_u[-1]

                # Do new state.
                m_l_u.append(m_l_u[-1] + dm_l_u)
                m_v_u.append(m_v_u[-1] + dm_v_u)
                T_u.append(T_u[-1] + dT_u)
                T_d.append(T_d[-1]) # TODO: dont assume constant
                P_u.append(P_u[-1] + dP_u)
                P_d.append(P_d[-1]) # TODO: dont assume constant
                if P_d[-1] < 100e3:
                    P_d[-1] = 100e3

                mdot_s.append(mdot)

            # Eventually vapour draining:
            else:
                break
                raise NotImplementedError()
    except Exception:
        traceback.print_exc()

    s.burn_time = (len(P_u) - 1) * dt
    t = np.linspace(0, s.burn_time, len(P_u))

    s.tank_pressure = np.array(P_u)
    s.cc_pressure = np.array(P_d)
    s.tank_temperature = np.array(T_u)
    s.cc_temperature = np.array(T_d)
    s.ox_mass_liquid = np.array(m_l_u)
    s.ox_mass_vapour = np.array(m_v_u)
    s.ox_mass = s.ox_mass_liquid + s.ox_mass_vapour

    plotme = [
        (s.tank_pressure, "Tank pressure", "Pressure [Pa]"),
        (s.cc_pressure, "CC pressure", "Pressure [Pa]"),
        (s.tank_temperature, "Tank temperature", "Temperature [K]"),
        (s.cc_temperature, "CC temperature", "Temperature [K]"),
        (s.ox_mass_liquid, "Tank liquid mass", "Mass [kg]"),
        (s.ox_mass_vapour, "Tank vapour mass", "Mass [kg]"),
        (s.ox_mass_vapour/s.ox_mass, "Tank quality", "Mass [kg]"),
        (mdot_s, "Injector mass flow rate", "Mass flow rate [kg/s]"),
    ]
    ynum = 2
    xnum = (len(plotme) + 1) // 2
    for i, elem in enumerate(plotme):
        if elem is ...:
            continue
        y, title, ylabel = elem
        plt.subplot(ynum, xnum, 1 + i // ynum + xnum * (i % ynum))
        if len(y) == 1:
            if len(y) == len(t) - 1:
                plt.plot(t[:-1], y, "o")
            else:
                plt.plot(t, y, "o")
        elif len(y) > 0:
            if len(y) == len(t) - 1:
                plt.plot(t[:-1], y)
            else:
                plt.plot(t, y)
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.grid()
    plt.subplots_adjust(left=0.05, right=0.97, wspace=0.4, hspace=0.3)
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

    assert s.ox_type == "N2O"
    assert s.fuel_type == "paraffin"


    # Firstly get the easy masses/coms/length out of the way.

    top = -s.locked_length

    s.locked_com = top + s.locked_local_com
    top += s.locked_length

    tank_wall_cyl = Cylinder.pipe(s.tank_length, outer_diameter=s.rocket_diameter)
    # Interdependance on thickness to determine pressure but need pressure to determine
    # thickness, obviously possible to solve simultaneous but here we just assume a small
    # thickness to find pressure then find required thickness for this pressure.
    tank_wall_cyl.thickness = 3e-3 # [m]

    tank_cyl = Cylinder.solid(s.tank_length, tank_wall_cyl.inner_diameter)

    simulate_burn(s, tank_cyl)

    # Recalculate actual wall thickness for the tank pressure.
    tank_wall_cyl.inner_diameter = None
    tank_wall_cyl.set_thickness_for_stress(max(s.tank_pressure), s.tank_wall_yield_strength)
    s.tank_com = tank_wall_cyl.com(top)
    s.tank_wall_thickness = tank_wall_cyl.thickness
    s.tank_wall_mass = tank_wall_cyl.mass(s.tank_wall_density)
    top += s.tank_length

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

sys.tank_length = 0.3 # [m] OUTPUT
sys.tank_wall_density = 2700.0 # Al 6061-O, [kg/m^3]
sys.tank_wall_yield_strength = 55e6 # Al 6061-O, [Pa]

sys.ox_type = "N2O" # required
sys.ox_volume_fill_frac = 0.8 # [-]

sys.mov_mass = 0.5 # [kg]
sys.mov_length = 0.1 # [m]
sys.mov_local_com = sys.mov_length / 2 # [m]

sys.injector_mass = 0.5 # [kg]
sys.injector_length = 0.02 # [m]
sys.injector_local_com = sys.injector_length / 2 # [m]
sys.injector_discharge_coeff = 0.67 # [-]
sys.injector_orifice_area = 20 * np.pi/4 * 0.5e-3**2 # [m^2], OUTPUT
sys.injector_initial_pressure_ratio = 0.5 # [-], INPUT

sys.cc_diameter = 0.060 # [m], OUTPUT
sys.cc_wall_density = 2700.0 # Al 6061-O, [kg/m^3]
sys.cc_wall_yield_strength = 55e6 # Al 6061-O, [Pa]

sys.fuel_type = "paraffin" # required.
sys.fuel_length = 0.125 # [m], OUTPUT
sys.fuel_thickness = 0.017 # [m], OUTPUT

sys.nozzle_exit_area = 0.01 # [m^2], OUTPUT
sys.nozzle_thrust_efficiency = 0.9 # [-]

sys.rocket_target_apogee = 30000 / 3.281 # [m]
sys.rocket_diameter = 0.25 # [m]
sys.rocket_stability = 1.5 # [-?]

sys.environment_temperature = 20 + 273.15 # [K]
sys.integration_dt = 0.02 # [-]


cost(sys)
