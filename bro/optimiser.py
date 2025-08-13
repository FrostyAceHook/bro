import traceback
import time
from math import pi

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from . import bridge




def singleton(cls):
    """
    Returns a single instantiation of the given class, and prevents further creation of the class.
    - class decorator.
    """

    instance = cls()
    if getattr(instance, "__doc__", None) is None:
        if getattr(cls, "__doc__", None) is not None:
            instance.__doc__ = cls.__doc__

    def throw(cls, *args, **kwargs):
        raise TypeError(f"cannot create another instance of singleton {repr(cls.__name__)}")
    cls.__new__ = throw

    return instance


def frozen(cls):
    """
    Disables creating or deleting new object attributes of the given class (outside the `__init__` method).
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

        s.ox_type = INPUT # must be "N2O"
        s.fuel_type = INPUT # must be "PARAFFIN"

        s.locked_mass = INPUT # [kg]
        s.locked_length = INPUT # [m]
        s.locked_com = INPUT # [m]

        s.rocket_diameter = INPUT # [m]

        s.initial_altitude = INPUT # [m]

        s.ambient_temperature = INPUT # [K]

        s.tank_interior_length = OUTPUT # [m]
        s.tank_wall_density = INPUT # [kg/m^3]
        s.tank_wall_yield_strength = INPUT # [Pa]
        s.tank_wall_specific_heat_capacity = INPUT # [J/kg/K]
        s.tank_wall_safety_factor = INPUT # [-]
        s.tank_initial_volumetric_fill_fraction = INPUT # [-]

        s.mov_mass = INPUT # [kg]
        s.mov_length = INPUT # [m]
        s.mov_com = INPUT # [m]

        s.injector_mass = INPUT # [kg]
        s.injector_length = INPUT # [m]
        s.injector_com = INPUT # [m]
        s.injector_discharge_coefficient = INPUT # [-]
        s.injector_orifice_area = OUTPUT # [m^2]

        s.combustion_chamber_diameter = OUTPUT # [m]
        s.combustion_chamber_wall_density = INPUT # [kg/m^3]
        s.combustion_chamber_wall_yield_strength = INPUT # [Pa]
        s.combustion_chamber_wall_safety_factor = INPUT # [-]

        s.fuel_length = OUTPUT # [m]
        s.fuel_initial_thickness = OUTPUT # [m]

        s.nozzle_discharge_coefficient = INPUT # [-]
        s.exit_area_to_throat_area_ratio = OUTPUT # [-]
        s.throat_area = OUTPUT # [m^2]


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

    # sim.c only supports paraffin+N2O.
    assert s.ox_type == "N2O"
    assert s.fuel_type == "PARAFFIN"

    # Coupla buffers.
    t        = np.empty(10_000_000, dtype=np.float64)
    alt_r    = np.empty(10_000_000, dtype=np.float64)
    vel_r    = np.empty(10_000_000, dtype=np.float64)
    acc_r    = np.empty(10_000_000, dtype=np.float64)
    m_r      = np.empty(10_000_000, dtype=np.float64)
    com_r    = np.empty(10_000_000, dtype=np.float64)
    T_t      = np.empty(10_000_000, dtype=np.float64)
    T_g      = np.empty(10_000_000, dtype=np.float64)
    P_t      = np.empty(10_000_000, dtype=np.float64)
    P_c      = np.empty(10_000_000, dtype=np.float64)
    P_a      = np.empty(10_000_000, dtype=np.float64)
    m_l      = np.empty(10_000_000, dtype=np.float64)
    m_v      = np.empty(10_000_000, dtype=np.float64)
    m_f      = np.empty(10_000_000, dtype=np.float64)
    dm_inj   = np.empty(10_000_000, dtype=np.float64)
    dm_reg   = np.empty(10_000_000, dtype=np.float64)
    dm_out   = np.empty(10_000_000, dtype=np.float64)
    m_g      = np.empty(10_000_000, dtype=np.float64)
    cp_g     = np.empty(10_000_000, dtype=np.float64)
    cv_g     = np.empty(10_000_000, dtype=np.float64)
    y_g      = np.empty(10_000_000, dtype=np.float64)
    R_g      = np.empty(10_000_000, dtype=np.float64)
    ofr      = np.empty(10_000_000, dtype=np.float64)
    Fthrust  = np.empty(10_000_000, dtype=np.float64)
    Fdrag    = np.empty(10_000_000, dtype=np.float64)
    Fgravity = np.empty(10_000_000, dtype=np.float64)

    # Pack it up real nice for the sim.
    state = bridge.State(
        t=t,
        alt_r=alt_r,
        vel_r=vel_r,
        acc_r=acc_r,
        m_r=m_r,
        com_r=com_r,
        T_t=T_t,
        T_g=T_g,
        P_t=P_t,
        P_c=P_c,
        P_a=P_a,
        m_l=m_l,
        m_v=m_v,
        m_f=m_f,
        dm_inj=dm_inj,
        dm_reg=dm_reg,
        dm_out=dm_out,
        m_g=m_g,
        cp_g=cp_g,
        cv_g=cv_g,
        y_g=y_g,
        R_g=R_g,
        ofr=ofr,
        Fthrust=Fthrust,
        Fdrag=Fdrag,
        Fgravity=Fgravity,

        target_apogee=s.target_apogee,
        m_locked=s.locked_mass,
        L_locked=s.locked_length,
        com_locked=s.locked_com,
        D_r=s.rocket_diameter,
        alt0_r=s.initial_altitude,
        T_a=s.ambient_temperature,
        L_tw=s.tank_interior_length,
        rho_tw=s.tank_wall_density,
        Ys_tw=s.tank_wall_yield_strength,
        c_tw=s.tank_wall_specific_heat_capacity,
        sf_tw=s.tank_wall_safety_factor,
        vff0_l=s.tank_initial_volumetric_fill_fraction,
        m_mov=s.mov_mass,
        L_mov=s.mov_length,
        com_mov=s.mov_com,
        m_inj=s.injector_mass,
        L_inj=s.injector_length,
        com_inj=s.injector_com,
        Cd_inj=s.injector_discharge_coefficient,
        A_inj=s.injector_orifice_area,
        D_c=s.combustion_chamber_diameter,
        rho_cw=s.combustion_chamber_wall_density,
        Ys_cw=s.combustion_chamber_wall_yield_strength,
        sf_cw=s.combustion_chamber_wall_safety_factor,
        L_f=s.fuel_length,
        th0_f=s.fuel_initial_thickness,
        Cd_nzl=s.nozzle_discharge_coefficient,
        eps=s.exit_area_to_throat_area_ratio,
        A_throat=s.throat_area,
    )
    # Send it.
    _start = time.perf_counter()
    count = state.sim()
    _end = time.perf_counter()
    print(f"Finished burn sim in {1e3*(_end - _start):.3f}ms")

    # Trim unused memory.
    t        = t[:count]
    alt_r    = alt_r[:count]
    vel_r    = vel_r[:count]
    acc_r    = acc_r[:count]
    m_r      = m_r[:count]
    com_r    = com_r[:count]
    T_t      = T_t[:count]
    T_g      = T_g[:count]
    P_t      = P_t[:count]
    P_c      = P_c[:count]
    P_a      = P_a[:count]
    m_l      = m_l[:count]
    m_v      = m_v[:count]
    m_f      = m_f[:count]
    dm_inj   = dm_inj[:count]
    dm_reg   = dm_reg[:count]
    dm_out   = dm_out[:count]
    m_g      = m_g[:count]
    cp_g     = cp_g[:count]
    cv_g     = cv_g[:count]
    y_g      = y_g[:count]
    R_g      = R_g[:count]
    ofr      = ofr[:count]
    Fthrust  = Fthrust[:count]
    Fdrag    = Fdrag[:count]
    Fgravity = Fgravity[:count]

    finalme = {
        "tank pressure": P_t,
        "CC pressure": P_c,
        "tank temperature": T_t,
        "N2O liquid mass": m_l,
        "N2O vapour mass": m_v,
        "N2O mass": m_l + m_v,
    }
    if False:
        for name, array in finalme.items():
            s = f"Final {name} .."
            s += "." * (25 - len(s))
            print(f"{s} {array[-1]:,}")

    plotme = [
            # data, title, ylabel, y_lower_limit_as_zero
        [
            (alt_r*1e-3, "Altitude", "Altitude [km]", False),
            (vel_r, "Velocity", "Speed [m/s]", False),
            (Fthrust, "Thrust", "Force [N]", False),
            (Fdrag, "Drag", "Force [N]", False),
            (P_t, "Tank pressure", "Pressure [Pa]", False),
            (P_c, "CC pressure", "Pressure [Pa]", False),
            (T_t - 273.15, "Tank temperature", "Temperature [dC]", False),
            (T_g, "CC temperature", "Temperature [K]", False),
            (ofr, "Oxidiser-fuel ratio", "Ratio [-]", False),
            (dm_out, "Exhaust mass flow rate", "Mass flow rate [kg/s]", True),
            (dm_inj, "Injector mass flow rate", "Mass flow rate [kg/s]", True),
            (dm_reg, "Regression mass flow rate", "Mass flow rate [kg/s]", True),
            (m_l + m_v, "Tank mass", "Mass [kg]", True),
            (m_f, "Fuel mass", "Mass [kg]", True),
        ],
        [
            (acc_r, "Rocket acceleration", "Acceleration [m/s^2]", False),
            (m_r, "Rocket mass", "Mass [kg]", False),
            (P_a, "Atmospheric pressure", "Pressure [Pa]", False),
            (Fgravity, "Gravity", "Force [N]", False),
            (m_l, "Tank liquid mass", "Mass [kg]", True),
            (m_v, "Tank vapour mass", "Mass [kg]", True),
            (m_g, "CC gas mass", "Mass [kg]", False),
            (cp_g, "CC gas cp", "Specific heat capacity [J/kg/K]", False),
            (cv_g, "CC gas cv", "Specific heat capacity [J/kg/K]", False),
            (y_g, "CC gas gamma", "Ratio [-]", False),
            (R_g, "CC gas R", "[J/kg/K]", False),
        ],
        # [
        #     (P_t, "Tank pressure", "Pressure [Pa]", False),
        #     (P_c, "CC pressure", "Pressure [Pa]", False),
        #     (T_t - 273.15, "Tank temperature", "Temperature [dC]", False),
        #     (dm_inj, "Injector mass flow rate", "Mass flow rate [kg/s]", True),
        #     (m_l, "Tank liquid mass", "Mass [kg]", True),
        #     (m_v, "Tank vapour mass", "Mass [kg]", True),
        # ],
    ]

    def doplot(plotme):
        if not plotme:
            return
        plt.figure()
        ynum = 2
        xnum = (len(plotme) + 1) // 2
        for i, elem in enumerate(plotme):
            if elem is ...:
                continue
            y, title, ylabel, snapzero = elem
            plt.subplot(ynum, xnum, 1 + i // ynum + xnum * (i % ynum))
            plt.plot(t, y, "-" + "o"*(len(y) == 1))
            plt.title(title)
            plt.xlabel("Time [s]")
            plt.ylabel(ylabel)
            plt.grid()
            plt.xlim(0, t.max())
            if snapzero:
                _, ymax = plt.ylim()
                plt.ylim(0, ymax)
        plt.subplots_adjust(left=0.05, right=0.97, wspace=0.4, hspace=0.3)
    for plotmefr in plotme:
        doplot(plotmefr)
    plt.show()
