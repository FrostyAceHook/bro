import math
import numpy as np
from dataclasses import dataclass


INPUT = object()
OUTPUT = object()
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
        s.tank_initial_temperature = INPUT # [K]
        s.tank_temperature = ... # [K]
        s.tank_pressure = ... # [Pa]

        s.ox_density = INPUT # [kg/m^3]
        s.ox_initial_mass = ... # [kg]
        s.ox_mass = ... # [kg]

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
        s.injector_pressure_ratio = ... # [Pa/Pa]
        s.injector_mass_flow_rate = ... # [kg/s]

        s.cc_diameter = OUTPUT # [m]
        s.cc_pressure = ... # [Pa]
        s.cc_temperature = ... # [K]
        s.cc_pre_length = ... # [m]
        s.cc_post_length = ... # [m]
        s.cc_length = ... # [m]
        s.cc_wall_density = INPUT # [kg/m^3]
        s.cc_wall_yield_strength = INPUT # [Pa]
        s.cc_wall_thickness = ... # [m]
        s.cc_wall_mass = ... # [kg]
        s.cc_wall_com = ... # [m]

        s.fuel_density = INPUT # [kg/m^3]
        s.fuel_length = OUTPUT # [m]
        s.fuel_thickness = OUTPUT # [m]
        s.fuel_com = ... # [m]
        s.fuel_initial_mass = ... # [kg]
        s.fuel_mass = ... # [kg]

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

    @classmethod
    def input_names(cls):
        return [name for name, value in cls().items() if value is INPUT]
    @classmethod
    def output_names(cls):
        return [name for name, value in cls().items() if value is OUTPUT]

    def has_all_inputs(s):
        return all(value is not INPUT for value in s.values())
    def has_all_independants(s):
        return all(value is not INPUT and value is not OUTPUT
                   for value in s.values())

    def __len__(s):
        return len(s.__dict__)
    def __iter__(s):
        return iter(s.__dict__)
    def __getitem__(s, name):
        if not isinstance(name, str):
            raise TypeError(f"expected string name, got '{type(name).__name__}'")
        return s.__dict__[name]
    def keys(s):
        return s.__dict__.keys()
    def values(s):
        return s.__dict__.values()
    def items(s):
        return s.__dict__.items()

    def __repr__(s):
        return "\n".join(f"{key} = {repr(value)}" for key, value in s.items())


class Cylinder:
    @classmethod
    def solid(cls, length, diameter):
        return cls(length, 0, diameter)

    @classmethod
    def hollow(cls, length, *, inner_diameter=None, outer_diameter=None):
        return cls(length, inner_diameter, outer_diameter)

    def __init__(self, length, inner_diameter, outer_diameter):
        self.length = None
        self.inner_diameter = None
        self.outer_diameter = None
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
    def set_thickness(self, new_thickness, keep_inner=None):
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
        b = self.outer_diameter
        a = self.inner_diameter
        return self.length * math.pi/4 * (b**2 - a**2)

    def mass(self, density):
        return density * self.volume()

    def com(self, top):
        return top + self.length / 2

    def set_thickness_for_stress(self, max_pressure, yield_strength, sf=2):
        # Finding min thickness for hoop stress, using given safety factor and
        # thin walled approximation:
        #   thickness = sf * max_pressure * inner_diameter / (2 * yield_strength)
        # If inner diameter is unconstrained:
        #   thickness = sf * max_pressure * (outer_diameter - thickness) / (2 yield_strength)
        #                     sf * max_pressure * outer_diameter
        #   => thickness = ----------------------------------------
        #                   sf * max_pressure + 2 * yield_strength
        if self.inner_diameter is None and self.outer_diameter is None:
            raise ValueError("unconstrained diameters")
        if self.outer_diameter is not None and self.outer_diameter is not None:
            raise ValueError("overconstrained diameters")
        two_ys = 2 * yield_strength
        max_pressure *= sf
        if self.inner_diameter is not None:
            self.thickness = max_pressure * self.inner_diameter / twoys
        else:
            self.thickness = max_pressure * self.outer_diameter / (max_pressure + twoys)


def cost(s):
    """
    Returns some float based on the given system state s.t. when the return is
    minimised, the system is optimised. Note that both inputs and outputs (aka
    all independant system parameters) must be set.
    """
    assert s.has_all_independants()

    # Firstly get the easy masses/coms/length out of the way.

    top = -s.locked_length

    s.locked_com = top + s.locked_local_com
    top += s.locked_length

    tank_wall_cyl = Cylinder.hollow(s.tank_length, outer_diameter=s.rocket_diameter)
    tank_wall_cyl.set_thickness_for_stress(s.tank_pressure.max(), s.tank_wall_yield_strength)
    s.tank_com = tank_wall_cyl.com(top)
    s.tank_wall_thickness = tank_wall_cyl.thickness
    s.tank_wall_mass = tank_wall_cyl.mass(s.tank_wall_density)

    ox_cyl = Cylinder.solid(s.tank_length, tank_wall_cyl.inner_diameter)
    s.ox_initial_mass = ox_cyl.mass(s.ox_density) # not right, gotta account for fill fraction.
    top += s.tank_length

    s.mov_com = top + s.mov_local_com
    top += s.mov_length

    s.injector_com = top + s.injector_local_com
    top += s.injector_length

    # Using rule-of-thumb pre- and post-cc lengths:
    s.cc_pre_length = s.cc_diameter
    s.cc_post_length = 1.5 * s.cc_diameter
    s.cc_length = s.cc_pre_length + s.fuel_length + s.cc_post_length
    cc_wall_cyl = Cylinder.hollow(s.cc_length, inner_diameter=s.cc_diameter)
    cc_wall_cyl.set_thickness_for_stress(s.cc_pressure.max(), s.cc_wall_yield_strength)
    s.cc_wall_thickness = cc_wall_cyl.thickness
    s.cc_wall_mass = cc_wall_cyl.mass(s.cc_wall_density)
    s.cc_wall_com = cc_wall_cyl.com(top)

    s.fuel_com = top + s.cc_pre_length + s.fuel_length / 2
    fuel_cyl = Cylinder.hollow(s.fuel_length, outer_diameter=s.cc_diameter)
    fuel_cyl.thickness = s.fuel_thickness
    s.fuel_initial_mass = fuel_cyl.mass(s.fuel_density)
    top += s.cc_length

    s.nozzle_length = ... # nasacea does it i think.
    s.nozzle_com = ...
    s.nozzle_mass = ...

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

sys.locked_mass = 5 # [kg]
sys.locked_local_com = 1.3 # downwards from top, [m]
sys.locked_length = 2 # [m]

sys.tank_length = 0.3 # [m] OUTPUT
sys.tank_wall_density = 2700 # Al, [kg/m^3]
sys.tank_wall_yield_strength = 55e6 # 6061-O [Pa]
sys.tank_initial_temperature = 25 + 273.15 # [K]

sys.ox_density = 750 # liquid N2O, [kg/m^3]

sys.mov_mass = 0.5 # [kg]
sys.mov_length = 0.1 # [m]
sys.mov_local_com = sys.mov_length / 2 # [m]

sys.injector_mass = 0.5 # [kg]
sys.injector_length = 0.02 # [m]
sys.injector_local_com = sys.injector_length / 2 # [m]
sys.injector_discharge_coeff = ...
sys.injector_orifice_area = ... # [m^2], OUTPUT

sys.cc_diameter = OUTPUT # OUTPUT
sys.cc_wall_density = 2700 # Al, [kg/m^3]
sys.cc_wall_yield_strength = 55e6 # 6061-O [Pa]

sys.fuel_density = 900 # solid paraffin wax, [kg/m^3]
sys.fuel_length = 0.125 # [m], OUTPUT
sys.fuel_thickness = 0.017 # [m], OUTPUT

sys.nozzle_thrust_efficiency = 0.9 # [unitless]

sys.rocket_target_apogee = 30000 / 3.281 # [m]
sys.rocket_diameter = 0.25 # [m]
sys.rocket_combustion_params = ...
sys.rocket_stability = 1.5 # [unitless?]

cost(sys)
