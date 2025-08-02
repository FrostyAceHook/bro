import os
import sys
from math import pi

from . import optimiser


def main():
    # Enable colours.
    os.system("")


    s = optimiser.Sys()

    s.target_apogee = 30000 / 3.281 # [m]

    s.locked_mass = 5.0 # [kg]
    s.locked_length = 2.0 # [m]
    s.locked_local_com = 1.3 # downwards from top, [m]

    s.tank_inner_length = 0.55 # [m], OUTPUT
    s.tank_wall_density = 2720.0 # Al6061, [kg/m^3]
    s.tank_wall_yield_strength = 241e6 # Al6061, [Pa]
    s.tank_wall_specific_heat_capacity = 896 # Al6061, [J/kg/K]

    s.ox_type = "N2O" # required.
    s.ox_volume_fill_frac = 0.8 # [-]

    s.mov_mass = 0.5 # [kg]
    s.mov_length = 0.1 # [m]
    s.mov_local_com = s.mov_length / 2 # [m]

    s.injector_mass = 0.5 # [kg]
    s.injector_length = 0.02 # [m]
    s.injector_local_com = s.injector_length / 2 # [m]
    s.injector_discharge_coeff = 0.9 # [-]
    s.injector_orifice_area = 40 * pi/4 * 0.5e-3**2 # [m^2], OUTPUT

    s.cc_diameter = 0.100 # [m], OUTPUT
    s.cc_combustion_efficiency = 1.0 # [-]
    s.cc_wall_density = 2720.0 # Al6061, [kg/m^3]
    s.cc_wall_yield_strength = 241e6 # Al6061, [Pa]

    s.fuel_type = "PARAFFIN" # required.
    s.fuel_length = 0.25 # [m], OUTPUT
    s.fuel_initial_thickness = 0.02 # [m], OUTPUT

    s.nozzle_throat_area = pi * 0.015**2 # [m^2]
    s.nozzle_discharge_coeff = 1.0 # [-]
    s.nozzle_thrust_efficiency = 0.9 # [-]

    s.rocket_diameter = 0.145 # [m]
    s.rocket_stability = 1.5 # [-?]

    s.ambient_temperature = 25 + 273.15 # [K]
    s.ambient_pressure = 101.325e3 # [Pa]
    s.ambient_density = 1.293 # [kg/m^3]
    s.ambient_molar_mass = 28.9647e-3 # [kg/mol]
    s.ambient_constant_pressure_specific_heat_capacity = 1005.0 # [J/kg/K]

    optimiser.cost(s)

if __name__ == "__main__":
    sys.exit(main())
