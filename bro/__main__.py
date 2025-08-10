import os
import sys
from math import pi

from . import optimiser


def main():
    # Enable colours.
    os.system("")


    s = optimiser.Sys()

    s.target_apogee = 30000 / 3.281 # [m]

    s.ox_type = "N2O" # required.
    s.fuel_type = "PARAFFIN" # required.

    s.locked_mass = 20.0 # [kg]
    s.locked_length = 1.0 # [m]
    s.locked_com = 1.3 # [m]

    s.rocket_diameter = 145e-3 # [m]

    s.initial_altitude = 0.0 # [m]

    s.ambient_temperature = 25 + 273.15 # [K]

    s.tank_interior_length = 600e-3 # [m] OUTPUT
    s.tank_wall_density = 2720.0 # [kg/m^3] Al6061
    s.tank_wall_yield_strength = 241e6 # [Pa] Al6061
    s.tank_wall_specific_heat_capacity = 896 # [J/kg/K] Al6061
    s.tank_wall_safety_factor = 2.5 # [-]
    s.tank_initial_volumetric_fill_fraction = 0.8 # [-]

    s.mov_mass = 0.5 # [kg]
    s.mov_length = 20e-3 # [m]
    s.mov_com = 10e-3 # [m]

    s.injector_mass = 0.5 # [kg]
    s.injector_length = 20e-3 # [m]
    s.injector_com = 10e-3 # [m]
    s.injector_discharge_coefficient = 0.8 # [-]
    s.injector_orifice_area = 70 * pi/4 * 0.5e-3**2 # [m^2] OUTPUT

    s.combustion_chamber_diameter = 80e-3 # [m] OUTPUT
    s.combustion_chamber_wall_density = 2720.0 # [kg/m^3] Al6061
    s.combustion_chamber_wall_yield_strength = 241e6 # [Pa] Al6061
    s.combustion_chamber_wall_safety_factor = 3.5 # [-]

    s.fuel_length = 300e-3 # [m] OUTPUT
    s.fuel_initial_thickness = 25e-3 # [m] OUTPUT

    s.nozzle_discharge_coefficient = 0.8 # [-]
    s.exit_area_to_throat_area_ratio = 4.0 # [-] OUTPUT
    s.throat_area = pi/4 * 30e-3**2 # [m^2] OUTPUT

    optimiser.cost(s)

if __name__ == "__main__":
    sys.exit(main())
