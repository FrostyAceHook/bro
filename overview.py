r"""
Cheeky rocket ascii:

   .     .
   :     : ----- locked (by rest of team)
   |     |
   | ___ | _________ origin line (coms measured from here, +ve down)
   ||   ||
   ||   || ----- tank (contains oxidiser)
   ||___||
   |     |
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


All parameters:

rocket (as a whole):
- apogee
- diameter
- mass
- length
- drag coefficient
- stability factor
- net force
    - thrust
    - drag
    - weight

locked
- mass + com + length

tank
- wall: mass + com + length + thickness
- ox mass
- temperature
- pressure

mov
- mass + com + length

injector
- mass + com + length
- discharge coefficient
- pressure ratio
- net orifice area
- mass flow rate

cc
- diameter
- wall: mass + com + length + thickness
- fuel: mass + com + length + thickness
- pre-cc length
- post-cc length
- temperature
- pressure
- thrust

nozzle
- mass + com + length
- efficiency (of some kind idk)
- thrust

fins
- mass + com
- effect on rocket drag coefficient
"""

from types import SimpleNamespace as Sn


# Note all "input" kinds may be considered fixed also.

locked = Sn()
locked.mass = "input"
locked.com = "input"
locked.length = "input"

tank = Sn()
tank.wall = Sn()
tank.wall.density = "input"
tank.wall.mass = lambda density, length, rocket_diameter, thickness: ...
tank.wall.com = lambda length: ...
tank.wall.thickness = lambda length: ...
tank.wall.length = "output"
tank.ox = Sn()
tank.ox.density = "input"
tank.ox.initial_mass = lambda density, wall_length, rocket_diameter, \
        wall_thickness: ...
tank.initial_temperature = "input"
tank.temperature = lambda initial_temperature: ... # assume constant? maybe sim.
tank.pressure = lambda temperature: ...

mov = Sn()
mov.mass = "input"
mov.com = "input"
mov.length = "input"

injector = Sn()
injector.mass = "input"
injector.com = "input"
injector.length = "input" # maybe neglible.
injector.discharge_coeff = "input"
injector.orifice_area = "output"
injector.pressure_ratio = lambda tank_pressure, cc_pressure: ...
injector.mass_flow_rate = lambda orifice_area, pressure_ratio, ox_density: ...

cc = Sn()
cc.diameter = "output"
cc.pressure = lambda *something: ...
cc.temperature = lambda *something: ...
cc.wall = Sn()
cc.wall.density = "input"
cc.wall.length = lambda fuel_length: ...
cc.wall.thickness = lambda max_pressure: ...
cc.wall.mass = lambda density, length, cc_diameter, thickness: ...
cc.fuel = Sn()
cc.fuel.density = "input"
cc.fuel.length = "output"
cc.fuel.thickness = "output"
cc.fuel.com = lambda length: ...
cc.fuel.initial_mass = lambda density, length, cc_diameter, thickness: ...
cc.pre_length = lambda fuel_length: ...
cc.post_length = lambda fuel_length: ...

nozzle = Sn()
nozzle.length = lambda *something: ...
nozzle.com = lambda length: ...
nozzle.mass = lambda length: ...
nozzle.efficiency = "input"
# Idk exact thrust dependancies, probably has more.
nozzle.thrust = lambda efficiency, rocket_combustion_params, \
        injector_mass_flow_rate, cc_pressure: ...

rocket = Sn()
rocket.target_apogee = "input"
rocket.diameter = "input"
rocket.combustion_params = ... # specific parameters for this ox + fuel
rocket.mass = lambda *section_masses: ...
rocket.length = lambda *section_lengthes: ...
# Approximate drag coefficient as some function of only length?
rocket.drag_coeff = lambda length: ...
rocket.stability = "input" # fixed by fin design.
rocket.net_force = lambda thrust, drag, mass: ...
rocket.altitude = lambda *everything: ...


# Objectives are:
# - final 'rocket.altitude' as close as possible to 'rocket.target_apogee'
# - minimise 'tank.ox.initial_mass'
# - minimise 'cc.fuel.initial_mass'
# - minimise 'rocket.length'
# - minimise peak thrust (?)
