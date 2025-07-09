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
locked.local_com = "input"
locked.length = "input"
locked.com = lambda local_com: ...

tank = Sn()
tank.length = "output"
tank.com = lambda length: ...
tank.wall = Sn()
tank.wall.density = "input"
tank.wall.yield_strength = "input"
tank.wall.thickness = lambda length: ...
tank.wall.mass = lambda density, length, rocket_diameter, thickness: ...
tank.initial_temperature = "input"
tank.temperature = lambda initial_temperature: ... # assume constant? maybe sim.
tank.pressure = lambda temperature: ...

ox = Sn()
ox.density = "input"
ox.initial_mass = lambda density, tank_length, rocket_diameter, \
        tank_wall_thickness: ...
ox.mass = lambda initial_mass, injector_mass_flow_rate: ...

mov = Sn()
mov.mass = "input"
mov.local_com = "input"
mov.length = "input"
mov.com = lambda local_com, *lengths: ...

injector = Sn()
injector.mass = "input"
injector.local_com = "input"
injector.length = "input" # maybe neglible.
injector.com = lambda local_com, *lengths: ...
injector.discharge_coeff = "input"
injector.orifice_area = "output"
injector.pressure_ratio = lambda tank_pressure, cc_pressure: ...
injector.mass_flow_rate = lambda orifice_area, pressure_ratio, ox_density: ...

cc = Sn()
cc.diameter = "output"
cc.pressure = lambda *something: ...
cc.temperature = lambda *something: ...
cc.pre_length = lambda diameter: ...
cc.post_length = lambda diameter: ...
cc.length = lambda diameter, fuel_length: ...
cc.wall = Sn()
cc.wall.density = "input"
cc.wall.yield_strength = "input"
cc.wall.thickness = lambda max_pressure: ...
cc.wall.mass = lambda density, length, cc_diameter, thickness: ...
cc.wall.com = lambda cc_length, *lengths: ...

fuel = Sn()
fuel.density = "input"
fuel.length = "output"
fuel.thickness = "output"
fuel.com = lambda length, *lengths: ...
fuel.initial_mass = lambda density, length, cc_diameter, thickness: ...
fuel.mass = lambda initial_mass, *other_things: ...

nozzle = Sn()
nozzle.length = lambda *something: ...
nozzle.com = lambda length, *lengths: ...
nozzle.mass = lambda length: ...
nozzle.efficiency = "input"
# Idk exact thrust dependancies, probably has more.
nozzle.thrust = lambda efficiency, rocket_combustion_params, \
        injector_mass_flow_rate, cc_pressure: ...

rocket = Sn()
rocket.target_apogee = "input"
rocket.diameter = "input"
rocket.combustion_params = ... # specific parameters for this ox + fuel
rocket.length = lambda *section_lengths: ...
rocket.dry_mass = lambda *section_masses: ...
rocket.mass = lambda *section_masses: ...
rocket.com = lambda *section_masses, *section_coms: ...
# Approximate drag coefficient as some function of only length?
rocket.drag_coeff = lambda length: ...
rocket.stability = "input" # fixed by fin design.
rocket.net_force = lambda thrust, drag, mass: ...
rocket.altitude = lambda *everything: ...


# Objectives are:
# - final 'rocket.altitude' as close as possible to 'rocket.target_apogee'
# - minimise 'ox.initial_mass'
# - minimise 'fuel.initial_mass'
# - minimise 'rocket.length'
# - minimise peak thrust (?)
