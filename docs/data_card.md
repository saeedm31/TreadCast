# Data card: the simulated fleet

| | |
|---|---|
| **Source** | `treadcast/sim/`, fixed seed, all numbers in `treadcast/config.py` |
| **Why simulated** | There is no public data from in-tire sensors; a simulator also gives the true depth every day |
| **Fleet** | 12 trucks: 5 long-haul, 4 regional, 3 urban; 6 tires each (2 steer, 4 drive) |
| **Period** | 2 years of daily operation (730 days) |
| **Tire** | 315/80 R22.5; new depth 16 mm (steer) and 21 mm (drive); two tire models per axle with different wear and stiffness |
| **Personal data** | None |

## What is simulated

- **Driving:** working days, km per day, speed, load and road roughness by route type.
- **Tire state:** pressure and temperature, true wear from load, speed, route and tire model.
- **Sensor signal:** radial acceleration around the contact patch (v²/R off the road, about 0 g in the patch,
  spikes at entry and exit, tread-block ringing), with sensor noise. On the sensor, steady turns are lined up by angle
  and one median template of 256 points is sent per day.
- **Telematics:** GPS distance (with noise and missing days) and an axle-load estimate.
- **Services:** every 30,000 km or 75 days; the gauge reading has 0.3 mm noise; a tire at or below 4 mm is replaced.

## Labels

| Kind | Where from | Weight |
|---|---|---|
| New depth | when a tire is fitted | 1 |
| Gauge reading | at each service | 1 |
| Pseudo-label | straight line between two readings | 0.3, training only |

## Split

| Use | What |
|---|---|
| Test trucks | T03, T08, T11 (one per route type); their tires are never trained on |
| Training | the other trucks, days before day 500 only |

## Limits
The simulator follows published physics of tire-mounted sensors, but it is a simplification. A real project starts
with test-drum data (tires buffed to known depths) and an instrumented fleet.
