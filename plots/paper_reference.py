"""Digitised reference data from Nieckele et al. (2000).

The coordinates below were re-read from 600 dpi renders of Figures 3, 7, 8,
10 and 11.  They are samples of the printed curves and statements, not the
authors' raw numerical output.  They are used only for plots and error
calculations and are never imported by the flow solver or case definitions.

The re-reading corrects the earlier low-biased rising branch of Figure 7 and
several Figure 10 points that had been taken from the wrong curve or position.
No solver setting or published physical parameter was changed as part of this
reference-data correction.
"""

# (x_pig [m], V_pig [m/s]), non-isothermal curve in Figure 7.
P_F7_NON = [
    (50.0, 3.50),
    (150.0, 1.75),
    (300.0, 0.88),
    (500.0, 0.55),
    (700.0, 0.75),
    (850.0, 0.82),
    (900.0, 2.75),
    (1000.0, 4.47),
    (1100.0, 7.15),
    (1200.0, 10.43),
    (1280.0, 14.50),
]

# (time [s], inlet pressure [atm]) read from the labelled Figure 3 curves.
P_F3 = [
    (0.0, 10.0),
    (50.0, 20.0),
    (200.0, 36.0),
    (600.0, 61.0),
    (800.0, 65.0),
    (1100.0, 62.0),
    (1150.0, 48.0),
]

# (x_pig [km], V_pig [m/s]), non-isothermal curve in Figure 10.
P_F10 = [
    (5.0, 33.1),
    (7.0, 27.5),
    (9.9, 23.0),
    (11.0, 43.0),
    (12.0, 35.0),
    (13.0, 31.2),
    (15.0, 27.5),
    (17.0, 24.1),
    (19.9, 19.5),
    (21.0, 12.1),
    (22.0, 12.5),
    (25.0, 12.5),
    (29.0, 11.7),
]

# Milestones stated in the paper text: (time [s], position [m or km]).
P_T1 = [(270.0, 350.0), (820.0, 650.0), (1080.0, 850.0)]
P_T2 = [(300.0, 10.0), (500.0, 15.0), (600.0, 20.0),
        (1500.0, 29.97)]
