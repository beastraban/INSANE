# INSANE
INflationary potential Simulator and ANalysis Engine


This package takes either a numeric inflationary potential or a symbolic one, calculates the background evolution (Friedmann's eqs.) and then, using the Mukhanov-Sasaki equations calculate the primordial power spectrum it yields.

Finally the package can analyse the results to extract the spectral index n_s, the index running alpha, the running of running and possibly higher moments.

The package contains 2 main modules: BackgroundSolver solves the background equations, and the MsSolver module solves and analyses the MS equations.
