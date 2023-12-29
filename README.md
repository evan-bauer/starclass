Starclass

These are the various algorithms I wrote during my time as part of the WOCS research team. 

The purpose of this set of constructors and basic programs is to process the large dataset of regularly measured intensities of stars in open clusters. For each star, three algorithms are utilized to determine whether a star exhibits periodic behavior and to determine the three most likely periods of variable brightness. From these estimates, the algorithm will create a `Star` object using the `starclass` constructor and store individual star's data in a comprehensive dictionary object. With a dictionary of stars and their most likely periods, the `RLC` file provides various methods of analyzing the periodic behavior and returns various calculated parameters describing the periodicity of the periodic variations from the raw data. The calculated features then can be used to train a machine-learning algorithm to classify stars based on the type of periodic variations they exhibit.

A more detailed walkthrough can be found in `StaRLC_usage.ipynb`.

NOTE: When I last tried to run some of the algorithms from the start, I discovered that the `astrobase` library is incompatible with the versions of Python after 3.7
