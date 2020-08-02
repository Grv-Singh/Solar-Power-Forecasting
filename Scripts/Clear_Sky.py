import itertools
import matplotlib.pyplot as plt
import pandas as pd
import pvlib
from pvlib import clearsky, atmosphere
from pvlib.location import Location

abq = Location(52.37745,9.7330013, altitude=55.67)
times = pd.DatetimeIndex(start='2018-03-16 10:30:00', tz='Etc/GMT+2', periods=8, freq='180min')
cs = abq.get_clearsky(times)
ghi = cs['ghi']*.953
ghi['2018-03-16 10:42:00':'2018-03-16 10:44:00'] = [500, 300, 400]
ghi['2018-03-16 10:56:00'] = 950
fig, ax = plt.subplots()
ghi.plot(label='input');
cs['ghi'].plot(label='ineichen clear');
ax.set_ylabel('Irradiance $W/m^2$');
plt.legend(loc=4);
plt.show();