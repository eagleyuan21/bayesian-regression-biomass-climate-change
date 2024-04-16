import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("the_arctic_plant_aboveground_biomass_synthesis_dataset.csv", sep=",", encoding="ISO-8859-1")

data = data[data['biomass_density_gm2'].notnull()]
#data = data[data['biomass_density_gm2'] != 0]

y = data["biomass_density_gm2"].to_numpy(copy=True)

y = np.log(y + 1)

minimum = int(min(y))
maximum = int(max(y)) + 1
binwidth = 1

plt.hist(y, bins=range(minimum, maximum + binwidth, binwidth))
plt.title("Biomass Density data distribution with logged density")
plt.xlabel("Biomass Density +1 and ln to avoid ln(0) (gm^2)")
plt.ylabel("Frequency")
plt.xlim(-1,20)
plt.savefig("density_log.jpg", bbox_inches='tight',dpi=100)
