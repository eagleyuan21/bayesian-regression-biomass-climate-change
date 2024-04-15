from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


model = LinearRegression()

data = pd.read_csv("the_arctic_plant_aboveground_biomass_synthesis_dataset.csv", sep=",", encoding="ISO-8859-1")

data = data[data['biomass_density_gm2'].notnull()]

y = data["biomass_density_gm2"].to_numpy(copy=True)


X_year = data["year"].to_numpy(copy=True)
X_year = X_year - np.min(X_year)
X_year = np.reshape(X_year, (-1, 1))

map_plant, X_plant = np.unique(data["pft"].to_numpy(copy=True), return_inverse=True)
X_plant = np.reshape(X_plant, (-1, 1))

map_zone, X_zone = np.unique(data["bioclim_zone"].to_numpy(copy=True), return_inverse=True)
X_zone = np.reshape(X_zone, (-1, 1))

X_temp = data["mat_degC"].to_numpy(copy=True)
X_temp = np.reshape(X_temp, (-1, 1))

X_grow = data["gdd_degC"].to_numpy(copy=True)
X_grow = np.reshape(X_grow, (-1, 1))

X_rain = data["map_mm"].to_numpy(copy=True)
X_rain = np.reshape(X_rain, (-1, 1))

# add intercept
X_aug = np.concatenate((X_year, X_plant, X_zone, X_temp, X_grow, X_rain), axis=1)


model.fit(X_aug, y)

vals = [['c0', model.intercept_]]

for i in range(len(model.coef_)):
    row = ['c'+str(i+1), model.coef_[i]]
    vals.append(row)

vals.append(['r2', model.score(X_aug, y)])

df = pd.DataFrame(vals, columns=['variables', 'value'])
df.to_csv('out_frequentist.csv')
