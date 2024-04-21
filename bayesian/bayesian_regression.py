import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

data = pd.read_csv("../original_source/the_arctic_plant_aboveground_biomass_synthesis_dataset.csv", sep=",", encoding="ISO-8859-1")

data = data[data['biomass_density_gm2'].notnull()]
#data = data[data['biomass_density_gm2'] != 0]

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
X_aug = np.concatenate((np.ones((X_year.shape[0], 1)), X_year, X_plant, X_zone, X_temp, X_grow, X_rain), axis=1)


n, p = X_aug.shape


with pm.Model() as m2d:
    X_data = pm.Data("X", X_aug, mutable=True)

    # priors
    beta = pm.Normal("beta", mu=0, sigma=1000, shape=(X_aug.shape[1]))

    mu = -1 / pm.math.dot(X_data, beta)

    # likelihood
    pm.Exponential("likelihood", lam=mu, observed=y)
    
    trace = pm.sample(1000, cores=1)
    ppc = pm.sample_posterior_predictive(trace)

y_pred = ppc.posterior_predictive.stack(sample=("chain", "draw"))["likelihood"].values.T
az.r2_score(y, y_pred).to_csv('zeros/r2.csv')

az.summary(trace, hdi_prob=0.95, kind='stats').to_csv('zeros/out.csv', index=True)

tree = pm.model_to_graphviz(m2d)
tree.render(filename='model_visual',format='jpg')

with m2d:
    idata = pm.sample(500, cores = 1)

az.plot_trace(idata)
fig = plt.gcf()
fig.savefig("zeros/out_plots_vars.jpg")

with m2d:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

az.plot_ppc(idata)
fig = plt.gcf()
fig.savefig("zeros/out_plots_post.jpg")
