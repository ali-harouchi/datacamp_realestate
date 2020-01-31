# Starting kit of the real estate price prediction RAMP challenge


## Getting started

This starting kit requires Python and the following dependencies:

* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `matplolib`
* `seaborn`
* `jupyter`
* `ramp-workflow`
* `geopandas`
* `json`
* `bokeh`


```
jupyter notebook Real_Estate_starting_kit.ipynb
```


## Advanced install using `conda` (optional)

We provide an `environment.yml` file which can be used with `conda` to
create a clean environment and install the necessary dependencies.

```
conda env create -f environment.yml
```

Then, you can activate the environment using:

```
source activate 
```

for Linux and MacOS. In Windows, use the following command instead:

```
activate REsk
```

For more information on the [RAMP](http:www.ramp.studio) ecosystem go to
[`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow).