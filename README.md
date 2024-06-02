[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Testing](https://github.com/allfed/allfed-repository-template/actions/workflows/testing.yml/badge.svg)

# WACCM4-Nuclear-Winter-Analysis
This repo was set up to analyze the nuclear winter climate model of [Coupe et al. 2019](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JD030509) to look at the impact of a nuclear winter on renewable energy production.

## Codebase orientation
* `scripts/Explore.ipynb` queries the Coupe et al. montly average data to make some useful figures. Also a good notebook to explore climate simulation data unrelated to this particular project.
* `scripts/Solar-Power.ipynb` combines the Coupe et al. data with a database of solar farms to quantify the effect of a nuclear winter on solar power production.
* `scripts/Wind-Power.ipynb` combines the Coupe et al. data with a database of wind farms to quantify the effect of a nuclear winter on wind power production.
* `scripts/Maps-2050.ipynb` makes the figures actually used in the paper.
* `src/analysis.py` contains the main code for this project.

## Installation
1. Clone the repo on your local machine.
2. Create the Conda environment using `conda env create -f environment.yml`.
3. Activate the new environment using `conda activate WACCM4-Nuclear-Winter-Analysis`.
4. Download and unzip the Coupe et al. data in the `data` directory. It can be downloaded [here](https://figshare.com/articles/dataset/WACCM4_150_Tg_US-Russia/7742735/2).

## Sources
In addition to the Coupe et al. data, the `data` directory now contains a database of solar farms and wind farms, downloaded from `https://globalenergymonitor.org/`. The data contained in `data/wind-data` has been calculated in the `Climate-Data-Importing` repo. It consists of wind speed data and wind power data. Both data sets are averaged monthly from data with a 3-hour resolution. Wind power is calculated by passing the 3-hour resolution wind speed data through a typical wind turbine speed-power response function. Unlike solar power (which responds linearly to solar flux), wind power must be pre-processed at a higher time resolution because of the non-linear response to wind speed.

`pct_fed_intmodel_simpleadaptations_rationing_humans.txt` contains data on the fraction of people fed in each country based on the example shown in Figure 2 of the integrated model paper ([version 3](https://docs.google.com/document/d/1EyUinpkLkMwVa3Ve-q1igVT8jqj_vcps-8bxQQYM1Ak/edit)).