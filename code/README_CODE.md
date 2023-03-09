# Code for UK Labour Flow Network Model

This README file contains descriptions of all scripts used to run simulations, and to generate datasets required to run simulations. Throughout we use LFS to refer to the UK Labour Force Survey.

To run a set of simulations, make all required parameter selections (detailed within each script) and launch the script. The expected runtime varies between scripts, but an individual simulation with 3500 agents should run in approximately 1 minute.

## Folder Structure

```               
├── Classifier.ipynb        <- Classifies age distributions based on whether their structure is monotonic decreasing with age
├── Model1.ipynb            <- Calculates parameters for, and simulates, model 1   
├── Model1.ipynb            <- Calibrates parameters for, and simulates, model 2 
├── Model1_fitted.ipynb     <- Fits monotonic decreasing distribution to observed distribution, then calculates parameters for, and simulates, model 1 for this fitted distribution
└──
```
## Run sequence

Below we detail the suggested run order for these scripts.

1. **LFS data collection:** this should *always* be run before non-LFS data collection.
    1.  [ActivationRateAnalysis.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/preprocessing/LFS%20data%20collection/ActivationRateAnalysis.py)
    2. [IndividualAttributesAnalysis.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/preprocessing/LFS%20data%20collection/IndividualAttributesAnalysis.py)
    3. [JobDistributionAnalysis.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/preprocessing/LFS%20data%20collection/JobDistributionAnalysis.py)
    4. [TransitionMatrixGeneration.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/preprocessing/LFS%20data%20collection/TransitionMatrixGeneration.py)
2. **Non-LFS data collection:** should *always* be run after LFS data collection, and before pre-simulation processing, as scripts depend on files produced during the LFS data collection step.
    1. [region_similarity.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/preprocessing/non-LFS%20data%20collection/region_similarity.py)
    2. [sic_similarity.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/preprocessing/non-LFS%20data%20collection/sic_similarity.py)
    3. [soc_skillgetter.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/preprocessing/non-LFS%20data%20collection/soc_skillgetter.pyy)
    4. [soc_similarity.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/preprocessing/non-LFS%20data%20collection/soc_similarity.py) - should *always* be run after soc_skillgetter.py, as dependent on a file produced by that script.
3. **Pre-simulation processing:** should *always* be run after both data collection steps, as scripts depend on files produced at those stages.
    1. [DataReweighter.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/preprocessing/pre-simulation%20processing/DataReweighter.py)
    2. [ExpandedSimilarityMatrixGeneration.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/preprocessing/pre-simulation%20processing/ExpandedSimilarityMatrixGeneration.py)

### Simulation

1. **Calibration:** run one of [Calibration.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/simulation/Calibration.py), [Calibration.ipynb](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/simulation/Calibration.ipynb) - both perform calibration routine.
2. **Basic simulation:** run one of [BasicSimulation.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/simulation/BasicSimulation.py), [BasicSimulation.ipynb](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/simulation/BasicSimulation.ipynb) - both run a single simulation using the parameters calibrated in the previous step. This can be used as a quick ''sense check'' on the results of the calibration procedure.
3. **Shock simulation:** run one of [ShockSimulation.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/simulation/ShockSimulation.py), [ShockSimulation.ipynb](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/simulation/ShockSimulation.ipynb) - both run a set of simulations where a shock has been introduced.

[ABMrun.py](https://github.com/alan-turing-institute/UK-LabourFlowNetwork-Model/blob/main/code/simulation/ABMrun.py) is not run as a standalone, but is called within the abovementioned simulation scripts.

## File Dependencies

The following files, produced during the pre-processing stage, should be placed in the data folder before running any simulations or calibration, as they are necessary inputs. The placeholders bracketed with {} are defined within the scripts.

- activation_dict.txt
- income_dict_LFS_{regvar}_{sicvar}_{socvar}.txt
- region_transitiondensity_empirical_LFS_{regvar}_{sicvar}_{socvar}.csv
- sic_transitiondensity_empirical_LFS_{regvar}_{sicvar}_{socvar}.csv
- soc_transitiondensity_empirical_LFS_{regvar}_{sicvar}_{socvar}.csv
- reg_expanded_similaritymat_LFS.sav
- sic_expanded_similaritymat_LFS.sav
- soc_expanded_similaritymat_LFS.sav
- positiondist_reweighted_LFS_{regvar}_{sicvar}_{socvar}.csv
- age_dist_reweighted_LFS_{regvar}_{sicvar}_{socvar}.csv
- consumptionpref_dist_reweighted_LFS_{regvar}_{sicvar}_{socvar}.csv

**Note:** in order to run BasicSimulation.py/.ipynb or ShockSimulation.py/.ipynb you will also require the files (generated using Calibration.py/.ipynb) containing the calibrated parameters, namely:
- graddescent_N{N}_reps{sim_num}_GDruns{fitrun_num}_ssthresh{ss_threshold}_nus_reg_scost_mat_LFS.sav
- graddescent_N{N}_reps{sim_num}_GDruns{fitrun_num}_ssthresh{ss_threshold}_nus_sic_scost_mat_LFS.sav
- graddescent_N{N}_reps{sim_num}_GDruns{fitrun_num}_ssthresh{ss_threshold}_nus_soc_scost_mat_LFS.sav

All other required files are provided in the data/required folder of this repository.
