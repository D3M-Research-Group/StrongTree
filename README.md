# Strong Optimal Classification Trees

![Screenshot](logo.png)

Code for the paper ''[Strong Optimal Classification Trees](https://sites.google.com/view/sina-aghaei/home)''.

The code will run in python3 ( version 6 and higher) and require [Gurobi9.x](https://www.gurobi.com/downloads/gurobi-optimizer-eula/) solver. The required packages are listed in `requirements.txt`.

***

## Overview

The content of this repository is as follows:

- `Code` contains the implementation of approaches **FlowOCT** and **BendersOCT**:

    - `formulations` includes the formulation of each approach in gurobipy.

    - `experiments` contains the replication file for each approach which is resposible for creating and solving the problem and producing result

    - `utils` contains
      - `Tree` which creates the binary tree
      - `logger` logs the output of console into a text file
- `Results` contains the experiments raw results.

- `plots` contains the R script for generating figures and tables from the tabular results.

- `DataSets` contains all datasets used for the experiments in the paper


***

## How to Run the Code

- First install the required packages in the `requirements.txt`
- For  running an instance of FlowOCT (BendersOCT) run the main function in `FlowOCTReplication.py` (`BendersOCTReplication.py`) with the following arguments
    - f : name of the dataset
    - d : maximum depth of the tree
    - t : time limit
    - l : value of _lambda
    - i : sample (for shuffling and spliting data)
    - c : 1 if we do calibration; in this case we train our model on 50% of the data otherwise we train on 75% of the data

You can call the main function within a python file as follows (This is what we do in `./experiments/run_exp.py`

```python
from experiments import FlowOCTReplication
FlowOCTReplication.main(["-f", 'monk1_enc.csv', "-d", 2, "-t", 3600, "-l", 0, "-i", 1, "-c", 1])
```
or you could run it from terminal    

```bash
python3 FlowOCTReplication.py -f monk1_enc.csv -d 2 -t 3600 -l 0 -i 1 -c 1
```

The result of each instance would get stored in a csv file in `./Results/`.

Remember that in the replication files we assume that the name of the class label column is `target`. If you want to use a dataset of your own choice,
change the hard-coded label name to the desired name.




