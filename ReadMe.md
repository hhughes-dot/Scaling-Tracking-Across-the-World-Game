# Tracking animator
Tools for visualising different sources of tracking data. 

## Requirements
This project requires the following to be installed:
- python (version 3.10.12)

### Installation and Setup
Create virtual environment: 

```bash
python3 -m venv env
source env/bin/activate
```

For installing the module's python dependencies:
```bash
pip install .
```

For using the interactive submodule you'll need to explicitly enable the following jupyter extensions:  
- `jupyter nbextension enable --py --sys-prefix qgrid`  
- `jupyter nbextension enable --py --sys-prefix widgetsnbextension`  

### Project Organisation

    │   .gitignore                             <- boilerplate .gitignore
    │   pyproject.toml                         <- manages project requirements and metadata.
    │   ReadMe.md                              <- this ReadMe.
    │   requirements.txt                       <- python requirements.
    │
    ├───notebooks
    │   │   imputed_visualizer.ipynb           <- notebook for interactively visualizing imputed broadcast.
    |
    └───src
        │   basic_plot.py                      <- visualiser of individual frames of tracking data.
        │   interactive_animator.py            <- animates the BasicPlot.
        │   interactive_controller.py          <- allows control of BasicPlot.
        │   visualiser.py                      <- backend artists for visualisation.

