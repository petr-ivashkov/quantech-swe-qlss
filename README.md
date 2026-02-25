# Linearization Scheme of Shallow Water Equations for Quantum Algorithms

This repository has code that was used to simulate the Shallow Water Equations (SWE) (from the Navier Stokes Equations) using a mapping (Carlemann Linearization) that theoretically allows the system to be solved using quantum algortihms and thus would prove to be a more efficient method of solving the SWE when quantum hardware becomes more developped. This repository contains files with the solution of the SWE after the mapping using analytical methods (to benchmark), as well as files with solving the SWE after the mapping using a quantum linear system solver based on quantum singular value transformation. 

## Installation
### Requirements

- Python (v3.13.1 or later)
- JupyterLab (v4.3.0 or later)

### Note

For the analytical solution of the SWE, in order to run the simulation with meaningful parameter sizes, we recommend using a high perfomrance computing cluster. In this project, a Linux-based, SLURM managed cluster with mixed CPU/GPU nodes was used. It was not possible to run the simulation with the parameters needed to get meaninful results for interpretation on macOS X 10.12 with 8 GB RAM.

## Authors

- Till Appel
- Zofia Binczyk
- Francesco Conoscenti
- Petr Ivashkov

## License

This project is licensed under the MIT License – see the LICENSE file for details.
