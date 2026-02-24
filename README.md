# Linearization Scheme of Shallow Water Equations for Quantum Algorithms

This repository has code that was used to simulate the Shallow Water Equations (SWE) (from the Navier Stokes Equations) using a mapping (Carlemann Linearization) that theoretically allows the system to be solved using quantum algortihms and thus would prove to be a more efficient method of solving the SWE when quantum hardware becomes more developped. This repository contains files with the solution of the SWE after the mapping using analytical methods (to benchmark), as well as files with solving the SWE after the mapping using a quantum linear system solver based on quantum singular value transformation. 

## Installation
### Requirements

- Python (v3.13.1 or later)
- JupyterLab (v4.3.0 or later)

### Note

For the analytical solution of the SWE, in order to run the simulation with meaningful parameter sizes, we recommend using an high perfomrance computing cluster. In this project, a Linux-based, SLURM managed cluster with mixed CPU/GPU nodes was used. It was not possible to run the simulation with the parameters needed to get meaninful results for interpretation on macOS X 10.12 with 8 GB RAM.

## Usage

How to use it, with examples.

## Features

- Feature 1
- Feature 2

## Contributing

Guidelines for contributing.

## License

License information.
