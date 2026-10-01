# MuFinder + MACE
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jollactic/MuFinder_MACE/blob/main/notebooks/MuFinder_MACE_tutorial.ipynb)


A lightweight workflow for exploring possible positive-muon (μ⁺) stopping
sites in solids and molecular materials using **MuFinder** for candidate-site
generation and **MACE** machine-learning interatomic potentials for structural
relaxation.

The aim is to provide a simple workflow that can be explored interactively on
small systems, for example in Google Colab, while using the same script for
larger calculations on a workstation or HPC system.

## Overview

The workflow is

    input structure
          |
          v
    optimize pristine host
       (atoms + cell)
          |
          v
    optional supercell
          |
          v
    generate candidate μ⁺ sites
       using MuFinder
          |
          v
    relax candidate sites
       using MACE
          |
          v
    cluster equivalent sites
          |
          v
    optional harmonic μ⁺ ZPE
          |
          v
    ranked insertion energies

The muon is represented by H when evaluating the interatomic potential.
Importantly, its atom index is stored explicitly, so the host material may
itself contain hydrogen.

For the optional vibrational calculation, the H surrogate is assigned the
physical positive-muon mass.

---

## Quick start

A calculation requires

- `mufinder_mace.py`
- an input structure such as a CIF file
- a small input file, for example `muon.in`

A minimal input might be

```ini
structure = structure.cif

level = full
supercell = 1 1 1

n_sites = 10

calculator = mace-mp
model = medium
dispersion = true

device = cuda
dtype = float64

zpe = true
