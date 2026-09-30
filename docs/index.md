---
layout: default
title: MuFinder + MACE
---

# MuFinder + MACE

### Positive-muon site exploration with machine-learning interatomic potentials

**MuFinder + MACE** is a lightweight workflow for exploring possible
positive-muon (μ⁺) stopping sites in solids and molecular materials.

MuFinder generates candidate interstitial sites, while MACE machine-learning
interatomic potentials are used to relax the structures and compare their
energies.

The workflow is designed to be simple enough for interactive exploration while
using the same underlying script for larger calculations on HPC systems.

---

## Try it in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](COLAB_URL)

The Colab tutorial walks through a small calculation from the initial crystal
structure to relaxed muon sites and their relative energies.

No local installation is required for the tutorial.

---

## The workflow

```text
Initial structure
       │
       ▼
Optimize pristine host
   (atoms + cell)
       │
       ▼
Optional supercell
       │
       ▼
Generate candidate μ⁺ sites
       │
       ▼
Relax with MACE
       │
       ▼
Cluster equivalent sites
       │
       ▼
Optional μ⁺ zero-point energy
       │
       ▼
Rank candidate sites
