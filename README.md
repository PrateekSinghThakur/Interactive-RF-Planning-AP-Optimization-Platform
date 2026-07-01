# Interactive RF Planner for Enterprise Wi-Fi Deployment

> **🚧 Status: Under Active Development**

An interactive RF planning platform for enterprise Wi-Fi deployment using **DXF floorplans**, **FastAPI**, **React**, the **Fast Marching Method (FMM)**, and **Particle Swarm Optimization (PSO)**.

The goal of this project is to automate enterprise Wi-Fi planning by converting CAD floorplans into intelligent RF simulations and recommending optimal Access Point (AP) placements.

---

## Overview

Traditional enterprise Wi-Fi planning is often a manual process involving floorplan inspection, signal estimation, and iterative AP placement.

This project aims to automate that workflow by combining computational geometry, numerical PDE solvers, optimization algorithms, and interactive visualization into a unified platform.

The complete pipeline includes:

- DXF Floorplan Parsing
- Automatic Geometry Extraction
- Material-aware RF Environment Modeling
- Fast Marching Method (FMM) Signal Propagation
- Particle Swarm Optimization (PSO) for AP Placement
- Interactive React-based Floorplan Visualization
- FastAPI Backend Services

---

## 🚧 Development Status

This repository is currently under active development.

Many components are functional, while others are being refactored and integrated into the final architecture.

Current development focuses on:

- Improving CAD parsing
- Better wall/material classification
- Faster propagation simulation
- Advanced AP optimization
- Interactive frontend visualization
- API integration

---

# Proof of Concept

A working Proof of Concept (PoC) demonstrating the mathematical pipeline is available.

Please check:

```
notebooks/
    rf_planner_proof_of_concept.ipynb
```

(or the equivalent notebook in this repository)

The PoC demonstrates:

- DXF parsing
- Automatic geometry extraction
- Speed map generation
- Fast Marching Method propagation
- Heatmap visualization
- Particle Swarm Optimization for AP placement

This notebook provides a high-level overview of the project's core mathematical and engineering concepts before they are fully integrated into the application.

---

# Architecture

```
          DXF Floorplan
                 │
                 ▼
         CAD Geometry Parser
                 │
                 ▼
      Geometry & Material Detection
                 │
                 ▼
          Grid / Speed Map Builder
                 │
                 ▼
     Fast Marching Method (FMM)
                 │
                 ▼
        RF Signal Heatmap Engine
                 │
                 ▼
 Particle Swarm Optimization (PSO)
                 │
                 ▼
     Optimal Access Point Placement
                 │
                 ▼
        FastAPI Backend API
                 │
                 ▼
         React Frontend Viewer
```

---

# Technologies

### Backend

- Python
- FastAPI
- NumPy
- SciPy
- OpenCV
- ezdxf
- scikit-fmm

### Frontend

- React
- TypeScript
- TailwindCSS
- Vite

### Algorithms

- Fast Marching Method (FMM)
- Particle Swarm Optimization (PSO)
- Computational Geometry
- Distance Transform
- CAD Parsing

---

# Repository Structure

```
backend/
frontend/
notebooks/
scripts/
tests/

README.md
requirements.txt
```

---

# Planned Features

- Automatic wall and obstacle detection
- Multi-floor building support
- Material-aware attenuation models
- Interactive AP placement
- Coverage optimization
- Capacity-aware planning
- Interference visualization
- Heatmap export
- CAD annotation
- API-based deployment planning

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/interactive-rf-planner.git
cd interactive-rf-planner
```

Install dependencies

```bash
pip install -r requirements.txt
```

Frontend

```bash
cd frontend
npm install
npm run dev
```

Backend

```bash
uvicorn backend.app.api:app --reload
```

---

# Disclaimer

This project is currently in the research and development phase.

Some modules are experimental and subject to significant architectural changes.

---

# Contributing

Contributions, suggestions, and discussions are welcome.

Feel free to open an issue or submit a pull request.

---

# License

This project is licensed under the MIT License.

---

## Author

**Pratee**

B.Tech Mathematics & Computing

Research Interests:
- Enterprise Networking
- RF Planning
- Optimization Algorithms
- Computer Vision
- Computational Geometry
- AI-assisted Network Design
