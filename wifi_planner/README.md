# Intelligent Floorplan Understanding and Wi-Fi Network Design Platform

This repository implements the v0.1 foundation for an interactive platform that converts floorplan images into editable Building Model JSON, then uses that model for Wi-Fi AP placement and coverage heatmaps.

## Architecture rule

All domain modules communicate through the canonical `building_model.schema.json` artifact. Modules are replaceable behind public interfaces:

- Detection: `detect(image_bytes) -> partial structural model`
- Scale: scale metadata writers only
- Grid builder: `BuildingModel -> BuildingModel` with `grid.cells`
- Propagation: `grid + access_points -> coverage_dbm[]`
- Placement: `BuildingModel + injected scorer -> BuildingModel` with recommended APs
- Wi-Fi app layer: orchestration, heatmap colors, AP dragging, report formatting

## Schema gate

The schema was built first and validated against five future-application fixtures:

```bash
python scripts/validate_models.py validation_models
```

Expected output: five `PASS` lines.

## Backend

Install dependencies and run FastAPI:

```bash
python -m pip install -r requirements.txt
python -m backend.app
```

Endpoints:

- `GET /schema`
- `POST /validate`
- `POST /detect` multipart image upload
- `POST /grid`
- `POST /coverage`
- `POST /placement`
- `POST /projects` (optional PostgreSQL JSONB persistence via `DATABASE_URL`)
- `GET /projects/{project_id}`
- `POST /report`
- `POST /rf/dxf-heatmap` DXF → speed map → FMM heatmap endpoint

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The React/TypeScript canvas editor starts from the Wi-Fi validation model. Coverage, AP recommendation, PSO optimization, reports, and DXF/FMM processing are backend-driven; run the FastAPI backend before using the frontend.

## Tests

```bash
pytest -q
python scripts/benchmark_propagation.py --runs 10
cd frontend && npm run build
```

## End-to-end RF paths

### Raster web-app path

```text
PNG/JPG upload → simplified footprint/wall detector → footprint grid → backend coverage → AP placement/PSO → heatmap
```

### DXF/FMM path

```text
DXF upload → layer/material parser → speed map → Fast Marching Method → RSSI heatmap preview
```

The frontend exposes DXF upload in the Setup tab and PSO optimization in the Wi‑Fi tab. The standalone POC script is still available:

```bash
python scripts/poc_wifi_planner_v2.py "uploads/your-floorplan.jpeg" --out poc_outputs
```

The DXF/FMM CLI path is:

```bash
python scripts/run_dxf_to_fmm_demo.py path/to/floorplan.dxf --out outputs/dxf_fmm --res 8
```

## Current v0.1 scope

Implemented:

- Canonical JSON Schema + CI workflow
- Five validation models for Wi-Fi, CCTV, fire safety, IoT, navigation
- Simplified raster detector: building footprint + wall obstacles only
- DXF parser with layer-based material mapping
- Eikonal/FMM RF solver and DXF-to-heatmap pipeline
- Manual spline tracing, grouped material editing, and PSO backend modules
- Manual/reference-object scale extraction functions
- Compact web workflow with drag/drop PNG/JPG upload, visible floorplan background, ROI drawing, wall/window/AP editing, backend heatmap, backend greedy placement, backend PSO optimization, DXF/FMM upload summary, material assignment for walls, dBm hover, and copyable report
- Grid rasterizer with footprint-based domain and row-major cells
- Backend propagation engine for raster grid coverage
- Greedy AP placement and PSO AP optimization endpoints
- FastAPI orchestration, optional PostgreSQL JSONB storage, report endpoint, and DXF/FMM heatmap endpoint

Deferred production hardening:

- Full visual DXF/FMM heatmap rendering in the React canvas instead of API summary only
- Calibration of FMM speed-map material parameters against measurements
- Robust door/window vector classification
- Full evaluation dataset and tier-specific metrics
