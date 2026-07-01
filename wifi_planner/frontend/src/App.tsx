import { useMemo, useRef, useState } from 'react';
import { FloorplanCanvas, type SignalHoverInfo } from './FloorplanCanvas';
import type { BuildingModel, CoverageResult } from './types';
import sampleModel from './fixtures/wifi_validation_model.json';
import * as api from './api';

type Tool = 'select' | 'roi' | 'wall' | 'window' | 'delete' | 'ap';
type StepKey = 'upload' | 'review' | 'plan' | 'report';
type PanelTab = 'setup' | 'tools' | 'wifi';

interface UploadedFileInfo {
  name: string;
  size: string;
  dimensions: string;
}

const toolLabels: Record<Tool, { title: string; help: string; icon: string }> = {
  select: { title: 'Move Wi‑Fi points', help: 'Drag a blue AP dot to improve coverage.', icon: '↔' },
  roi: { title: 'Select coverage area', help: 'Click two corners to limit AP planning to that rectangle.', icon: '▣' },
  wall: { title: 'Draw red wall line', help: 'Click the start and end of a missing wall.', icon: '╱' },
  window: { title: 'Add window', help: 'Click on/near a wall to mark a window.', icon: '▱' },
  delete: { title: 'Remove item', help: 'Click a wall, window, or blue AP marker to remove it.', icon: '⌫' },
  ap: { title: 'Add Wi‑Fi point', help: 'Click where you want a new access point.', icon: '+' },
};

const steps: { key: StepKey; title: string; description: string }[] = [
  { key: 'upload', title: 'Upload', description: 'Add your floorplan image' },
  { key: 'review', title: 'Review', description: 'Check red walls and room outlines' },
  { key: 'plan', title: 'Plan Wi‑Fi', description: 'Place APs and view heatmap' },
  { key: 'report', title: 'Report', description: 'Export the result' },
];

const MATERIAL_ATTENUATION: Record<string, number> = {
  drywall: 4,
  wood: 5,
  glass: 3,
  brick: 9,
  concrete: 12,
  reinforced_concrete: 18,
  metal: 25,
};

function emptyCoverage(model: BuildingModel, mode: 'preview' | 'full' = 'preview'): CoverageResult {
  const total = model.grid.rows * model.grid.cols;
  return {
    coverage_dbm: Array.from({ length: total }, () => -120),
    rows: model.grid.rows,
    cols: model.grid.cols,
    resolution_m: model.grid.resolution_m,
    mode,
  };
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function pointInPolygon(x: number, y: number, polygon: [number, number][]) {
  let inside = false;
  let j = polygon.length - 1;
  for (let i = 0; i < polygon.length; i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-12) + xi) inside = !inside;
    j = i;
  }
  return inside;
}

function loadImageDimensions(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth || img.width, height: img.naturalHeight || img.height });
    img.onerror = reject;
    img.src = url;
  });
}

function createBlankImageModel(fileName: string, imageWidthPx: number, imageHeightPx: number): BuildingModel {
  const now = new Date().toISOString();
  const longSideM = 42;
  const factor = longSideM / Math.max(imageWidthPx, imageHeightPx, 1);
  const width = Math.round(imageWidthPx * factor * 1000) / 1000;
  const height = Math.round(imageHeightPx * factor * 1000) / 1000;
  const cols = Math.max(1, Math.ceil(width));
  const rows = Math.max(1, Math.ceil(height));
  const cells = Array.from({ length: rows * cols }, (_, idx) => {
    const row = Math.floor(idx / cols);
    const col = idx % cols;
    return {
      center_m: [col + 0.5, row + 0.5] as [number, number],
      type: 'open' as const,
      attenuation_db: 0,
      room_id: 'r_uploaded_area',
      placeable: true,
    };
  });
  return {
    schema_version: '0.1.0',
    metadata: {
      source_image_ref: fileName,
      scale_m_per_px: factor,
      scale_confidence: 0.2,
      scale_method: 'manual',
      floor_dimensions_m: { width, height },
      created_at: now,
      last_modified_at: now,
    },
    walls: [],
    rooms: [
      {
        id: 'r_uploaded_area',
        boundary_polygon_m: [[0, 0], [width, 0], [width, height], [0, height]],
        area_m2: Math.round(width * height * 100) / 100,
        centroid_m: [width / 2, height / 2],
        label: 'unknown',
        occupancy_level: 'medium',
        adjacent_room_ids: [],
        confidence: 0.25,
        user_edited: false,
      },
    ],
    doors: [],
    windows: [],
    grid: {
      resolution_m: 1,
      origin_m: [0, 0],
      cols,
      rows,
      floor_id: 'floor_1',
      cells,
    },
    access_points: [],
    building_footprints: [
      {
        id: 'fp_uploaded_area',
        boundary_polygon_m: [[0, 0], [width, 0], [width, height], [0, height]],
        confidence: 0.2,
        user_edited: false,
      },
    ],
    analysis_regions: [],
  };
}

function StatCard({ label, value, tone = 'slate' }: { label: string; value: string | number; tone?: 'slate' | 'green' | 'amber' | 'blue' }) {
  const toneClass = {
    slate: 'from-slate-50 to-white text-slate-900 border-slate-200',
    green: 'from-emerald-50 to-white text-emerald-950 border-emerald-200',
    amber: 'from-amber-50 to-white text-amber-950 border-amber-200',
    blue: 'from-blue-50 to-white text-blue-950 border-blue-200',
  }[tone];
  return (
    <div className={`rounded-2xl border bg-gradient-to-br p-4 shadow-sm ${toneClass}`}>
      <div className="text-2xl font-bold">{value}</div>
      <div className="mt-1 text-xs font-medium uppercase tracking-wide opacity-70">{label}</div>
    </div>
  );
}

function HeatmapLegend() {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-2xl bg-white/95 px-3 py-2 text-xs font-semibold shadow-sm ring-1 ring-slate-200">
      <span className="text-slate-500">Heatmap:</span>
      <span className="inline-flex items-center gap-1"><i className="h-3 w-5 rounded bg-emerald-500/70" /> Strong</span>
      <span className="inline-flex items-center gap-1"><i className="h-3 w-5 rounded bg-lime-400/70" /> Good</span>
      <span className="inline-flex items-center gap-1"><i className="h-3 w-5 rounded bg-yellow-300/80" /> Fair</span>
      <span className="inline-flex items-center gap-1"><i className="h-3 w-5 rounded bg-sky-300/70" /> Weak edge</span>
    </div>
  );
}

export default function App() {
  const [model, setModel] = useState<BuildingModel>(sampleModel as BuildingModel);
  const [coverage, setCoverage] = useState<CoverageResult>(() => emptyCoverage(sampleModel as BuildingModel));
  const [tool, setTool] = useState<Tool>('select');
  const [heatmap, setHeatmap] = useState(true);
  const [confidence, setConfidence] = useState(true);
  const [busy, setBusy] = useState(false);
  const [activeStep, setActiveStep] = useState<StepKey>('upload');
  const [status, setStatus] = useState('Ready. Upload a floorplan image or try the sample already loaded on the screen.');
  const [report, setReport] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [backgroundImageUrl, setBackgroundImageUrl] = useState<string>('');
  const [uploadedFile, setUploadedFile] = useState<UploadedFileInfo | null>(null);
  const [imageOpacity, setImageOpacity] = useState(72);
  const [minSignalDbm, setMinSignalDbm] = useState(-67);
  const [hoverSignal, setHoverSignal] = useState<SignalHoverInfo | null>(null);
  const [dxfResult, setDxfResult] = useState<api.DxfHeatmapResult | null>(null);
  const dxfFileRef = useRef<HTMLInputElement | null>(null);
  const [panelTab, setPanelTab] = useState<PanelTab>('setup');
  const fileRef = useRef<HTMLInputElement | null>(null);

  const lowConfidence = useMemo(
    () => model.walls.filter((w) => w.confidence < 0.75).length + model.rooms.filter((r) => r.confidence < 0.75).length,
    [model],
  );

  const activeRegion = useMemo(() => (model.analysis_regions ?? []).find((region) => region.active), [model.analysis_regions]);

  const coveredPercent = useMemo(() => {
    const target = model.grid.cells
      .map((cell, i) => ({ cell, value: coverage.coverage_dbm[i] }))
      .filter(({ cell }) => cell.placeable)
      .filter(({ cell }) => !activeRegion || pointInPolygon(cell.center_m[0], cell.center_m[1], activeRegion.boundary_polygon_m));
    if (!target.length) return model.access_points.length ? 100 : 0;
    return Math.round((100 * target.filter(({ value }) => value >= minSignalDbm).length) / target.length);
  }, [activeRegion, coverage.coverage_dbm, minSignalDbm, model.access_points.length, model.grid.cells]);


  function handleModelChange(next: BuildingModel) {
    setModel(next);
  }


  async function upload(file?: File) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setStatus('Please upload an image file such as PNG or JPG.');
      return;
    }

    setBusy(true);
    setReport('');
    setDxfResult(null);
    setActiveStep('upload');
    const objectUrl = URL.createObjectURL(file);
    setBackgroundImageUrl((old) => {
      if (old.startsWith('blob:')) URL.revokeObjectURL(old);
      return objectUrl;
    });

    try {
      const dims = await loadImageDimensions(objectUrl);
      setUploadedFile({ name: file.name, size: formatBytes(file.size), dimensions: `${dims.width} × ${dims.height}px` });
      const imageOnlyModel = createBlankImageModel(file.name, dims.width, dims.height);
      setModel(imageOnlyModel);
      setCoverage(emptyCoverage(imageOnlyModel));
      setStatus('Image is visible on the workspace. Running simplified detection now...');

      const detected = await api.uploadFloorplan(file);
      setModel(detected);
      setCoverage(await api.computeCoverage(detected, false));
      setActiveStep('review');
      setStatus('Floorplan loaded. Red lines show detected/edited walls. Review the plan, then prepare Wi‑Fi planning.');
    } catch (error) {
      setActiveStep('review');
      setStatus(`The image is shown, but automatic backend detection did not finish. You can still trace walls and add APs manually. Details: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function rebuildGrid() {
    setBusy(true);
    try {
      setStatus('Preparing the floorplan for Wi‑Fi calculations...');
      const gridded = await api.buildGrid(model, model.grid.resolution_m || 0.5);
      setModel(gridded);
      setCoverage(await api.computeCoverage(gridded, false));
      setActiveStep('plan');
      setStatus('Floorplan is ready for Wi‑Fi planning.');
    } catch {
      setActiveStep('review');
      setStatus('Backend grid/coverage failed. Check that the backend is running.');
    } finally {
      setBusy(false);
    }
  }

  async function recommend() {
    setBusy(true);
    setReport('');
    try {
      setStatus(model.grid.cells.length === 0 ? `Preparing floorplan, then finding Wi‑Fi points for at least ${minSignalDbm} dBm...` : `Finding Wi‑Fi access point locations for at least ${minSignalDbm} dBm...`);
      const planningModel = model.grid.cells.length === 0 ? await api.buildGrid(model, model.grid.resolution_m || 1.0) : model;
      const result = await api.recommendPlacement(planningModel, minSignalDbm);
      setModel(result.model);
      setCoverage(result.coverage);
      setActiveStep('plan');
      setStatus('Backend recommendation complete. Blue dots are APs; drag any AP to fine tune coverage.');
    } catch (error) {
      setStatus(`Backend recommendation failed: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function optimizePso() {
    setBusy(true);
    setReport('');
    try {
      setStatus('Running backend PSO optimizer. This may take a few seconds...');
      const planningModel = model.grid.cells.length === 0 ? await api.buildGrid(model, model.grid.resolution_m || 1.0) : model;
      const result = await api.optimizePlacementPso(planningModel, minSignalDbm);
      setModel(result.model);
      setCoverage(result.coverage);
      setActiveStep('plan');
      setStatus('PSO optimization complete. APs are constrained to the building/ROI.');
    } catch (error) {
      setStatus(`PSO optimization failed: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function refreshCoverage(nextModel = model, preview = false) {
    try {
      const result = await api.computeCoverage(nextModel, preview);
      setCoverage(result);
    } catch (error) {
      setStatus(`Backend coverage failed: ${String(error)}`);
    }
  }

  async function makeReport() {
    setBusy(true);
    try {
      setReport(await api.exportReport(model, coverage, minSignalDbm));
    } catch {
      setReport(`Wi‑Fi Design Report\n====================\nSource image: ${model.metadata.source_image_ref}\nAccess points: ${model.access_points.length}\nCoverage target cells above ${minSignalDbm} dBm: ${coveredPercent}%\n\nGenerated locally because the backend report endpoint is unavailable.\n`);
    } finally {
      setActiveStep('report');
      setStatus('Report generated. You can copy it from the panel below the map.');
      setBusy(false);
    }
  }

  function loadSample() {
    if (backgroundImageUrl.startsWith('blob:')) URL.revokeObjectURL(backgroundImageUrl);
    setBackgroundImageUrl('');
    setUploadedFile(null);
    setModel(sampleModel as BuildingModel);
    setCoverage(emptyCoverage(sampleModel as BuildingModel));
    setActiveStep('review');
    setStatus('Sample floorplan model loaded. Use it to test AP placement and heatmap controls.');
  }

  function clearCoverageRegion() {
    const next = { ...model, analysis_regions: [] };
    setModel(next);
    setStatus('Coverage area cleared. AP recommendation will use all placeable cells again.');
  }

  function updateAllWallMaterial(material: string) {
    const attenuation = MATERIAL_ATTENUATION[material];
    if (attenuation === undefined) return;
    const next: BuildingModel = {
      ...model,
      walls: model.walls.map((wall) => ({ ...wall, material, attenuation_db: attenuation, user_edited: true })),
    };
    setModel(next);
    setStatus(`All walls set to ${material}. Click Refresh heatmap to recalculate.`);
  }

  async function uploadDxf(file?: File) {
    if (!file) return;
    setBusy(true);
    setDxfResult(null);
    try {
      setStatus('Uploading DXF and running FMM heatmap...');
      const result = await api.uploadDxfHeatmap(file, 10, true);
      setDxfResult(result);
      setStatus(result.ok ? 'DXF/FMM heatmap generated on backend.' : `DXF needs layer classification: ${result.error}`);
    } catch (error) {
      setStatus(`DXF/FMM failed: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    upload(e.dataTransfer.files?.[0]);
  }

  return (
    <div className="min-h-full bg-slate-100 text-slate-950">
      <header className="bg-gradient-to-r from-blue-700 via-indigo-700 to-slate-900 px-5 py-4 text-white shadow-lg">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="mb-1 inline-flex rounded-full bg-white/15 px-3 py-1 text-xs font-semibold ring-1 ring-white/20">Floorplan AI + Wi‑Fi Designer</div>
            <h1 className="text-2xl font-bold tracking-tight">Plan Wi‑Fi from a floorplan</h1>
            <p className="mt-1 max-w-3xl text-sm text-blue-50">Upload → select area → recommend APs → inspect heatmap.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button onClick={loadSample} className="rounded-2xl bg-white/10 px-5 py-3 text-sm font-bold text-white ring-1 ring-white/25 hover:bg-white/15">Try sample</button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-4 p-4">
        <input
          ref={fileRef}
          className="hidden"
          type="file"
          accept="image/png,image/jpeg,image/jpg,image/webp"
          onChange={(e) => { upload(e.target.files?.[0]); e.currentTarget.value = ''; }}
        />
        <input
          ref={dxfFileRef}
          className="hidden"
          type="file"
          accept=".dxf"
          onChange={(e) => { uploadDxf(e.target.files?.[0]); e.currentTarget.value = ''; }}
        />

        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="Walls" value={model.walls.length} />
          <StatCard label="Wi‑Fi points" value={model.access_points.length} tone="green" />
          <StatCard label={`Coverage ≥ ${minSignalDbm} dBm`} value={`${coveredPercent}%`} tone={coveredPercent >= 90 ? 'green' : coveredPercent >= 70 ? 'amber' : 'slate'} />
          <StatCard
            label="Mouse signal"
            value={hoverSignal?.dbm === null || !hoverSignal ? '—' : `${hoverSignal.dbm.toFixed(1)} dBm`}
            tone={hoverSignal?.dbm !== null && hoverSignal?.dbm !== undefined && hoverSignal.dbm >= -67 ? 'green' : hoverSignal?.dbm !== null && hoverSignal?.dbm !== undefined && hoverSignal.dbm >= -75 ? 'amber' : 'blue'}
          />
        </section>

        <section className="grid gap-4 xl:grid-cols-[330px_minmax(0,1fr)]">
          <aside className="rounded-3xl border border-slate-200 bg-white shadow-sm xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)] xl:overflow-y-auto">
            <div className="grid grid-cols-3 gap-1 border-b border-slate-200 p-2 text-xs font-bold">
              {([
                ['setup', 'Setup'],
                ['tools', 'Tools'],
                ['wifi', 'Wi‑Fi'],
              ] as [PanelTab, string][]).map(([key, label]) => (
                <button key={key} onClick={() => setPanelTab(key)} className={`rounded-xl px-2 py-2 ${panelTab === key ? 'bg-blue-600 text-white' : 'bg-slate-50 text-slate-700 hover:bg-slate-100'}`}>{label}</button>
              ))}
            </div>

            <div className="space-y-4 p-4">
              {panelTab === 'setup' && (
                <>
                  <section
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={onDrop}
                    className={`rounded-2xl border-2 border-dashed p-4 text-center transition ${dragOver ? 'border-blue-500 bg-blue-50' : 'border-slate-300 bg-white'}`}
                  >
                    {backgroundImageUrl ? (
                      <img src={backgroundImageUrl} alt="Uploaded floorplan preview" className="mx-auto h-28 w-full rounded-xl border border-slate-200 object-contain bg-slate-50" />
                    ) : (
                      <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-100 text-3xl">📄</div>
                    )}
                    <h2 className="mt-3 font-bold">Floorplan</h2>
                    <p className="mt-1 text-xs text-slate-500">Drop PNG/JPG here or choose a file.</p>
                    {uploadedFile && (
                      <div className="mt-3 rounded-xl bg-slate-50 p-2 text-left text-xs">
                        <div className="truncate font-bold text-slate-900">{uploadedFile.name}</div>
                        <div className="mt-1 text-slate-500">{uploadedFile.dimensions} · {uploadedFile.size}</div>
                      </div>
                    )}
                    <button onClick={() => fileRef.current?.click()} className="mt-3 w-full rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-bold text-white shadow hover:bg-blue-700 disabled:opacity-60" disabled={busy}>
                      {busy ? 'Processing...' : backgroundImageUrl ? 'Replace image' : 'Choose image'}
                    </button>
                    <button onClick={() => dxfFileRef.current?.click()} className="mt-2 w-full rounded-xl bg-slate-800 px-4 py-2.5 text-sm font-bold text-white shadow hover:bg-slate-700 disabled:opacity-60" disabled={busy}>
                      Upload DXF for FMM heatmap
                    </button>
                    {dxfResult && (
                      <div className="mt-3 rounded-xl bg-slate-50 p-2 text-left text-xs">
                        <div className="font-bold">DXF/FMM: {dxfResult.ok ? 'ready' : 'needs classification'}</div>
                        <div className="mt-1 text-slate-600">Layers: {dxfResult.layers?.length ?? 0} · Walls: {dxfResult.assigned_wall ?? 0} · Glass: {dxfResult.assigned_glass ?? 0}</div>
                        {dxfResult.rssi_min_dbm !== undefined && <div className="mt-1 text-slate-600">RSSI: {dxfResult.rssi_min_dbm.toFixed(1)} to {dxfResult.rssi_max_dbm?.toFixed(1)} dBm</div>}
                        {dxfResult.unclassified_layers?.length > 0 && <div className="mt-1 text-amber-700">Unclassified: {dxfResult.unclassified_layers.join(', ')}</div>}
                      </div>
                    )}
                  </section>

                  <section className="rounded-2xl bg-slate-50 p-3">
                    <h3 className="font-bold">Layers</h3>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                      <label className="flex items-center gap-2 rounded-xl bg-white p-3"><input type="checkbox" checked={confidence} onChange={(e) => setConfidence(e.target.checked)} /> Review</label>
                      <label className="flex items-center gap-2 rounded-xl bg-white p-3"><input type="checkbox" checked={heatmap} onChange={(e) => setHeatmap(e.target.checked)} /> Heatmap</label>
                    </div>
                    {backgroundImageUrl && (
                      <label className="mt-3 block text-sm font-semibold text-slate-700">
                        Image visibility
                        <input className="mt-2 w-full accent-blue-600" type="range" min="25" max="100" value={imageOpacity} onChange={(e) => setImageOpacity(Number(e.target.value))} />
                      </label>
                    )}
                  </section>

                  <section className="rounded-2xl border border-blue-100 bg-blue-50 p-3 text-sm">
                    <div className="font-bold text-blue-950">Coverage area</div>
                    <p className="mt-1 text-blue-800">{activeRegion ? `Active: ${activeRegion.label}` : 'No ROI selected. Using all interior grid cells.'}</p>
                    <div className="mt-3 flex gap-2">
                      <button className="rounded-xl bg-blue-600 px-3 py-2 text-xs font-bold text-white" onClick={() => { setTool('roi'); setPanelTab('tools'); }}>Draw area</button>
                      <button className="rounded-xl bg-white px-3 py-2 text-xs font-bold text-blue-700 ring-1 ring-blue-200" onClick={clearCoverageRegion}>Clear</button>
                    </div>
                  </section>
                </>
              )}

              {panelTab === 'tools' && (
                <section>
                  <h2 className="text-lg font-bold">Canvas tools</h2>
                  <div className="mt-3 grid gap-2">
                    {(Object.keys(toolLabels) as Tool[]).map((key) => (
                      <button key={key} onClick={() => setTool(key)} className={`flex items-center gap-3 rounded-2xl border p-3 text-left transition ${tool === key ? 'border-blue-400 bg-blue-50 ring-2 ring-blue-100' : 'border-slate-200 hover:bg-slate-50'}`}>
                        <span className={`flex h-10 w-10 items-center justify-center rounded-xl text-lg font-bold ${tool === key ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700'}`}>{toolLabels[key].icon}</span>
                        <span><span className="block font-semibold">{toolLabels[key].title}</span><span className="block text-xs text-slate-500">{toolLabels[key].help}</span></span>
                      </button>
                    ))}
                  </div>
                </section>
              )}

              {panelTab === 'wifi' && (
                <section>
                  <h2 className="text-lg font-bold">Wi‑Fi actions</h2>
                  <div className="mt-3 rounded-2xl bg-slate-50 p-3">
                    <div className="flex items-center justify-between gap-3">
                      <label className="text-sm font-bold text-slate-800" htmlFor="min-signal">Minimum signal</label>
                      <span className="rounded-xl bg-slate-900 px-3 py-1 text-sm font-bold text-white">{minSignalDbm} dBm</span>
                    </div>
                    <input id="min-signal" className="mt-3 w-full accent-emerald-600" type="range" min="-80" max="-50" step="1" value={minSignalDbm} onChange={(e) => setMinSignalDbm(Number(e.target.value))} />
                    <div className="mt-1 flex justify-between text-[11px] font-semibold text-slate-500"><span>-80</span><span>-67 typical</span><span>-50</span></div>
                  </div>
                  <label className="mt-3 block text-sm font-bold text-slate-700">
                    Wall material
                    <select className="mt-2 w-full rounded-xl border border-slate-200 bg-white p-3" onChange={(e) => updateAllWallMaterial(e.target.value)} defaultValue="">
                      <option value="" disabled>Choose material for all walls</option>
                      {Object.keys(MATERIAL_ATTENUATION).map((material) => <option key={material} value={material}>{material.replace('_', ' ')} ({MATERIAL_ATTENUATION[material]} dB)</option>)}
                    </select>
                  </label>
                  <div className="mt-3 grid gap-2">
                    <button className="rounded-2xl bg-slate-800 px-4 py-3 font-bold text-white hover:bg-slate-700 disabled:opacity-60" onClick={rebuildGrid} disabled={busy}>Prepare floorplan</button>
                    <button className="rounded-2xl bg-emerald-600 px-4 py-3 font-bold text-white hover:bg-emerald-700 disabled:opacity-60" onClick={recommend} disabled={busy}>Recommend Wi‑Fi points</button>
                    <button className="rounded-2xl bg-indigo-600 px-4 py-3 font-bold text-white hover:bg-indigo-700 disabled:opacity-60" onClick={optimizePso} disabled={busy}>Optimize APs with PSO</button>
                    <button className="rounded-2xl bg-amber-500 px-4 py-3 font-bold text-white hover:bg-amber-600 disabled:opacity-60" onClick={() => refreshCoverage(model, false)} disabled={busy}>Refresh heatmap</button>
                    <button className="rounded-2xl bg-violet-600 px-4 py-3 font-bold text-white hover:bg-violet-700 disabled:opacity-60" onClick={makeReport} disabled={busy}>Create report</button>
                  </div>
                </section>
              )}
            </div>
          </aside>

          <section className="space-y-4">
            <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-3 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-xl font-bold">{dxfResult?.heatmap_image_data_url ? 'DXF/FMM Heatmap' : 'Floorplan workspace'}</h2>
                  <p className="text-sm text-slate-500">{dxfResult?.heatmap_image_data_url ? 'Vector DXF parsed on backend; Fast Marching Method heatmap shown below.' : <>Tool: <b>{toolLabels[tool].title}</b>. Uploaded image is the base layer; APs and heatmap are overlays.</>}</p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <HeatmapLegend />
                  <div className="rounded-2xl bg-slate-100 px-4 py-2 text-sm font-medium text-slate-700">{coverage.mode === 'preview' ? 'Fast preview' : 'Full view'}</div>
                </div>
              </div>
              <div className="flex justify-center overflow-hidden rounded-2xl bg-slate-100 p-3 ring-1 ring-slate-200">
                <FloorplanCanvas
                  model={model}
                  coverage={coverage}
                  tool={tool}
                  heatmap={heatmap}
                  showConfidence={confidence}
                  backgroundImageUrl={backgroundImageUrl}
                  imageOpacity={imageOpacity / 100}
                  onChange={handleModelChange}
                  onSignalHover={setHoverSignal}
                  onPreviewDrag={(m) => refreshCoverage(m, true)}
                  onDragEnd={(m) => refreshCoverage(m, false)}
                />
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="font-bold">Current status</h2>
              <p className="mt-1 text-sm text-slate-600">{status}</p>
            </div>

            {report && (
              <div className="rounded-3xl border border-violet-200 bg-white p-4 shadow-sm">
                <div className="mb-2 flex items-center justify-between">
                  <h2 className="text-lg font-bold">Report</h2>
                  <button className="rounded-xl bg-violet-100 px-3 py-2 text-sm font-bold text-violet-700" onClick={() => navigator.clipboard?.writeText(report)}>Copy</button>
                </div>
                <pre className="max-h-72 overflow-auto rounded-2xl bg-slate-950 p-4 text-xs text-slate-100">{report}</pre>
              </div>
            )}
          </section>
        </section>
      </main>
    </div>
  );
}
