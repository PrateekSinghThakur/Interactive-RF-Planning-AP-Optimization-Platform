import { useEffect, useRef, useState } from 'react';
import type { BuildingModel, Coord, CoverageResult, Wall, WindowElement } from './types';

type Tool = 'select' | 'roi' | 'wall' | 'window' | 'delete' | 'ap';

export interface SignalHoverInfo {
  position_m: Coord;
  dbm: number | null;
  label: string;
  cellType?: string;
}

interface Props {
  model: BuildingModel;
  coverage?: CoverageResult;
  tool: Tool;
  showConfidence: boolean;
  heatmap: boolean;
  backgroundImageUrl?: string;
  imageOpacity?: number;
  onChange: (model: BuildingModel) => void;
  onPreviewDrag?: (model: BuildingModel) => void;
  onDragEnd?: (model: BuildingModel) => void;
  onSignalHover?: (hover: SignalHoverInfo | null) => void;
}

function colorForDbm(dbm: number): string {
  if (dbm <= -110) return 'rgba(0,0,0,0)';
  if (dbm < -82) return 'rgba(59,130,246,0.30)';
  if (dbm < -72) return 'rgba(56,189,248,0.34)';
  if (dbm < -67) return 'rgba(250,204,21,0.42)';
  if (dbm < -55) return 'rgba(132,204,22,0.38)';
  return 'rgba(34,197,94,0.42)';
}

function distPoint(a: Coord, b: Coord) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function fsplDb(distanceM: number, freqGhz: number) {
  const d = Math.max(distanceM, 1);
  return 32.44 + 20 * Math.log10(d / 1000) + 20 * Math.log10(freqGhz * 1000);
}

function estimateDbmAtPoint(model: BuildingModel, p: Coord, fallbackAttenuation = 0) {
  if (!model.access_points.length) return null;
  let best = -120;
  for (const ap of model.access_points) {
    const distance = distPoint(p, ap.position_m);
    const value = ap.tx_power_dbm - fsplDb(distance, ap.freq_ghz) - fallbackAttenuation * 1.15 - Math.min(distance * 0.18, 10);
    best = Math.max(best, value);
  }
  return Math.round(best * 10) / 10;
}

function signalLabel(dbm: number | null) {
  if (dbm === null) return 'No AP signal';
  if (dbm <= -110) return 'No signal';
  if (dbm >= -55) return 'Excellent';
  if (dbm >= -67) return 'Good';
  if (dbm >= -75) return 'Fair';
  return 'Weak';
}

function wallConfidenceColor(confidence: number) {
  if (confidence >= 0.9) return '#16a34a';
  if (confidence >= 0.7) return '#eab308';
  if (confidence >= 0.5) return '#f97316';
  return '#ef4444';
}

function pointInPolygon(x: number, y: number, polygon: Coord[]) {
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

function distanceToSegment(p: Coord, a: Coord, b: Coord) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  if (dx === 0 && dy === 0) return distPoint(p, a);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)));
  return distPoint(p, [a[0] + t * dx, a[1] + t * dy]);
}

function drawRoundedLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number) {
  ctx.font = 'bold 12px Inter, system-ui, sans-serif';
  const paddingX = 7;
  const width = ctx.measureText(text).width + paddingX * 2;
  const height = 20;
  const rx = x - width / 2;
  const ry = y;
  ctx.fillStyle = 'rgba(255,255,255,0.86)';
  ctx.strokeStyle = 'rgba(15,23,42,0.18)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(rx, ry, width, height, 7);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#0f172a';
  ctx.textAlign = 'center';
  ctx.fillText(text, x, y + 14);
}

function drawSignalTooltip(ctx: CanvasRenderingContext2D, hover: SignalHoverInfo & { screen: Coord }, canvasWidth: number) {
  const [mx, my] = hover.screen;
  const valueText = hover.dbm === null ? 'No AP' : `${hover.dbm.toFixed(1)} dBm`;
  const line2 = `${hover.label}${hover.cellType ? ` · ${hover.cellType}` : ''}`;
  ctx.save();
  ctx.font = 'bold 14px Inter, system-ui, sans-serif';
  const width = Math.max(ctx.measureText(valueText).width, ctx.measureText(line2).width) + 28;
  const height = 52;
  const x = Math.min(Math.max(10, mx + 16), canvasWidth - width - 10);
  const y = Math.max(10, my - 62);
  ctx.fillStyle = 'rgba(15,23,42,0.94)';
  ctx.strokeStyle = hover.dbm !== null && hover.dbm >= -67 ? '#22c55e' : hover.dbm !== null && hover.dbm >= -75 ? '#facc15' : '#38bdf8';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, 14);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'left';
  ctx.fillText(valueText, x + 14, y + 21);
  ctx.font = '12px Inter, system-ui, sans-serif';
  ctx.fillStyle = '#cbd5e1';
  ctx.fillText(line2, x + 14, y + 40);
  ctx.restore();
}

export function FloorplanCanvas({
  model,
  coverage,
  tool,
  showConfidence,
  heatmap,
  backgroundImageUrl,
  imageOpacity = 0.72,
  onChange,
  onPreviewDrag,
  onDragEnd,
  onSignalHover,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [scale, setScale] = useState(60);
  const [wallStart, setWallStart] = useState<Coord | null>(null);
  const [roiStart, setRoiStart] = useState<Coord | null>(null);
  const [dragApId, setDragApId] = useState<string | null>(null);
  const [selected, setSelected] = useState<string>('');
  const [backgroundImage, setBackgroundImage] = useState<HTMLImageElement | null>(null);
  const [hover, setHover] = useState<(SignalHoverInfo & { screen: Coord }) | null>(null);

  const toScreen = (p: Coord): Coord => [p[0] * scale + 24, p[1] * scale + 24];
  const fromScreen = (x: number, y: number): Coord => [Math.max(0, (x - 24) / scale), Math.max(0, (y - 24) / scale)];

  function activeRegionPolygon(): Coord[] | undefined {
    return (model.analysis_regions ?? []).find((region) => region.active)?.boundary_polygon_m;
  }

  function insideActiveRegion(p: Coord) {
    const polygon = activeRegionPolygon();
    return !polygon || pointInPolygon(p[0], p[1], polygon);
  }

  function signalAtPosition(p: Coord): SignalHoverInfo {
    const grid = model.grid;
    if (grid.cells.length && grid.cols > 0 && grid.rows > 0) {
      const col = Math.floor((p[0] - grid.origin_m[0]) / grid.resolution_m);
      const row = Math.floor((p[1] - grid.origin_m[1]) / grid.resolution_m);
      if (row >= 0 && row < grid.rows && col >= 0 && col < grid.cols) {
        const idx = row * grid.cols + col;
        const cell = grid.cells[idx];
        const fromCoverage = coverage?.coverage_dbm?.[idx];
        const dbm = typeof fromCoverage === 'number' ? fromCoverage : estimateDbmAtPoint(model, p, cell?.attenuation_db ?? 0);
        return { position_m: p, dbm, label: signalLabel(dbm), cellType: cell?.type };
      }
    }
    const dbm = estimateDbmAtPoint(model, p, 0);
    return { position_m: p, dbm, label: signalLabel(dbm), cellType: 'outside grid' };
  }

  useEffect(() => {
    if (!backgroundImageUrl) {
      setBackgroundImage(null);
      return;
    }
    const img = new Image();
    img.onload = () => setBackgroundImage(img);
    img.src = backgroundImageUrl;
  }, [backgroundImageUrl]);

  useEffect(() => {
    const maxDim = Math.max(model.metadata.floor_dimensions_m.width, model.metadata.floor_dimensions_m.height, 1);
    setScale(Math.min(75, Math.max(0.45, 860 / maxDim)));
  }, [model.metadata.floor_dimensions_m.width, model.metadata.floor_dimensions_m.height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const widthPx = Math.max(360, model.metadata.floor_dimensions_m.width * scale + 48);
    const heightPx = Math.max(260, model.metadata.floor_dimensions_m.height * scale + 48);
    canvas.width = widthPx;
    canvas.height = heightPx;
    ctx.clearRect(0, 0, widthPx, heightPx);

    // Workspace base.
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, widthPx, heightPx);
    ctx.strokeStyle = 'rgba(148,163,184,0.22)';
    ctx.lineWidth = 1;
    const gridStep = Math.max(28, scale * 2);
    for (let x = 24; x < widthPx; x += gridStep) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, heightPx); ctx.stroke();
    }
    for (let y = 24; y < heightPx; y += gridStep) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(widthPx, y); ctx.stroke();
    }

    // Uploaded floorplan image must be visible under all analysis layers.
    if (backgroundImage) {
      ctx.save();
      ctx.globalAlpha = imageOpacity;
      const [x, y] = toScreen([0, 0]);
      ctx.drawImage(
        backgroundImage,
        x,
        y,
        model.metadata.floor_dimensions_m.width * scale,
        model.metadata.floor_dimensions_m.height * scale,
      );
      ctx.restore();
    }

    // Building footprint: the propagation domain. This is deliberately drawn
    // before rooms/walls to show that heatmap/grid is based on interior area,
    // not wall pixels.
    for (const footprint of model.building_footprints ?? []) {
      ctx.beginPath();
      footprint.boundary_polygon_m.forEach((p, idx) => {
        const [x, y] = toScreen(p);
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(16,185,129,0.055)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(16,185,129,0.48)';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Heatmap below walls/APs, like the reference image.
    if (heatmap && coverage) {
      for (let i = 0; i < model.grid.cells.length; i++) {
        const cell = model.grid.cells[i];
        const dbm = coverage.coverage_dbm[i] ?? -120;
        if (dbm <= -110) continue;
        const [sx, sy] = toScreen([cell.center_m[0] - model.grid.resolution_m / 2, cell.center_m[1] - model.grid.resolution_m / 2]);
        ctx.fillStyle = colorForDbm(dbm);
        ctx.fillRect(sx, sy, model.grid.resolution_m * scale + 0.4, model.grid.resolution_m * scale + 0.4);
      }
    }

    for (const region of model.analysis_regions ?? []) {
      if (!region.active) continue;
      ctx.beginPath();
      region.boundary_polygon_m.forEach((p, idx) => {
        const [x, y] = toScreen(p);
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(37,99,235,0.10)';
      ctx.fill();
      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 3;
      ctx.setLineDash([10, 6]);
      ctx.stroke();
      ctx.setLineDash([]);
      const first = region.boundary_polygon_m[0];
      if (first) {
        const [lx, ly] = toScreen(first);
        drawRoundedLabel(ctx, region.label || 'Coverage ROI', lx + 64, ly + 8);
      }
    }

    for (const room of model.rooms) {
      ctx.beginPath();
      room.boundary_polygon_m.forEach((p, idx) => {
        const [x, y] = toScreen(p);
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = room.occupancy_level === 'high' ? 'rgba(59,130,246,0.08)' : 'rgba(148,163,184,0.06)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(15,23,42,0.20)';
      ctx.lineWidth = 1;
      ctx.stroke();
      if (!backgroundImage) {
        const [lx, ly] = toScreen(room.centroid_m);
        ctx.fillStyle = '#334155';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${room.label}`, lx, ly);
      }
    }

    // Walls are red overlays to match the provided reference.
    for (const wall of model.walls) {
      const [x1, y1] = toScreen(wall.start_m);
      const [x2, y2] = toScreen(wall.end_m);
      ctx.strokeStyle = wall.id === selected ? '#7c3aed' : (showConfidence ? wallConfidenceColor(wall.confidence) : '#ef1d2f');
      ctx.lineWidth = Math.max(3, Math.min(7, wall.thickness_m * scale));
      ctx.lineCap = 'square';
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      if (showConfidence && wall.confidence < 0.75) {
        ctx.strokeStyle = 'rgba(15,23,42,0.9)';
        ctx.lineWidth = 2;
        ctx.setLineDash([7, 5]);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    for (const door of model.doors) {
      const [x, y] = toScreen(door.position_m);
      ctx.fillStyle = '#f97316';
      ctx.strokeStyle = '#7c2d12';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, Math.max(5, Math.min(15, door.width_m * scale * 0.18)), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    }
    for (const win of model.windows) {
      const [x, y] = toScreen(win.position_m);
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 5;
      ctx.beginPath(); ctx.moveTo(x - 9, y - 9); ctx.lineTo(x + 9, y + 9); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x + 9, y - 9); ctx.lineTo(x - 9, y + 9); ctx.stroke();
    }

    // AP style inspired by the reference: amber square halo, dotted orange ring, blue center, label.
    for (const ap of model.access_points) {
      const [x, y] = toScreen(ap.position_m);
      const selectedAp = ap.id === selected;
      ctx.save();
      ctx.fillStyle = 'rgba(251,191,36,0.28)';
      ctx.fillRect(x - 24, y - 24, 48, 48);
      ctx.strokeStyle = selectedAp ? '#7c3aed' : '#f59e0b';
      ctx.lineWidth = selectedAp ? 4 : 3;
      ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.arc(x, y, 22, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#2563eb';
      ctx.strokeStyle = '#dbeafe';
      ctx.lineWidth = 4;
      ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.restore();
      drawRoundedLabel(ctx, ap.id.startsWith('ap_') ? 'DAP-X2810' : ap.id, x, y + 27);
    }

    if (wallStart) {
      const [x, y] = toScreen(wallStart);
      ctx.fillStyle = '#ef1d2f'; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
      drawRoundedLabel(ctx, 'Click wall end point', x + 60, y - 12);
    }

    if (roiStart) {
      const [x, y] = toScreen(roiStart);
      ctx.fillStyle = '#2563eb'; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
      drawRoundedLabel(ctx, 'Click opposite ROI corner', x + 84, y - 12);
    }

    if (hover) {
      const [hx, hy] = hover.screen;
      ctx.save();
      ctx.strokeStyle = 'rgba(15,23,42,0.35)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(hx, 0); ctx.lineTo(hx, heightPx); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, hy); ctx.lineTo(widthPx, hy); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = hover.dbm !== null && hover.dbm >= -67 ? '#22c55e' : hover.dbm !== null && hover.dbm >= -75 ? '#facc15' : '#38bdf8';
      ctx.beginPath(); ctx.arc(hx, hy, 4, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      drawSignalTooltip(ctx, hover, widthPx);
    }
  }, [model, coverage, heatmap, scale, selected, showConfidence, wallStart, roiStart, backgroundImage, imageOpacity, hover]);

  function updateApPosition(id: string, pos: Coord, commit = false) {
    const next = structuredClone(model) as BuildingModel;
    const ap = next.access_points.find((item) => item.id === id);
    if (!ap) return;
    const rounded: Coord = [Math.round(pos[0] * 10) / 10, Math.round(pos[1] * 10) / 10];
    if (!insideActiveRegion(rounded)) return;
    ap.position_m = rounded;
    ap.source = 'manual';
    onChange(next);
    if (commit) onDragEnd?.(next); else onPreviewDrag?.(next);
  }

  function pointerPos(e: React.MouseEvent<HTMLCanvasElement>): Coord {
    const rect = e.currentTarget.getBoundingClientRect();
    const canvas = e.currentTarget;
    const internalX = (e.clientX - rect.left) * (canvas.width / Math.max(1, rect.width));
    const internalY = (e.clientY - rect.top) * (canvas.height / Math.max(1, rect.height));
    return fromScreen(internalX, internalY);
  }

  function onMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const p = pointerPos(e);

    if (tool === 'roi') {
      if (!roiStart) {
        setRoiStart(p);
      } else {
        const x1 = Math.min(roiStart[0], p[0]);
        const x2 = Math.max(roiStart[0], p[0]);
        const y1 = Math.min(roiStart[1], p[1]);
        const y2 = Math.max(roiStart[1], p[1]);
        const next = structuredClone(model) as BuildingModel;
        const roiPolygon: Coord[] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
        next.analysis_regions = [
          {
            id: `roi_${Date.now()}`,
            label: 'Coverage ROI',
            boundary_polygon_m: roiPolygon,
            active: true,
            user_edited: true,
          },
        ];
        // If the user defines a coverage area, keep APs inside it. This prevents
        // stale recommendations/manual points outside ROI from staying visible.
        next.access_points = next.access_points.filter((ap) => pointInPolygon(ap.position_m[0], ap.position_m[1], roiPolygon));
        setRoiStart(null);
        onChange(next);
        onDragEnd?.(next);
      }
      return;
    }

    const ap = model.access_points.find((item) => distPoint(item.position_m, p) * scale < 18);
    const clickedWindow = model.windows.find((item) => distPoint(item.position_m, p) * scale < 18);
    const nearestWall = model.walls
      .map((w) => ({ wall: w, d: distanceToSegment(p, w.start_m, w.end_m) }))
      .sort((a, b) => a.d - b.d)[0];

    if (tool === 'delete' && clickedWindow) {
      const next = structuredClone(model) as BuildingModel;
      next.windows = next.windows.filter((item) => item.id !== clickedWindow.id);
      onChange(next);
      onDragEnd?.(next);
      return;
    }

    if (tool === 'delete' && ap) {
      const next = structuredClone(model) as BuildingModel;
      next.access_points = next.access_points.filter((item) => item.id !== ap.id);
      setSelected('');
      setDragApId(null);
      onChange(next);
      onDragEnd?.(next);
      return;
    }

    if (ap) { setDragApId(ap.id); setSelected(ap.id); return; }

    if (tool === 'window') {
      if (!nearestWall || nearestWall.d * scale > 26) return;
      const next = structuredClone(model) as BuildingModel;
      const win: WindowElement = {
        id: `win_user_${Date.now()}`,
        wall_id: nearestWall.wall.id,
        position_m: [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10],
        width_m: 1.2,
        attenuation_db: 3,
      };
      next.windows.push(win);
      onChange(next);
      onDragEnd?.(next);
      return;
    }

    if (tool === 'wall') {
      if (!wallStart) setWallStart(p);
      else {
        const next = structuredClone(model) as BuildingModel;
        const wall: Wall = {
          id: `w_user_${Date.now()}`,
          start_m: wallStart,
          end_m: p,
          thickness_m: Math.max(0.2, 4 / scale),
          material: 'drywall',
          attenuation_db: 4,
          confidence: 1,
          user_edited: true,
        };
        next.walls.push(wall);
        setWallStart(null);
        onChange(next);
      }
      return;
    }

    if (tool === 'ap') {
      if (!insideActiveRegion(p)) return;
      const next = structuredClone(model) as BuildingModel;
      next.access_points.push({ id: `ap_${next.access_points.length + 1}`, position_m: p, tx_power_dbm: 18, freq_ghz: 5, source: 'manual' });
      onChange(next);
      onDragEnd?.(next);
      return;
    }

    const wall = nearestWall && nearestWall.d * scale < 10 ? nearestWall.wall : undefined;
    if (tool === 'delete' && wall) {
      const next = structuredClone(model) as BuildingModel;
      next.walls = next.walls.filter((w) => w.id !== wall.id);
      next.windows = next.windows.filter((win) => win.wall_id !== wall.id);
      onChange(next);
      onDragEnd?.(next);
      return;
    }
    setSelected(wall?.id ?? '');
  }

  function onMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const canvas = e.currentTarget;
    const screen: Coord = [
      (e.clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
      (e.clientY - rect.top) * (canvas.height / Math.max(1, rect.height)),
    ];
    const p = fromScreen(screen[0], screen[1]);
    const signal = signalAtPosition(p);
    const nextHover = { ...signal, screen };
    setHover(nextHover);
    onSignalHover?.(signal);

    if (!dragApId) return;
    updateApPosition(dragApId, p, false);
  }

  function onMouseLeave() {
    setHover(null);
    onSignalHover?.(null);
  }

  function onMouseUp(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!dragApId) return;
    updateApPosition(dragApId, pointerPos(e), true);
    setDragApId(null);
  }

  return (
    <canvas
      ref={canvasRef}
      className="h-auto max-w-full rounded-2xl border border-slate-300 bg-white shadow-inner"
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseLeave}
    />
  );
}
