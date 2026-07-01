export type Coord = [number, number];

export interface Metadata {
  source_image_ref: string;
  scale_m_per_px: number;
  scale_confidence: number;
  scale_method: 'ocr_text' | 'scale_bar' | 'dimension' | 'reference_object' | 'manual';
  floor_dimensions_m: { width: number; height: number };
  created_at: string;
  last_modified_at: string;
}

export interface Wall {
  id: string;
  start_m: Coord;
  end_m: Coord;
  thickness_m: number;
  material: string;
  attenuation_db: number;
  confidence: number;
  user_edited: boolean;
}

export interface Room {
  id: string;
  boundary_polygon_m: Coord[];
  area_m2: number;
  centroid_m: Coord;
  label: string;
  occupancy_level: 'low' | 'medium' | 'high';
  adjacent_room_ids: string[];
  confidence: number;
  user_edited: boolean;
}

export interface Door {
  id: string;
  wall_id: string;
  position_m: Coord;
  width_m: number;
  material: string;
  attenuation_db: number;
}

export interface WindowElement {
  id: string;
  wall_id: string;
  position_m: Coord;
  width_m: number;
  attenuation_db: number;
}

export interface GridCell {
  center_m: Coord;
  type: 'open' | 'wall' | 'door' | 'window' | 'outside';
  attenuation_db: number;
  room_id: string | null;
  placeable: boolean;
}

export interface Grid {
  resolution_m: number;
  origin_m: Coord;
  cols: number;
  rows: number;
  floor_id: string;
  cells: GridCell[];
}

export interface AccessPoint {
  id: string;
  position_m: Coord;
  tx_power_dbm: number;
  freq_ghz: number;
  source: 'algorithm' | 'manual';
}

export interface BuildingFootprint {
  id: string;
  boundary_polygon_m: Coord[];
  confidence: number;
  user_edited: boolean;
}

export interface AnalysisRegion {
  id: string;
  label: string;
  boundary_polygon_m: Coord[];
  active: boolean;
  user_edited: boolean;
}

export interface BuildingModel {
  schema_version: '0.1.0';
  metadata: Metadata;
  walls: Wall[];
  rooms: Room[];
  doors: Door[];
  windows: WindowElement[];
  grid: Grid;
  access_points: AccessPoint[];
  building_footprints?: BuildingFootprint[];
  analysis_regions?: AnalysisRegion[];
}

export interface CoverageResult {
  coverage_dbm: number[];
  rows: number;
  cols: number;
  resolution_m: number;
  mode: 'preview' | 'full';
}
