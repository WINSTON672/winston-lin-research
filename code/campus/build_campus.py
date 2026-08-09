"""
Campus map from OpenStreetMap footprints.
=========================================

Pulls building outlines and paths from OSM, projects them to metres, extrudes
the buildings into blocks, and writes a 3D model plus a preview image.

    python3 build_campus.py --fetch          # download and cache OSM data
    python3 build_campus.py                  # build from cache
    python3 build_campus.py --preview-only   # just the PNG

Outputs
    campus.obj    3D model -- opens in Blender, three.js, Preview, anything
    campus.png    top-down preview, so you can see it without a 3D app
    campus.json   per-building record, ready for the name/photo labels

Why this is worth doing before any photography happens: OSM knows the shape of
every building on campus and gives it away, so the outdoor half of a campus
tour costs nothing and can exist tonight. What OSM does not have is interiors --
checked, and the entire campus has eleven features that mention anything indoor,
all of them "this building has N floors" plus a tennis court. That absence is
the reason the project is worth doing at all: the outsides are free, the insides
can only come from someone standing in the room.

Heights are estimates. OSM carries `building:levels` for a handful of buildings
here and nothing for the rest, so the rest are inferred from footprint area --
a big building is probably taller than a shed. That is a guess and the model
labels it as one; anything that matters should be corrected by hand in
campus.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'blair_osm.json')

# Blair Academy, from Nominatim.
BBOX = (40.9824, -74.9742, 40.9915, -74.9581)   # S, W, N, E
# Project about the campus itself (Nominatim's point for Blair Academy) rather
# than the bbox centre, so the radius filter below is measured from the school.
LAT0, LON0 = 40.9870461, -74.9665719

# The OSM bbox necessarily includes Blairstown town. Building counts by radius
# from the campus centre show a clean break: 34 within 400 m, 52 within 500 m,
# then 127 by 700 m as the town begins. 500 m keeps the campus and drops it.
CAMPUS_RADIUS_M = 500.0

ENDPOINTS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://overpass.osm.ch/api/interpreter',
]

QUERY = f"""[out:json][timeout:60];
(
  way["building"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["highway"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  way["natural"="water"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out geom;"""

DEFAULT_STOREY_M = 3.5


# ── fetch ───────────────────────────────────────────────────────────────────

def fetch(force: bool = False) -> dict:
    """Download from OSM, trying mirrors. Overpass rate-limits the main
    endpoint aggressively, so a single failed request means nothing."""
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as fh:
            return json.load(fh)

    body = urllib.parse.urlencode({'data': QUERY}).encode()
    last = None
    for attempt, url in enumerate(ENDPOINTS * 2):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={'User-Agent': 'campus-map/1.0 (student project)'})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode())
            if 'elements' not in data:
                raise ValueError('no elements in response')
            with open(CACHE, 'w') as fh:
                json.dump(data, fh)
            print(f'fetched from {urllib.parse.urlparse(url).netloc}')
            return data
        except Exception as exc:                       # noqa: BLE001
            last = exc
            wait = 4 * (attempt + 1)
            print(f'  {urllib.parse.urlparse(url).netloc} failed ({exc}); '
                  f'retrying in {wait}s')
            time.sleep(wait)
    raise SystemExit(f'all Overpass mirrors failed: {last}')


# ── geometry ────────────────────────────────────────────────────────────────

def to_metres(lat: float, lon: float) -> tuple[float, float]:
    """Equirectangular projection about the campus centre. Over a ~1 km span
    the distortion is centimetres, far below the accuracy of the footprints."""
    x = math.radians(lon - LON0) * 6378137.0 * math.cos(math.radians(LAT0))
    y = math.radians(lat - LAT0) * 6378137.0
    return x, y


def ring(el: dict) -> list[tuple[float, float]]:
    pts = [to_metres(p['lat'], p['lon']) for p in el.get('geometry', [])]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    return pts


def area(poly: list[tuple[float, float]]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    s = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
            for i in range(n))
    return abs(s) / 2.0


def is_ccw(poly: list[tuple[float, float]]) -> bool:
    n = len(poly)
    s = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
            for i in range(n))
    return s > 0


def height_for(el: dict, foot_area: float) -> tuple[float, str]:
    """Real height if OSM has one, otherwise a guess from footprint size."""
    t = el.get('tags', {})
    if 'height' in t:
        try:
            return float(str(t['height']).split()[0]), 'osm:height'
        except ValueError:
            pass
    if 'building:levels' in t:
        try:
            return float(t['building:levels']) * DEFAULT_STOREY_M, 'osm:levels'
        except ValueError:
            pass
    # Guess: sheds are one storey, halls are three or four.
    if foot_area < 60:
        return 3.0, 'guessed'
    if foot_area < 200:
        return 6.0, 'guessed'
    if foot_area < 700:
        return 9.5, 'guessed'
    return 13.0, 'guessed'


# ── build ───────────────────────────────────────────────────────────────────

def _centroid(poly):
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def build(data: dict, radius: float = CAMPUS_RADIUS_M) -> tuple[list[dict], list[dict]]:
    buildings, paths = [], []
    for el in data['elements']:
        t = el.get('tags', {})
        poly = ring(el)
        if 'building' in t:
            if len(poly) < 3:
                continue
            a = area(poly)
            if a < 12:                       # bins, porches, mapping noise
                continue
            cx, cy = _centroid(poly)
            if radius and math.hypot(cx, cy) > radius:
                continue                     # Blairstown town, not campus
            h, src = height_for(el, a)
            buildings.append(dict(
                id=el['id'], name=t.get('name'), kind=t.get('building', 'yes'),
                poly=poly, area_m2=round(a, 1), height_m=h, height_source=src,
            ))
        elif 'highway' in t and len(poly) >= 2:
            if radius and min(math.hypot(x, y) for x, y in poly) > radius * 1.3:
                continue
            paths.append(dict(id=el['id'], kind=t['highway'], poly=poly))
    return buildings, paths


def write_obj(buildings: list[dict], path: str) -> None:
    """Extrude each footprint into a prism. Walls plus a flat roof; no floor,
    since nothing ever sees the underside of the campus."""
    lines = ['# Blair campus, extruded from OpenStreetMap footprints',
             '# OSM data (c) OpenStreetMap contributors, ODbL',
             f'# {len(buildings)} buildings']
    voff = 1
    for b in buildings:
        poly = b['poly'] if is_ccw(b['poly']) else b['poly'][::-1]
        n = len(poly)
        h = b['height_m']
        lines.append(f"o building_{b['id']}" + (f"_{_slug(b['name'])}" if b['name'] else ''))
        for x, y in poly:
            lines.append(f'v {x:.3f} 0.000 {-y:.3f}')      # OBJ is Y-up
        for x, y in poly:
            lines.append(f'v {x:.3f} {h:.3f} {-y:.3f}')
        for i in range(n):
            j = (i + 1) % n
            a, b_, c, d = voff + i, voff + j, voff + n + j, voff + n + i
            lines.append(f'f {a} {b_} {c} {d}')            # wall quad
        lines.append('f ' + ' '.join(str(voff + n + i) for i in range(n)))  # roof
        voff += 2 * n
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def _slug(s: str) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in s)[:40]


def write_preview(buildings, paths, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        print('matplotlib missing, skipping preview')
        return

    fig, ax = plt.subplots(figsize=(13, 13), facecolor='#11100e')
    ax.set_facecolor('#11100e')

    for p in paths:
        xs = [q[0] for q in p['poly']]
        ys = [q[1] for q in p['poly']]
        ax.plot(xs, ys, color='#3a3730', lw=1.0, zorder=1)

    hs = [b['height_m'] for b in buildings] or [1]
    lo, hi = min(hs), max(hs)
    for b in buildings:
        frac = (b['height_m'] - lo) / (hi - lo) if hi > lo else 0.5
        # taller = warmer, so the big halls read at a glance
        col = (0.42 + 0.45 * frac, 0.34 + 0.20 * frac, 0.22 + 0.05 * frac)
        ax.add_patch(Polygon(b['poly'], closed=True, facecolor=col,
                             edgecolor='#0c0b09', lw=0.6, zorder=2))
        if b['name']:
            cx = sum(q[0] for q in b['poly']) / len(b['poly'])
            cy = sum(q[1] for q in b['poly']) / len(b['poly'])
            ax.annotate(b['name'], (cx, cy), color='#e8e2d4', fontsize=7,
                        ha='center', va='center', zorder=3)

    # Frame on the buildings. A single road leaving the bbox would otherwise
    # set the extent and shrink the campus to a smudge.
    bxs = [q[0] for b in buildings for q in b['poly']]
    bys = [q[1] for b in buildings for q in b['poly']]
    if bxs:
        pad = 40
        ax.set_xlim(min(bxs) - pad, max(bxs) + pad)
        ax.set_ylim(min(bys) - pad, max(bys) + pad)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Blair Academy — {len(buildings)} buildings from OpenStreetMap\n'
                 'heights mostly estimated from footprint area',
                 color='#c9c2b2', fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor='#11100e')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true', help='re-download OSM data')
    ap.add_argument('--preview-only', action='store_true')
    ap.add_argument('--radius', type=float, default=CAMPUS_RADIUS_M,
                    help='metres from campus centre; 0 keeps everything')
    args = ap.parse_args()

    data = fetch(force=args.fetch)
    buildings, paths = build(data, args.radius)
    buildings.sort(key=lambda b: -b['area_m2'])

    # Names, scene links and corrected heights live in campus.json and are the
    # only things here a person made. Rebuilding from OSM must not destroy
    # them, so they are merged back over the freshly derived records.
    KEEP = ('name', 'also_houses', 'name_source', 'scene', 'status')
    prior = {}
    jpath = os.path.join(HERE, 'campus.json')
    if os.path.exists(jpath):
        with open(jpath) as fh:
            for r in json.load(fh):
                prior[r['id']] = {k: r[k] for k in KEEP if r.get(k) is not None}
        # A height edited by hand outranks anything guessed from area.
        for b in buildings:
            was = prior.get(b['id'], {})
            if was.get('height_source') in ('measured',):
                b['height_m'] = was['height_m']
                b['height_source'] = 'measured'

    for b in buildings:
        b.update(prior.get(b['id'], {}))
    kept = sum(1 for b in buildings if b.get('name'))
    if prior:
        print(f'carried {kept} names forward from the existing campus.json')


    print(f'{len(buildings)} buildings, {len(paths)} paths')
    real = sum(1 for b in buildings if b['height_source'] != 'guessed')
    print(f'heights: {real} from OSM, {len(buildings) - real} estimated')
    named = [b for b in buildings if b['name']]
    print(f'named in OSM: {len(named)}'
          + (f" ({', '.join(b['name'] for b in named[:4])})" if named else ''))
    print('\nlargest footprints (label these first):')
    for b in buildings[:8]:
        print(f"  {b['area_m2']:>8.0f} m2  h={b['height_m']:>4.1f}m  "
              f"{b['name'] or '(unnamed — id ' + str(b['id']) + ')'}")

    write_preview(buildings, paths, os.path.join(HERE, 'campus.png'))
    written = ['campus.png']
    if not args.preview_only:
        write_obj(buildings, os.path.join(HERE, 'campus.obj'))
        with open(jpath, 'w') as fh:
            json.dump([{k: v for k, v in b.items() if k != 'poly'}
                       for b in buildings], fh, indent=2)
        written += ['campus.obj', 'campus.json']
    print('\nwrote ' + ', '.join(written))


if __name__ == '__main__':
    main()
