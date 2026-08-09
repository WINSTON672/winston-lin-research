"""
Attach OSM's campus names to the building footprints.
=====================================================

    python3 name_buildings.py            # dry run, prints matches
    python3 name_buildings.py --write    # update campus.json

Blair's building names *are* in OpenStreetMap, but they are not on the building
polygons. They sit on separate `amenity=school` features -- some are points
dropped inside a building, some are their own areas -- which is why a query for
`building` with a `name` came back with nothing. The names and the footprints
are two disjoint sets of objects that happen to occupy the same ground.

So this joins them spatially rather than by guesswork:

  1. a named feature whose point falls *inside* a footprint names that building
  2. otherwise the nearest footprint within MAX_MATCH_M gets the name
  3. anything further away is left unmatched and reported

Rule 1 is certain. Rule 2 is an inference, and every match made that way is
printed with its distance so it can be checked rather than trusted. Nothing is
assigned by eyeballing a campus map, because a plausible label on the wrong
dormitory is worse than no label.
"""

from __future__ import annotations

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LAT0, LON0 = 40.9870461, -74.9665719          # same origin as build_campus.py

# Beyond this a named point is probably describing something else -- a field,
# a road, a neighbouring structure -- so the match is refused.
MAX_MATCH_M = 35.0

# Named features that are not buildings and must never claim a footprint.
SKIP_KINDS = {'pitch', 'track', 'golf_course', 'service', 'residential',
              'footway', 'path', 'parking', 'water'}
SKIP_NAMES = {'Blair Academy'}                 # the campus itself


def to_m(lat: float, lon: float) -> tuple[float, float]:
    return (math.radians(lon - LON0) * 6378137.0 * math.cos(math.radians(LAT0)),
            math.radians(lat - LAT0) * 6378137.0)


def point_in_poly(pt, poly) -> bool:
    """Ray casting. Footprints are simple closed rings, so this is enough."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def dist_to_poly(pt, poly) -> float:
    """Distance from a point to a polygon's boundary, 0 if inside."""
    if point_in_poly(pt, poly):
        return 0.0
    x, y = pt
    best = float('inf')
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
        best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best


def load_named() -> list[dict]:
    with open(os.path.join(HERE, 'named_features.json')) as fh:
        els = json.load(fh)['elements']
    out = []
    for e in els:
        t = e.get('tags', {})
        name = t.get('name')
        if not name or name in SKIP_NAMES:
            continue
        kind = (t.get('building') or t.get('amenity') or t.get('leisure')
                or t.get('highway') or t.get('landuse') or '?')
        if kind in SKIP_KINDS:
            continue
        lat = e.get('lat') or (e.get('center') or {}).get('lat')
        lon = e.get('lon') or (e.get('center') or {}).get('lon')
        if lat is None:
            continue
        out.append(dict(name=name, kind=kind, pt=to_m(lat, lon),
                        ref=f"{e['type']}/{e['id']}"))
    return out


def load_footprints() -> list[dict]:
    """Footprints come from the raw OSM cache, since campus.json drops the
    geometry."""
    with open(os.path.join(HERE, 'blair_osm.json')) as fh:
        els = json.load(fh)['elements']
    out = []
    for e in els:
        if 'building' not in e.get('tags', {}):
            continue
        g = e.get('geometry') or []
        if len(g) < 3:
            continue
        poly = [to_m(p['lat'], p['lon']) for p in g]
        if poly[0] == poly[-1]:
            poly.pop()
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        if math.hypot(cx, cy) > 500:          # campus radius, matches builder
            continue
        out.append(dict(id=e['id'], poly=poly))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--max-m', type=float, default=MAX_MATCH_M)
    args = ap.parse_args()

    named = load_named()
    foots = load_footprints()
    print(f'{len(named)} named campus features, {len(foots)} footprints\n')

    # A footprint may legitimately hold several named features: the athletic
    # complex here contains Tracy Hall and two separately-named parts of
    # Hardwick Hall. Dropping the extras would lose real information, so a
    # building accumulates every name that falls inside it.
    claimed: dict[int, list[tuple[str, float]]] = {}
    unmatched = []
    for f in named:
        d, best = min(((dist_to_poly(f['pt'], b['poly']), b) for b in foots),
                      key=lambda s: s[0])
        if d > args.max_m:
            unmatched.append((f['name'], d))
            continue
        claimed.setdefault(best['id'], []).append((f['name'], d))

    inside = {i: v for i, v in claimed.items() if all(d == 0.0 for _, d in v)}
    near = {i: v for i, v in claimed.items() if any(d > 0.0 for _, d in v)}

    print(f'INSIDE a footprint — certain ({len(inside)}):')
    for i, names in sorted(inside.items(), key=lambda kv: kv[1][0][0]):
        head = names[0][0]
        print(f'  way {i:<12} {head}')
        for extra, _ in names[1:]:
            print(f'  {"":16} + also {extra}')
    print(f'\nNEAREST within {args.max_m:.0f} m — check these ({len(near)}):')
    for i, names in sorted(near.items(), key=lambda kv: kv[1][0][1]):
        for n, d in names:
            print(f'  way {i:<12} {d:>5.1f}m  {n}')
    if unmatched:
        print(f'\nNOT MATCHED ({len(unmatched)}) — no footprint close enough:')
        for n, d in sorted(unmatched, key=lambda x: x[1]):
            print(f'  {d:>6.0f}m  {n}')

    if not args.write:
        print('\n(dry run — pass --write to update campus.json)')
        return

    path = os.path.join(HERE, 'campus.json')
    with open(path) as fh:
        recs = json.load(fh)
    n = 0
    for r in recs:
        hits = claimed.get(r['id'])
        if not hits:
            continue
        hits.sort(key=lambda h: h[1])
        r['name'] = hits[0][0]
        if len(hits) > 1:
            r['also_houses'] = [h[0] for h in hits[1:]]
        worst = max(h[1] for h in hits)
        r['name_source'] = 'osm:inside' if worst == 0.0 else f'osm:near:{worst:.1f}m'
        n += 1
    with open(path, 'w') as fh:
        json.dump(recs, fh, indent=2)
    print(f'\nwrote {n} names into campus.json')


if __name__ == '__main__':
    main()
