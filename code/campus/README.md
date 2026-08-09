# Campus map

The outdoor half of a school tour, free, from OpenStreetMap.

```bash
python3 build_campus.py --fetch     # download OSM data (cached after)
python3 build_campus.py             # build
python3 build_campus.py --radius 0  # keep the town too
```

Outputs `campus.obj` (opens in Blender), `campus.png` (preview), `campus.json`
(one record per building, ready for names and photo links).

## Why bother before any photos exist

OSM gives away the shape of every building on campus. What it does not have is
interiors: the whole campus has eleven features mentioning anything indoor, and
they are all "this building has N floors" plus a tennis court. Zero rooms, zero
hallways.

That absence is the point. Outsides are free; insides can only come from someone
standing in the room with a camera. The map is the thing you click *into*.

## Two things to know about the data

**The bbox includes Blairstown town.** Building counts by distance from campus
centre break cleanly — 34 within 400 m, 52 within 500 m, then 127 by 700 m as
the town starts. Default radius is 500 m. `--radius 0` disables it.

**Almost every height is a guess.** One building on campus carries
`building:levels`; the other 51 are estimated from footprint area on the theory
that a big building is taller than a shed. Fix the ones that matter by hand in
`campus.json` — the field `height_source` says which are real.

## Nothing is named

OSM has no name for any Blair building. The largest footprints, to label first:

| Area | OSM way id |
|---|---|
| 7423 m² | 1114756236 |
| 2061 m² | 1114666218 |
| 1701 m² | 1113764405 |
| 1155 m² | 628794613 |

Open `campus.png`, compare against the school map, and fill in `name` in
`campus.json`. About an hour for the twenty that matter.

## Attribution

OSM data © OpenStreetMap contributors, ODbL. Any published map must say so.
