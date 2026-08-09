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

## editor.html — the 3D editor

```bash
python3 -m http.server 8000
# open http://localhost:8000/editor.html
```

Must be served over HTTP; opening the file directly fails because the browser
blocks `fetch` on `file://`.

Orbit with drag, zoom with scroll, pan with right-drag, click a building to
edit it. Set the name, correct the height (live preview while you type), point
it at an interior scene file, and mark tour status. **Download campus.json**
writes the edits back — replace the file in this folder.

**Orange means the height is a guess, grey means it came from OSM.** Right now
that is 51 orange and 1 grey, which is an honest picture of how much of this
model is inferred rather than known.

three.js is vendored in `vendor/` rather than loaded from a CDN. The CDN import
failed here even though the CDN answered HEAD with 200, and the editor has to
work on a machine with no internet.

## room.html — 360 photo viewer

```
http://localhost:8000/room.html                    drop a photo
http://localhost:8000/room.html?src=my-room.jpg    or link one directly
```

Drop an equirectangular JPG and look around. **No processing at all** — the
image is mapped onto the inside of a sphere by the browser. No COLMAP, no
training, no GPU.

**A full sphere is 2:1.** The viewer measures the aspect ratio and says which
you gave it. iPhone *Panorama* mode does **not** produce one — it makes a wide
strip, and the viewer renders it as the vertical wedge it actually covers
rather than smearing it over the whole sphere and inventing a ceiling. For a
real photosphere: a 360 camera, or Google Street View's app, which stitches one
from a guided capture.

`pano_test.jpg` is a generated test pattern with compass letters and coloured
poles — if the sphere is mapped wrong the letters come out mirrored.

### What 360 photos cannot do

They have **no geometry**. You are pinned to the point the camera stood at:
you cannot walk, and nothing can be measured. That is the trade for costing
nothing. Splats give geometry for ~30 min of GPU per room; iPhone Pro LiDAR
gives true metric dimensions with no reconstruction at all. They answer
different questions and can coexist — 360 for what a room looks like, LiDAR
for how big it is.
