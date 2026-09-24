# Archived: own-photo capture plan (obsolete)

The project was first designed around a dataset photographed at two gyms. This folder keeps
that plan as it stood, so the design history stays readable:

- `CAPTURE_PROTOCOL.md` — 24-load schedule, camera rules, and a split fixed by load.
- `loads.csv` — the answer key for the loaded-weight metric. Header only: no photo was taken.

**Why it was dropped (2026-09-24):** collecting and labelling ~225 photos was the slowest step,
and a public dataset with per-weight boxes turned up (ProjektCiezary *Weightlifting Plates*,
CC BY 4.0). After an audit it replaced the capture plan. The loaded-weight metric went with it:
it needed a load recorded independently of the labels, and the public dataset has none.

The pipeline code for this plan (burst sorter, `loads.csv` validator, uploader) was removed from
`src/` in the same change; it is in history at commit `ac1b874`.
