#!/usr/bin/env python3
# Constructed using Claude; tables and columns verified against the
# react-vt-data warehouse (backend/Data/warehouse.duckdb).

"""Build a parcel-level flood exposure / assessed value table for one town -> CSV.

    python3 get_montpelier_flood.py WAREHOUSE.duckdb montpelier_parcels.csv
    python3 get_montpelier_flood.py WAREHOUSE.duckdb out.csv --town "Barre city"

Requires the `duckdb` package (the spatial extension is loaded at runtime).

Sources (all read from the react-vt-data DuckDB warehouse):
    VCGIParcels_geom / _info / _tax   VCGI statewide parcels and the town
                                      grand list (assessed values)
    VCGI_buildingFootprints_geom      VCGI building footprints
    FEMA_floodHazard_geom             FEMA digitized flood hazard areas

Observational unit: one residential parcel (grand list category
"Residential I" or "Residential II") with a positive assessed value.

Output columns
    object_id                    parcel ID
    category                     grand list category (Residential I / II)
    listed_real_value            total assessed value, $ (the outcome)
    land_value, improvements_value  components of listed_real_value, $
    acres                        parcel size
    n_homes                      residential footprints whose centroid falls
                                 inside the parcel
    in_sfha                      1 if any of those footprints intersects a
                                 FEMA Special Flood Hazard Area (zones A, AE,
                                 AH, AO); empty when n_homes = 0
    parcel_in_sfha               1 if the parcel boundary intersects an SFHA
                                 (looser definition, for a sensitivity check)

Caveats
    - Values are town assessments, not sale prices. Comparable within one
      town (single appraisal), not across towns.
    - Footprints carry no parcel ID; they are linked by centroid-in-parcel.
      Parcels with n_homes != 1 (duplexes, split outlines, vacant lots) are
      kept here; filter them in the analysis and report how many.
"""

import argparse
import sys

import duckdb

RESIDENTIAL_BUILDINGS = (
    "Single Family Dwelling",
    "Mobile Home",
    "Multi-Family Complex",
    "Multi-family Residence",
    "Residential Other",
)

QUERY = """
with parcels as (
    select g.object_id, g.geometry, i.category, t.listed_real_value,
           t.land_value, t.improvements_value, i.acres
    from VCGIParcels_geom g
    join VCGIParcels_tax t using (object_id)
    join VCGIParcels_info i using (object_id)
    where g.town = $town
      and i.category ilike 'residential%'
      and t.listed_real_value > 0
),
homes as (
    select st_geomfromwkb(geometry) as geom
    from VCGI_buildingFootprints_geom
    where town = $town and list_contains($building_types, building_type)
),
sfha as (
    select geometry from FEMA_floodHazard_geom
    where special_flood_hazard_zone
),
parcel_homes as (
    select p.object_id, h.geom
    from parcels p
    join homes h on st_contains(p.geometry, st_centroid(h.geom))
),
home_flags as (
    select object_id,
           exists (select 1 from sfha where st_intersects(sfha.geometry, geom))
               as home_in_sfha
    from parcel_homes
)
select p.object_id, p.category, p.listed_real_value, p.land_value,
       p.improvements_value, p.acres,
       count(f.object_id) as n_homes,
       case when count(f.object_id) > 0
            then bool_or(f.home_in_sfha)::int end as in_sfha,
       exists (select 1 from sfha where st_intersects(sfha.geometry, p.geometry))::int
           as parcel_in_sfha
from parcels p
left join home_flags f using (object_id)
group by all
order by p.object_id
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("warehouse", help="path to warehouse.duckdb")
    parser.add_argument("out", help="output CSV path")
    parser.add_argument(
        "--town",
        default="Montpelier city",
        help='exact town name as in VCGIParcels_geom, e.g. "Montpelier city"',
    )
    args = parser.parse_args()

    con = duckdb.connect(args.warehouse, read_only=True)
    con.sql("load spatial")
    towns = {
        r[0] for r in con.sql("select distinct town from VCGIParcels_geom").fetchall()
    }
    if args.town not in towns:
        close = sorted(t for t in towns if args.town.split()[0].lower() in t.lower())
        sys.exit(f"Unknown town {args.town!r}. Similar: {close}")

    rel = con.sql(
        QUERY,
        params={"town": args.town, "building_types": list(RESIDENTIAL_BUILDINGS)},
    )
    rel.write_csv(args.out)

    n, one_home, in_zone = con.sql(
        "select count(*), sum((n_homes = 1)::int), sum(in_sfha) from rel"
    ).fetchone()
    print(
        f"{args.town}: wrote {n} residential parcels to {args.out}; "
        f"{one_home} have exactly one home; {in_zone} have a home in the SFHA.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
