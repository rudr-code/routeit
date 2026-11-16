from __future__ import annotations

import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException

from backend.app.db import Database, get_db
from backend.app.schemas import DepotCreate, DepotResponse

router = APIRouter(prefix="/v1/depots", tags=["Depots"])


@router.post("", response_model=DepotResponse, status_code=201)
def create_depot(depot: DepotCreate, db: Database = Depends(get_db)):
    depot_id = str(uuid.uuid4())
    snapped = f"node_{int(depot.lat * 1000)}_{int(depot.lon * 1000)}"
    db.execute(
        """
        INSERT INTO depot (id, name, lat, lon, snapped_node_id, opening_time, closing_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (depot_id, depot.name, depot.lat, depot.lon, snapped, depot.opening_time, depot.closing_time),
    )
    db.commit()
    return DepotResponse(
        id=depot_id,
        name=depot.name,
        lat=depot.lat,
        lon=depot.lon,
        snapped_node_id=snapped,
        opening_time=depot.opening_time,
        closing_time=depot.closing_time,
    )


@router.get("", response_model=List[DepotResponse])
def list_depots(db: Database = Depends(get_db)):
    rows = db.fetchall("SELECT * FROM depot")
    return [
        DepotResponse(
            id=r["id"],
            name=r["name"],
            lat=r["lat"],
            lon=r["lon"],
            snapped_node_id=r["snapped_node_id"],
            opening_time=r["opening_time"],
            closing_time=r["closing_time"],
        )
        for r in rows
    ]


@router.get("/{id}", response_model=DepotResponse)
def get_depot(id: str, db: Database = Depends(get_db)):
    r = db.fetchone("SELECT * FROM depot WHERE id = ?", (id,))
    if not r:
        raise HTTPException(status_code=404, detail="Depot not found")
    return DepotResponse(
        id=r["id"],
        name=r["name"],
        lat=r["lat"],
        lon=r["lon"],
        snapped_node_id=r["snapped_node_id"],
        opening_time=r["opening_time"],
        closing_time=r["closing_time"],
    )
