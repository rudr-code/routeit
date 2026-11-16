from __future__ import annotations

import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.db import Database, get_db
from backend.app.schemas import VehicleCreate, VehicleResponse

router = APIRouter(prefix="/v1/vehicles", tags=["Vehicles"])


class VehicleUpdate(BaseModel):
    available: Optional[bool] = None
    capacity: Optional[int] = None
    activation_cost_paise: Optional[int] = None


@router.post("", response_model=VehicleResponse, status_code=201)
def create_vehicle(v: VehicleCreate, db: Database = Depends(get_db)):
    v_id = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO vehicle (id, code, capacity, activation_cost_paise, available, profile)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (v_id, v.code, v.capacity, v.activation_cost_paise, 1 if v.available else 0, v.profile),
    )
    db.commit()
    return VehicleResponse(
        id=v_id,
        code=v.code,
        capacity=v.capacity,
        activation_cost_paise=v.activation_cost_paise,
        available=v.available,
        profile=v.profile,
    )


@router.get("", response_model=List[VehicleResponse])
def list_vehicles(db: Database = Depends(get_db)):
    rows = db.fetchall("SELECT * FROM vehicle")
    return [
        VehicleResponse(
            id=r["id"],
            code=r["code"],
            capacity=r["capacity"],
            activation_cost_paise=r["activation_cost_paise"],
            available=bool(r["available"]),
            profile=r["profile"],
        )
        for r in rows
    ]


@router.patch("/{id}", response_model=VehicleResponse)
def update_vehicle(id: str, u: VehicleUpdate, db: Database = Depends(get_db)):
    r = db.fetchone("SELECT * FROM vehicle WHERE id = ?", (id,))
    if not r:
        raise HTTPException(status_code=404, detail="Vehicle not found")

    new_avail = r["available"] if u.available is None else (1 if u.available else 0)
    new_cap = r["capacity"] if u.capacity is None else u.capacity
    new_act = r["activation_cost_paise"] if u.activation_cost_paise is None else u.activation_cost_paise

    db.execute(
        "UPDATE vehicle SET available = ?, capacity = ?, activation_cost_paise = ? WHERE id = ?",
        (new_avail, new_cap, new_act, id),
    )
    db.commit()
    return VehicleResponse(
        id=id,
        code=r["code"],
        capacity=new_cap,
        activation_cost_paise=new_act,
        available=bool(new_avail),
        profile=r["profile"],
    )
