from __future__ import annotations

import time
import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException

from backend.app.db import Database, get_db
from backend.app.schemas import (
    ConfirmLocationRequest,
    CustomerCreate,
    CustomerResponse,
)

router = APIRouter(prefix="/v1/customers", tags=["Customers"])


@router.post("", response_model=CustomerResponse, status_code=201)
def create_customer(c: CustomerCreate, db: Database = Depends(get_db)):
    cid = str(uuid.uuid4())
    snapped = f"node_{int(c.lat * 1000)}_{int(c.lon * 1000)}"
    db.execute(
        """
        INSERT INTO customer (
            id, name, lat, lon, snapped_node_id, address_text, address_source,
            delivery_instructions, confirmation_status
        ) VALUES (?, ?, ?, ?, ?, ?, 'DISPATCHER_TYPED', ?, 'UNCONFIRMED')
        """,
        (cid, c.name, c.lat, c.lon, snapped, c.address_text, c.delivery_instructions),
    )
    db.commit()
    return CustomerResponse(
        id=cid,
        name=c.name,
        lat=c.lat,
        lon=c.lon,
        snapped_node_id=snapped,
        address_text=c.address_text,
        address_source="DISPATCHER_TYPED",
        delivery_instructions=c.delivery_instructions,
        confirmation_status="UNCONFIRMED",
    )


@router.get("", response_model=List[CustomerResponse])
def list_customers(db: Database = Depends(get_db)):
    rows = db.fetchall("SELECT * FROM customer")
    return [
        CustomerResponse(
            id=r["id"],
            name=r["name"],
            lat=r["lat"],
            lon=r["lon"],
            snapped_node_id=r["snapped_node_id"],
            address_text=r["address_text"],
            address_source=r["address_source"],
            delivery_instructions=r["delivery_instructions"] or "",
            confirmation_status=r["confirmation_status"],
        )
        for r in rows
    ]


@router.post("/{id}/confirm-location", response_model=CustomerResponse)
def confirm_location(id: str, req: ConfirmLocationRequest, db: Database = Depends(get_db)):
    """
    AC-13 & Phase 4 Section 8:
    Dispatcher moves/confirms pin, updates coordinates and instructions, marks CONFIRMED.
    """
    r = db.fetchone("SELECT * FROM customer WHERE id = ?", (id,))
    if not r:
        raise HTTPException(status_code=404, detail="Customer not found")

    snapped = f"node_{int(req.lat * 1000)}_{int(req.lon * 1000)}"
    instructions = req.delivery_instructions if req.delivery_instructions else (r["delivery_instructions"] or "")
    now_ts = time.time()

    db.execute(
        """
        UPDATE customer
        SET lat = ?, lon = ?, snapped_node_id = ?, delivery_instructions = ?,
            confirmation_status = 'CONFIRMED', confirmed_at = ?
        WHERE id = ?
        """,
        (req.lat, req.lon, snapped, instructions, now_ts, id),
    )
    db.commit()

    return CustomerResponse(
        id=id,
        name=r["name"],
        lat=req.lat,
        lon=req.lon,
        snapped_node_id=snapped,
        address_text=r["address_text"],
        address_source=r["address_source"],
        delivery_instructions=instructions,
        confirmation_status="CONFIRMED",
    )
