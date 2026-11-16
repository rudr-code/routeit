from __future__ import annotations

import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException

from backend.app.db import Database, get_db
from backend.app.schemas import OrderCreate, OrderResponse

router = APIRouter(prefix="/v1/orders", tags=["Orders"])


@router.post("", response_model=OrderResponse, status_code=201)
def create_order(o: OrderCreate, db: Database = Depends(get_db)):
    # Verify customer exists and is confirmed
    cust = db.fetchone("SELECT * FROM customer WHERE id = ?", (o.customer_id,))
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")

    order_id = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO delivery_order (
            id, customer_id, external_ref, demand_units, service_seconds,
            window_open, window_close, window_type, priority, sla_rate_num, sla_rate_den, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'NEW')
        """,
        (
            order_id,
            o.customer_id,
            o.external_ref,
            o.demand_units,
            o.service_seconds,
            o.window_open,
            o.window_close,
            o.window_type,
            o.priority,
            o.sla_rate_num,
            o.sla_rate_den,
        ),
    )
    db.commit()

    return OrderResponse(
        id=order_id,
        customer_id=o.customer_id,
        external_ref=o.external_ref,
        demand_units=o.demand_units,
        service_seconds=o.service_seconds,
        window_open=o.window_open,
        window_close=o.window_close,
        window_type=o.window_type,
        priority=o.priority,
        sla_rate_num=o.sla_rate_num,
        sla_rate_den=o.sla_rate_den,
        status="NEW",
    )


@router.get("", response_model=List[OrderResponse])
def list_orders(db: Database = Depends(get_db)):
    rows = db.fetchall("SELECT * FROM delivery_order")
    return [
        OrderResponse(
            id=r["id"],
            customer_id=r["customer_id"],
            external_ref=r["external_ref"],
            demand_units=r["demand_units"],
            service_seconds=r["service_seconds"],
            window_open=r["window_open"],
            window_close=r["window_close"],
            window_type=r["window_type"],
            priority=r["priority"],
            sla_rate_num=r["sla_rate_num"],
            sla_rate_den=r["sla_rate_den"],
            status=r["status"],
        )
        for r in rows
    ]


@router.delete("/{id}", status_code=204)
def cancel_order(id: str, db: Database = Depends(get_db)):
    r = db.fetchone("SELECT * FROM delivery_order WHERE id = ?", (id,))
    if not r:
        raise HTTPException(status_code=404, detail="Order not found")
    db.execute("UPDATE delivery_order SET status = 'CANCELLED' WHERE id = ?", (id,))
    db.commit()
    return None
