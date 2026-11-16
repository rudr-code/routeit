from __future__ import annotations

import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException

from backend.app.db import Database, get_db
from backend.app.schemas import CostProfileCreate, CostProfileResponse

router = APIRouter(prefix="/v1/cost-profiles", tags=["Cost Profiles"])


@router.post("", response_model=CostProfileResponse, status_code=201)
def create_cost_profile(p: CostProfileCreate, db: Database = Depends(get_db)):
    pid = str(uuid.uuid4())
    db.execute(
        """
        INSERT INTO cost_profile (
            id, name, distance_rate_num, distance_rate_den, time_rate_num, time_rate_den, rounding_policy_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (pid, p.name, p.distance_rate_num, p.distance_rate_den, p.time_rate_num, p.time_rate_den, p.rounding_policy_version),
    )
    db.commit()
    return CostProfileResponse(
        id=pid,
        name=p.name,
        version="v1",
        distance_rate_num=p.distance_rate_num,
        distance_rate_den=p.distance_rate_den,
        time_rate_num=p.time_rate_num,
        time_rate_den=p.time_rate_den,
        rounding_policy_version=p.rounding_policy_version,
    )


@router.get("", response_model=List[CostProfileResponse])
def list_cost_profiles(db: Database = Depends(get_db)):
    rows = db.fetchall("SELECT * FROM cost_profile")
    return [
        CostProfileResponse(
            id=r["id"],
            name=r["name"],
            version=r["version"],
            distance_rate_num=r["distance_rate_num"],
            distance_rate_den=r["distance_rate_den"],
            time_rate_num=r["time_rate_num"],
            time_rate_den=r["time_rate_den"],
            rounding_policy_version=r["rounding_policy_version"],
        )
        for r in rows
    ]
