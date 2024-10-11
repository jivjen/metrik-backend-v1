import asyncio
import os
import uuid
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List
from researcher import research
import logging
from pymongo import MongoClient
from datetime import datetime
from models import ResearchStatus
from logging_config import setup_logger

app = FastAPI()

# MongoDB connection
client = MongoClient("mongodb+srv://hireloom_admin:QuickVogue%40123@quickvogue.x163n.mongodb.net/?retryWrites=true&w=majority&appName=quickvogue")
db = client.research_jobs

class ResearchRequest(BaseModel):
    user_input: str

class JobStatus(BaseModel):
    status: ResearchStatus
    details: str
    sub_statuses: List[Dict[str, str]] = []

def update_job_status(job_id: str, status: ResearchStatus, details: str, sub_status: Dict[str, str] = None):
    update_data = {
        "status": status,
        "details": details,
        "updated_at": datetime.utcnow()
    }
    if sub_status:
        update_data["$push"] = {"sub_statuses": sub_status}
    
    db.job_statuses.update_one(
        {"job_id": job_id},
        {"$set": update_data},
        upsert=True
    )
    logger = logging.getLogger()
    logger.info(f"Job status updated: {status} - {details}")

async def run_research(job_id: str, user_input: str):
    logger = setup_logger(job_id)
    update_job_status(job_id, ResearchStatus.STARTED, "Starting research")
    
    try:
        logger.info(f"Starting research for job {job_id}")
        logger.debug(f"User input: {user_input}")
        result = await research(user_input, lambda status, details: update_job_status(job_id, status, details))
        logger.info(f"Research completed for job {job_id}")
        logger.debug(f"Research result: {result}")
        
        # Store result in MongoDB
        db.results.insert_one({
            "job_id": job_id,
            "user_input": user_input,
            "result": result,
            "timestamp": datetime.utcnow()
        })
        logger.info(f"Research result stored in MongoDB for job {job_id}")
        
        update_job_status(job_id, ResearchStatus.COMPLETED, "Research finished")
    except Exception as e:
        logger.error(f"Error in research for job {job_id}: {str(e)}", exc_info=True)
        update_job_status(job_id, ResearchStatus.FAILED, str(e))
    finally:
        logging.shutdown()
        logger.info(f"Logging finished for job {job_id}")

@app.post("/start_job")
async def start_job(request: ResearchRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    logger = setup_logger(job_id)
    logger.info(f"New job started: {job_id}")
    logger.debug(f"User input: {request.user_input}")
    background_tasks.add_task(run_research, job_id, request.user_input)
    return {"job_id": job_id, "status": ResearchStatus.STARTED}

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    logger = logging.getLogger(job_id)
    logger.info(f"Job status requested for job {job_id}")
    job_status = db.job_statuses.find_one({"job_id": job_id})
    if not job_status:
        logger.warning(f"Job status not found for job {job_id}")
        return {"status": "Not Found"}
    logger.debug(f"Job status: {job_status}")
    return {
        "status": job_status["status"],
        "details": job_status["details"],
        "sub_statuses": job_status.get("sub_statuses", [])
    }

@app.get("/job_result/{job_id}")
async def get_job_result(job_id: str):
    logger = logging.getLogger(job_id)
    logger.info(f"Job result requested for job {job_id}")
    result = db.results.find_one({"job_id": job_id})
    if result:
        logger.debug(f"Job result: {result}")
        return {"result": result["result"]}
    logger.warning(f"Job result not found for job {job_id}")
    return {"status": "Result not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
