import asyncio
import os
import uuid
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any
from researcher import research
import logging
from pymongo import MongoClient
from datetime import datetime

app = FastAPI()

# MongoDB connection
client = MongoClient("mongodb+srv://hireloom_admin:QuickVogue%40123@quickvogue.x163n.mongodb.net/?retryWrites=true&w=majority&appName=quickvogue")
db = client.research_jobs

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

class ResearchRequest(BaseModel):
    user_input: str

class JobStatus(BaseModel):
    status: str
    details: str

def update_job_status(job_id: str, status: str, details: str):
    db.job_statuses.update_one(
        {"job_id": job_id},
        {"$set": {"status": status, "details": details, "updated_at": datetime.utcnow()}},
        upsert=True
    )

def setup_logger(job_id: str):
    logger = logging.getLogger(job_id)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"logs/{job_id}.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

async def run_research(job_id: str, user_input: str):
    logger = setup_logger(job_id)
    update_job_status(job_id, "In Progress", "Starting research")
    
    try:
        logger.info(f"Starting research for job {job_id}")
        result = await research(user_input)
        logger.info(f"Research completed for job {job_id}")
        
        # Store result in MongoDB
        db.results.insert_one({
            "job_id": job_id,
            "user_input": user_input,
            "result": result,
            "timestamp": datetime.utcnow()
        })
        
        update_job_status(job_id, "Completed", "Research finished")
    except Exception as e:
        logger.error(f"Error in research for job {job_id}: {str(e)}")
        update_job_status(job_id, "Failed", str(e))

@app.post("/start_job")
async def start_job(request: ResearchRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_research, job_id, request.user_input)
    return {"job_id": job_id, "status": "Started"}

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    job_status = db.job_statuses.find_one({"job_id": job_id})
    if not job_status:
        return {"status": "Not Found"}
    return {"status": job_status["status"], "details": job_status["details"]}

@app.get("/job_result/{job_id}")
async def get_job_result(job_id: str):
    result = db.results.find_one({"job_id": job_id})
    if result:
        return {"result": result["result"]}
    return {"status": "Result not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
