"""
Metrics endpoint for Prometheus-style metrics (simplified JSON version).

Provides:
- Average processing times
- Job counts (finished/failed)
- Model usage statistics
"""
from typing import Dict
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json
from pathlib import Path

# Simple in-memory metrics store
# In production, use Redis or a proper metrics backend
_metrics_store: Dict[str, float] = {
    "jobs_finished_total": 0.0,
    "jobs_failed_total": 0.0,
    "rvc_time_sum": 0.0,
    "rvc_time_count": 0.0,
    "dsp_time_sum": 0.0,
    "dsp_time_count": 0.0,
    "total_time_sum": 0.0,
    "total_time_count": 0.0,
}


def update_metrics(metrics: dict):
    """
    Update metrics store with job metrics.
    
    Args:
        metrics: Dictionary with processing_time_* keys
    """
    if "processing_time_total" in metrics:
        _metrics_store["total_time_sum"] += metrics["processing_time_total"]
        _metrics_store["total_time_count"] += 1.0
    
    if "processing_time_rvc" in metrics and metrics["processing_time_rvc"] > 0:
        _metrics_store["rvc_time_sum"] += metrics["processing_time_rvc"]
        _metrics_store["rvc_time_count"] += 1.0
    
    if "processing_time_dsp" in metrics:
        _metrics_store["dsp_time_sum"] += metrics["processing_time_dsp"]
        _metrics_store["dsp_time_count"] += 1.0


def increment_job_count(success: bool = True):
    """Increment job count."""
    if success:
        _metrics_store["jobs_finished_total"] += 1.0
    else:
        _metrics_store["jobs_failed_total"] += 1.0


def get_metrics() -> Dict[str, float]:
    """
    Get current metrics.
    
    Returns:
        Dictionary with all metrics
    """
    metrics = _metrics_store.copy()
    
    # Compute averages
    if metrics["rvc_time_count"] > 0:
        metrics["rvc_time_avg"] = metrics["rvc_time_sum"] / metrics["rvc_time_count"]
    else:
        metrics["rvc_time_avg"] = 0.0
    
    if metrics["dsp_time_count"] > 0:
        metrics["dsp_time_avg"] = metrics["dsp_time_sum"] / metrics["dsp_time_count"]
    else:
        metrics["dsp_time_avg"] = 0.0
    
    if metrics["total_time_count"] > 0:
        metrics["total_time_avg"] = metrics["total_time_sum"] / metrics["total_time_count"]
    else:
        metrics["total_time_avg"] = 0.0
    
    return metrics


def register_metrics_endpoint(app: FastAPI):
    """Register metrics endpoint with FastAPI app."""
    
    @app.get("/metrics")
    async def metrics():
        """
        Prometheus-style metrics endpoint (simplified JSON version).
        
        Returns:
            JSON with all metrics
        """
        return JSONResponse(get_metrics())

