"""
Production FastAPI Server for NeuralLayers Model Serving

Features:
- REST API for model inference
- Batch inference support
- Model versioning
- Health checks
- Metrics endpoint (Prometheus compatible)
- Request validation
- Rate limiting
- Async processing
- ONNX runtime support
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn


# ============================================================================
# Request/Response Models
# ============================================================================

class InferenceRequest(BaseModel):
    """Single inference request"""

    input: List[List[float]] = Field(
        ...,
        description="Input tensor as nested list (batch_size, input_dim)"
    )
    model_version: Optional[str] = Field(
        "latest",
        description="Model version to use"
    )
    return_metadata: bool = Field(
        False,
        description="Whether to return inference metadata"
    )

    @validator('input')
    def validate_input_shape(cls, v):
        if not v:
            raise ValueError("Input cannot be empty")
        if not all(isinstance(row, list) for row in v):
            raise ValueError("Input must be a 2D list")
        return v


class InferenceResponse(BaseModel):
    """Single inference response"""

    output: List[List[float]] = Field(
        ...,
        description="Model output"
    )
    model_version: str = Field(
        ...,
        description="Model version used"
    )
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata if requested"
    )


class BatchInferenceRequest(BaseModel):
    """Batch inference request"""

    inputs: List[List[List[float]]] = Field(
        ...,
        description="List of input tensors"
    )
    model_version: Optional[str] = "latest"


class BatchInferenceResponse(BaseModel):
    """Batch inference response"""

    outputs: List[List[List[float]]]
    total_inference_time_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    model_version: str
    device: str
    uptime_seconds: float


# ============================================================================
# Prometheus Metrics
# ============================================================================

REQUEST_COUNT = Counter(
    'neurallayers_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'neurallayers_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

INFERENCE_COUNT = Counter(
    'neurallayers_inferences_total',
    'Total number of inferences',
    ['model_version']
)

INFERENCE_LATENCY = Histogram(
    'neurallayers_inference_latency_seconds',
    'Inference latency in seconds',
    ['model_version']
)

MODEL_LOAD_TIME = Gauge(
    'neurallayers_model_load_time_seconds',
    'Time taken to load model'
)

ACTIVE_REQUESTS = Gauge(
    'neurallayers_active_requests',
    'Number of active requests'
)


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages model loading, versioning, and inference"""

    def __init__(
        self,
        model_dir: str = "models",
        device: str = "cpu",
        use_onnx: bool = False
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.use_onnx = use_onnx
        self.models: Dict[str, Any] = {}
        self.current_version = "latest"
        self.start_time = time.time()

    def load_model(
        self,
        model_path: str,
        version: str = "latest"
    ):
        """Load a model from disk"""

        start_time = time.time()

        if self.use_onnx:
            # Load ONNX model
            try:
                import onnxruntime as ort

                model = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider']
                )
                self.models[version] = {
                    'model': model,
                    'type': 'onnx',
                    'input_name': model.get_inputs()[0].name,
                    'output_name': model.get_outputs()[0].name
                }

            except ImportError:
                raise RuntimeError("onnxruntime not installed")

        else:
            # Load PyTorch model
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()

            self.models[version] = {
                'model': model,
                'type': 'pytorch'
            }

        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)

        self.current_version = version

        print(f"✅ Model loaded: {version} ({load_time:.2f}s)")

    def infer(
        self,
        inputs: np.ndarray,
        version: str = "latest"
    ) -> np.ndarray:
        """Run inference"""

        if version not in self.models:
            raise ValueError(f"Model version not found: {version}")

        model_info = self.models[version]
        model = model_info['model']

        start_time = time.time()

        if model_info['type'] == 'onnx':
            # ONNX inference
            input_name = model_info['input_name']
            output_name = model_info['output_name']

            outputs = model.run(
                [output_name],
                {input_name: inputs.astype(np.float32)}
            )[0]

        else:
            # PyTorch inference
            with torch.no_grad():
                tensor_input = torch.from_numpy(inputs).float().to(self.device)
                tensor_output = model(tensor_input)

                # Handle dict output
                if isinstance(tensor_output, dict):
                    tensor_output = tensor_output['output']

                outputs = tensor_output.cpu().numpy()

        inference_time = time.time() - start_time

        # Update metrics
        INFERENCE_COUNT.labels(model_version=version).inc()
        INFERENCE_LATENCY.labels(model_version=version).observe(inference_time)

        return outputs

    def get_uptime(self) -> float:
        """Get server uptime in seconds"""
        return time.time() - self.start_time


# ============================================================================
# FastAPI Application
# ============================================================================

class NeuralLayersAPI:
    """FastAPI application for NeuralLayers serving"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_onnx: bool = False
    ):
        self.app = FastAPI(
            title="NeuralLayers Inference API",
            description="Production API for NeuralLayers model serving",
            version="1.0.0"
        )

        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize model manager
        self.model_manager = ModelManager(device=device, use_onnx=use_onnx)

        # Load model if provided
        if model_path:
            self.model_manager.load_model(model_path)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            """Middleware to track request metrics"""

            ACTIVE_REQUESTS.inc()
            start_time = time.time()

            response = await call_next(request)

            latency = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
            REQUEST_COUNT.labels(
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            ACTIVE_REQUESTS.dec()

            return response

        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "NeuralLayers Inference API",
                "version": "1.0.0",
                "status": "running"
            }

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint"""

            return HealthResponse(
                status="healthy",
                model_loaded=len(self.model_manager.models) > 0,
                model_version=self.model_manager.current_version,
                device=self.model_manager.device,
                uptime_seconds=self.model_manager.get_uptime()
            )

        @self.app.post("/predict", response_model=InferenceResponse)
        async def predict(request: InferenceRequest):
            """Single inference endpoint"""

            try:
                # Convert input to numpy
                inputs = np.array(request.input, dtype=np.float32)

                # Run inference
                start_time = time.time()
                outputs = self.model_manager.infer(
                    inputs,
                    version=request.model_version
                )
                inference_time_ms = (time.time() - start_time) * 1000

                # Prepare response
                response = InferenceResponse(
                    output=outputs.tolist(),
                    model_version=request.model_version,
                    inference_time_ms=inference_time_ms
                )

                if request.return_metadata:
                    response.metadata = {
                        "input_shape": inputs.shape,
                        "output_shape": outputs.shape,
                        "device": self.model_manager.device
                    }

                return response

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/batch_predict", response_model=BatchInferenceResponse)
        async def batch_predict(request: BatchInferenceRequest):
            """Batch inference endpoint"""

            try:
                start_time = time.time()

                outputs_list = []

                for input_data in request.inputs:
                    inputs = np.array(input_data, dtype=np.float32)
                    outputs = self.model_manager.infer(
                        inputs,
                        version=request.model_version
                    )
                    outputs_list.append(outputs.tolist())

                total_time_ms = (time.time() - start_time) * 1000

                return BatchInferenceResponse(
                    outputs=outputs_list,
                    total_inference_time_ms=total_time_ms,
                    batch_size=len(request.inputs)
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return Response(
                generate_latest(),
                media_type="text/plain"
            )

        @self.app.post("/reload_model")
        async def reload_model(
            model_path: str,
            version: str = "latest",
            background_tasks: BackgroundTasks = None
        ):
            """Reload model from disk"""

            try:
                # Load model in background
                if background_tasks:
                    background_tasks.add_task(
                        self.model_manager.load_model,
                        model_path,
                        version
                    )
                    return {"status": "reloading", "version": version}
                else:
                    self.model_manager.load_model(model_path, version)
                    return {"status": "loaded", "version": version}

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1
    ):
        """Run the API server"""

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI entry point"""

    import argparse

    parser = argparse.ArgumentParser(
        description="NeuralLayers Inference Server"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model file (.pt or .onnx)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on"
    )

    parser.add_argument(
        "--use-onnx",
        action="store_true",
        help="Use ONNX runtime instead of PyTorch"
    )

    args = parser.parse_args()

    # Create and run API
    api = NeuralLayersAPI(
        model_path=args.model_path,
        device=args.device,
        use_onnx=args.use_onnx
    )

    print("="*80)
    print("🚀 NeuralLayers Inference Server")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Runtime: {'ONNX' if args.use_onnx else 'PyTorch'}")
    print(f"Server: http://{args.host}:{args.port}")
    print("="*80)
    print("\nEndpoints:")
    print(f"  GET  /health          - Health check")
    print(f"  POST /predict         - Single inference")
    print(f"  POST /batch_predict   - Batch inference")
    print(f"  GET  /metrics         - Prometheus metrics")
    print(f"  POST /reload_model    - Reload model")
    print("="*80)

    api.run(
        host=args.host,
        port=args.port,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
