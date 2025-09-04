from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="qc-compliance-tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"project": "qc-compliance-tracker", "status": "operational"}

@app.get("/health")
def health():
    return {"status": "healthy"}
