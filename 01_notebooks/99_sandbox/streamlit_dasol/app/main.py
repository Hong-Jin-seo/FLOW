from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.chat import router as chat_router
from app.routes.curate import router as curate_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="스킨 큐레이터 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router,   prefix="/api")   # /api/chat
app.include_router(curate_router, prefix="/api")   # /api/curate


@app.get("/")
def root():
    return {"status": "ok", "service": "스킨 큐레이터"}
