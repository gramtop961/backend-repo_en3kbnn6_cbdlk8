import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bson import ObjectId

from database import db, create_document, get_documents

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Utility
# -----------------------------
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


def now_ts():
    return datetime.now(timezone.utc)


# -----------------------------
# Schemas (simplified for MVP)
# -----------------------------
class UserPreferences(BaseModel):
    theme: str = "system"
    language: str = "en"


class UserProfile(BaseModel):
    uid: str
    displayName: Optional[str] = None
    email: Optional[str] = None
    photoURL: Optional[str] = None
    isAnonymous: bool = True
    createdAt: datetime = Field(default_factory=now_ts)
    lastSeen: datetime = Field(default_factory=now_ts)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    tier: str = "guest"  # guest | free | pro | enterprise


class RoomSecurity(BaseModel):
    visibility: str = "public"  # public | private | unlisted
    accessControl: str = "open"  # open | password | allowlist
    password: Optional[str] = None
    allowedEmails: Optional[List[str]] = None
    encryptionEnabled: bool = False
    encryptionKey: Optional[str] = None


class RoomSettings(BaseModel):
    maxUsers: int = 50
    maxFileSize: int = 100
    allowGuests: bool = True
    allowDownloads: bool = True
    allowForking: bool = True
    expiresAt: Optional[datetime] = None
    theme: Dict[str, Any] = Field(default_factory=dict)


class RoomLimits(BaseModel):
    notesCount: int = 0
    totalStorage: int = 0
    bandwidthUsed: int = 0


class RoomMetadata(BaseModel):
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    cover: Optional[str] = None
    createdBy: str
    createdAt: datetime = Field(default_factory=now_ts)
    lastActivityAt: datetime = Field(default_factory=now_ts)


class RoomCreate(BaseModel):
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    visibility: str = "public"


class RoomModel(BaseModel):
    id: Optional[str] = None
    slug: Optional[str] = None
    metadata: RoomMetadata
    security: RoomSecurity = Field(default_factory=RoomSecurity)
    settings: RoomSettings = Field(default_factory=RoomSettings)
    limits: RoomLimits = Field(default_factory=RoomLimits)
    collaborators: Dict[str, str] = Field(default_factory=dict)


# -----------------------------
# Root and health
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# -----------------------------
# Auth (Anonymous MVP)
# -----------------------------
class AnonymousAuthRequest(BaseModel):
    displayName: Optional[str] = None
    photoURL: Optional[str] = None


@app.post("/api/auth/anonymous")
def auth_anonymous(body: AnonymousAuthRequest):
    """Create or return an anonymous user profile"""
    import uuid
    uid = str(uuid.uuid4())

    profile = UserProfile(
        uid=uid,
        displayName=body.displayName or f"Guest-{uid[:6]}",
        photoURL=body.photoURL,
    )

    # Persist in DB
    if db is None:
        raise HTTPException(status_code=500, detail="Database is not configured")

    user_doc = profile.model_dump()
    user_doc["_id"] = uid
    user_doc["type"] = "anonymous"
    db["user"].insert_one(user_doc)

    return profile


@app.get("/api/auth/session")
def get_session(uid: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database is not configured")

    doc = db["user"].find_one({"_id": uid})
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")

    doc["id"] = doc.pop("_id")
    return doc


# -----------------------------
# Rooms CRUD
# -----------------------------
@app.get("/api/rooms")
def list_rooms(uid: Optional[str] = None):
    if db is None:
        raise HTTPException(status_code=500, detail="Database is not configured")

    query = {}
    if uid:
        query = {"$or": [
            {"metadata.createdBy": uid},
            {f"collaborators.{uid}": {"$exists": True}}
        ]}

    rooms = []
    for r in db["room"].find(query).sort("metadata.lastActivityAt", -1).limit(50):
        r["id"] = str(r.pop("_id"))
        rooms.append(r)
    return {"rooms": rooms}


@app.post("/api/rooms")
def create_room(room: RoomCreate, uid: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database is not configured")

    metadata = RoomMetadata(
        name=room.name,
        description=room.description,
        icon=room.icon,
        createdBy=uid,
    )
    model = RoomModel(
        slug=None,
        metadata=metadata,
        security=RoomSecurity(visibility=room.visibility),
    ).model_dump()

    inserted_id = db["room"].insert_one(model).inserted_id
    db["room"].update_one({"_id": inserted_id}, {"$set": {"collaborators." + uid: "owner"}})

    doc = db["room"].find_one({"_id": inserted_id})
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.get("/api/rooms/{room_id}")
def get_room(room_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database is not configured")

    try:
        _id = ObjectId(room_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid room id")

    doc = db["room"].find_one({"_id": _id})
    if not doc:
        raise HTTPException(status_code=404, detail="Room not found")

    doc["id"] = str(doc.pop("_id"))
    return doc


class RoomUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    visibility: Optional[str] = None


@app.put("/api/rooms/{room_id}")
def update_room(room_id: str, payload: RoomUpdate, uid: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database is not configured")

    try:
        _id = ObjectId(room_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid room id")

    updates: Dict[str, Any] = {}
    if payload.name is not None:
        updates["metadata.name"] = payload.name
    if payload.description is not None:
        updates["metadata.description"] = payload.description
    if payload.icon is not None:
        updates["metadata.icon"] = payload.icon
    if payload.visibility is not None:
        updates["security.visibility"] = payload.visibility

    updates["metadata.lastActivityAt"] = now_ts()

    result = db["room"].update_one({"_id": _id}, {"$set": updates})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Room not found")

    doc = db["room"].find_one({"_id": _id})
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.delete("/api/rooms/{room_id}")
def delete_room(room_id: str, uid: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database is not configured")

    try:
        _id = ObjectId(room_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid room id")

    result = db["room"].delete_one({"_id": _id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"ok": True}


# -----------------------------
# Presence via WebSocket (ephemeral)
# -----------------------------
class PresencePayload(BaseModel):
    userId: str
    cursor: Optional[Dict[str, float]] = None
    selection: Optional[List[str]] = None
    viewport: Optional[Dict[str, float]] = None
    status: Optional[str] = None
    device: Optional[str] = None
    lastActivity: Optional[datetime] = None


class RoomConnectionManager:
    def __init__(self):
        self.active: Dict[str, List[WebSocket]] = {}

    async def connect(self, room_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active.setdefault(room_id, []).append(websocket)

    def disconnect(self, room_id: str, websocket: WebSocket):
        if room_id in self.active and websocket in self.active[room_id]:
            self.active[room_id].remove(websocket)
            if not self.active[room_id]:
                self.active.pop(room_id, None)

    async def broadcast(self, room_id: str, message: dict):
        for ws in list(self.active.get(room_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                # Drop broken connections
                self.disconnect(room_id, ws)


manager = RoomConnectionManager()


@app.websocket("/ws/rooms/{room_id}/{user_id}")
async def ws_room(websocket: WebSocket, room_id: str, user_id: str):
    await manager.connect(room_id, websocket)
    try:
        # Notify others that user joined
        await manager.broadcast(room_id, {"type": "user-joined", "userId": user_id})
        while True:
            data = await websocket.receive_json()
            # Broadcast payloads to all peers in the same room
            await manager.broadcast(room_id, {"type": "presence", "userId": user_id, "data": data})
    except WebSocketDisconnect:
        manager.disconnect(room_id, websocket)
        await manager.broadcast(room_id, {"type": "user-left", "userId": user_id})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
