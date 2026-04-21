from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from bson import ObjectId
import motor.motor_asyncio
import bcrypt
import jwt
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB setup
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "taxi_queue_db")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Collections
users_collection = db["users"]
zones_collection = db["zones"]
queue_collection = db["queue"]
rides_collection = db["rides"]
notifications_collection = db["notifications"]

# JWT Config
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days

# Admin code for registration
ADMIN_CODE = os.getenv("ADMIN_CODE", "BAURAULAC2024")

app = FastAPI(title="Taxi Queue API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ============ MODELS ============

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str = "driver"
    vehicle_class: Optional[str] = None
    driver_photo: Optional[str] = None

class AdminRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    admin_code: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class GoogleCallback(BaseModel):
    session_id: str
    role: str = "driver"

class ZoneCreate(BaseModel):
    name: str
    latitude: float
    longitude: float
    radius: float = 100

class PolygonUpdate(BaseModel):
    polygon: List[Dict[str, float]]

class QueueJoin(BaseModel):
    zone_id: str
    latitude: float
    longitude: float

class QueueLeave(BaseModel):
    zone_id: str

class LocationUpdate(BaseModel):
    latitude: float
    longitude: float

class RideRequest(BaseModel):
    zone_id: str
    vehicle_class: Optional[str] = None
    driver_id: Optional[str] = None
    customer_info: Optional[str] = None

class RideStart(BaseModel):
    destination: Optional[str] = None

class RideComplete(BaseModel):
    price: float
    commission: float = 0
    ride_type: str = "city"
    destination: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class BlockDriver(BaseModel):
    reason: Optional[str] = None

class VehicleClassUpdate(BaseModel):
    vehicle_classes: List[str]

class NotificationRegister(BaseModel):
    user_id: str
    push_token: str
    platform: str

# ============ HELPERS ============

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        user["user_id"] = str(user["_id"])
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def point_in_polygon(point: Dict[str, float], polygon: List[Dict[str, float]]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm"""
    if len(polygon) < 3:
        return False
    
    x, y = point['lng'], point['lat']
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]['lng'], polygon[i]['lat']
        xj, yj = polygon[j]['lng'], polygon[j]['lat']
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside

def serialize_user(user: dict) -> dict:
    return {
        "user_id": str(user["_id"]),
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "vehicle_class": user.get("vehicle_class"),
        "driver_photo": user.get("driver_photo"),
        "is_blocked": user.get("is_blocked", False),
    }

# ============ STARTUP ============

@app.on_event("startup")
async def startup_event():
    # Create indexes
    await users_collection.create_index("email", unique=True)
    await zones_collection.create_index("zone_id", unique=True)
    await queue_collection.create_index([("zone_id", 1), ("driver_id", 1)])
    
    # Create default admin if not exists
    admin = await users_collection.find_one({"email": "admin@bauraulac.ch"})
    if not admin:
        await users_collection.insert_one({
            "email": "admin@bauraulac.ch",
            "password": hash_password("Zurich2025"),
            "name": "Admin",
            "role": "admin",
            "created_at": datetime.utcnow(),
        })
    
    # Create default hotel if not exists
    hotel = await users_collection.find_one({"email": "taxi@bauraulac.ch"})
    if not hotel:
        hotel_result = await users_collection.insert_one({
            "email": "taxi@bauraulac.ch",
            "password": hash_password("Marguita2025"),
            "name": "Baur au Lac",
            "role": "hotel",
            "created_at": datetime.utcnow(),
        })
        hotel_id = str(hotel_result.inserted_id)
        logger.info(f"Created default hotel account: taxi@bauraulac.ch / Marguita2025")
        
        # Create default zone for hotel
        existing_zone = await zones_collection.find_one({"zone_id": "zone_baur_au_lac"})
        if not existing_zone:
            await zones_collection.insert_one({
                "zone_id": "zone_baur_au_lac",
                "hotel_user_id": hotel_id,
                "name": "Baur au Lac",
                "latitude": 47.36735048307041,
                "longitude": 8.540770835142638,
                "radius": None,
                "polygon": [
                    {"lat": 47.367800793449014, "lng": 8.53916359557819},
                    {"lat": 47.366525972979694, "lng": 8.540306219564677},
                    {"lat": 47.36695996498498, "lng": 8.542aborrar41954498045},
                    {"lat": 47.36815498498763, "lng": 8.541aborrar22}
                ],
                "created_at": datetime.utcnow(),
            })
    else:
        # Ensure zone exists for existing hotel
        hotel_id = str(hotel["_id"])
        existing_zone = await zones_collection.find_one({"zone_id": "zone_baur_au_lac"})
        if not existing_zone:
            await zones_collection.insert_one({
                "zone_id": "zone_baur_au_lac",
                "hotel_user_id": hotel_id,
                "name": "Baur au Lac",
                "latitude": 47.36735048307041,
                "longitude": 8.540770835142638,
                "radius": None,
                "polygon": [
                    {"lat": 47.367800793449014, "lng": 8.53916359557819},
                    {"lat": 47.366525972979694, "lng": 8.540306219564677},
                    {"lat": 47.36695996498498, "lng": 8.54241954498045},
                    {"lat": 47.36815498498763, "lng": 8.54122}
                ],
                "created_at": datetime.utcnow(),
            })

# ============ AUTH ROUTES ============

@app.post("/api/auth/register")
async def register(user_data: UserRegister):
    existing = await users_collection.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    vehicle_classes = None
    if user_data.role == "driver" and user_data.vehicle_class:
        vehicle_classes = [vc.strip() for vc in user_data.vehicle_class.split(",")]
    
    user_doc = {
        "email": user_data.email,
        "password": hash_password(user_data.password),
        "name": user_data.name,
        "role": user_data.role,
        "vehicle_class": vehicle_classes,
        "driver_photo": user_data.driver_photo,
        "is_online": False,
        "is_blocked": False,
        "created_at": datetime.utcnow(),
    }
    
    result = await users_collection.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id
    
    token = create_token(str(result.inserted_id))
    
    return {
        "access_token": token,
        "user": serialize_user(user_doc)
    }

@app.post("/api/auth/register/admin")
async def register_admin(admin_data: AdminRegister):
    if admin_data.admin_code != ADMIN_CODE:
        raise HTTPException(status_code=403, detail="Invalid admin code")
    
    existing = await users_collection.find_one({"email": admin_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_doc = {
        "email": admin_data.email,
        "password": hash_password(admin_data.password),
        "name": admin_data.name,
        "role": "admin",
        "created_at": datetime.utcnow(),
    }
    
    result = await users_collection.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id
    
    token = create_token(str(result.inserted_id))
    
    return {
        "access_token": token,
        "user": serialize_user(user_doc)
    }

@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    user = await users_collection.find_one({"email": credentials.email})
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(str(user["_id"]))
    
    return {
        "access_token": token,
        "user": serialize_user(user)
    }

@app.post("/api/auth/google/callback")
async def google_callback(data: GoogleCallback):
    # For demo, create or get user based on session
    # In production, verify with Google OAuth
    demo_email = f"google_{data.session_id[:8]}@demo.com"
    
    user = await users_collection.find_one({"email": demo_email})
    if not user:
        user_doc = {
            "email": demo_email,
            "password": hash_password("google_oauth_user"),
            "name": f"Google User",
            "role": data.role,
            "created_at": datetime.utcnow(),
        }
        result = await users_collection.insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
        user = user_doc
    
    token = create_token(str(user["_id"]))
    
    return {
        "access_token": token,
        "user": serialize_user(user)
    }

@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    return serialize_user(user)

# ============ ZONE ROUTES ============

@app.get("/api/zones")
async def get_zones():
    zones = await zones_collection.find().to_list(100)
    return [{
        "zone_id": z["zone_id"],
        "name": z["name"],
        "latitude": z["latitude"],
        "longitude": z["longitude"],
        "radius": z.get("radius"),
        "polygon": z.get("polygon", []),
    } for z in zones]

@app.get("/api/zones/my")
async def get_my_zone(user: dict = Depends(get_current_user)):
    if user["role"] == "hotel":
        zone = await zones_collection.find_one({"hotel_user_id": str(user["_id"])})
    elif user["role"] == "admin":
        zone = await zones_collection.find_one()
    else:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if not zone:
        return None
    
    return {
        "zone_id": zone["zone_id"],
        "name": zone["name"],
        "latitude": zone["latitude"],
        "longitude": zone["longitude"],
        "radius": zone.get("radius"),
        "polygon": zone.get("polygon", []),
    }

@app.post("/api/zones")
async def create_zone(zone_data: ZoneCreate, user: dict = Depends(get_current_user)):
    if user["role"] not in ["admin", "hotel"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    zone_id = f"zone_{zone_data.name.lower().replace(' ', '_')}"
    
    existing = await zones_collection.find_one({"zone_id": zone_id})
    if existing:
        raise HTTPException(status_code=400, detail="Zone already exists")
    
    zone_doc = {
        "zone_id": zone_id,
        "hotel_user_id": str(user["_id"]),
        "name": zone_data.name,
        "latitude": zone_data.latitude,
        "longitude": zone_data.longitude,
        "radius": zone_data.radius,
        "polygon": [],
        "created_at": datetime.utcnow(),
    }
    
    await zones_collection.insert_one(zone_doc)
    
    return {"zone_id": zone_id, "message": "Zone created"}

@app.put("/api/zones/{zone_id}/polygon")
async def update_zone_polygon(zone_id: str, data: PolygonUpdate, user: dict = Depends(get_current_user)):
    if user["role"] not in ["admin", "hotel"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    result = await zones_collection.update_one(
        {"zone_id": zone_id},
        {"$set": {"polygon": data.polygon, "updated_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    return {"message": "Polygon updated"}

# ============ DRIVER ROUTES ============

@app.post("/api/drivers/go-online")
async def go_online(location: LocationUpdate, user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    if user.get("is_blocked"):
        raise HTTPException(status_code=403, detail="You are blocked")
    
    await users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {
            "is_online": True,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "last_location_update": datetime.utcnow()
        }}
    )
    
    return {"message": "You are now online"}

@app.post("/api/drivers/go-offline")
async def go_offline(user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    await users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"is_online": False}}
    )
    
    # Leave any queues
    await queue_collection.delete_many({"driver_id": str(user["_id"])})
    
    return {"message": "You are now offline"}

@app.post("/api/drivers/update-location")
async def update_location(location: LocationUpdate, user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    await users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "last_location_update": datetime.utcnow()
        }}
    )
    
    return {"message": "Location updated"}

@app.get("/api/drivers/all")
async def get_all_drivers(zone_id: Optional[str] = None):
    query = {"role": "driver", "is_online": True}
    drivers = await users_collection.find(query).to_list(100)
    
    result = []
    for d in drivers:
        driver_id = str(d["_id"])
        queue_entry = await queue_collection.find_one({"driver_id": driver_id})
        
        result.append({
            "driver_id": driver_id,
            "driver_name": d["name"],
            "vehicle_class": d.get("vehicle_class", ["E-Class"])[0] if isinstance(d.get("vehicle_class"), list) else d.get("vehicle_class", "E-Class"),
            "photo": d.get("driver_photo"),
            "latitude": d.get("latitude"),
            "longitude": d.get("longitude"),
            "in_queue": queue_entry is not None,
            "queue_position": queue_entry.get("position") if queue_entry else None,
        })
    
    return {"drivers": result}

@app.get("/api/drivers/all/registered")
async def get_all_registered_drivers(user: dict = Depends(get_current_user)):
    if user["role"] not in ["admin", "hotel"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    drivers = await users_collection.find({"role": "driver"}).to_list(500)
    
    return {
        "drivers": [{
            "user_id": str(d["_id"]),
            "email": d["email"],
            "name": d["name"],
            "vehicle_class": d.get("vehicle_class", []),
            "driver_photo": d.get("driver_photo"),
            "is_online": d.get("is_online", False),
            "is_blocked": d.get("is_blocked", False),
            "created_at": d.get("created_at", datetime.utcnow()).isoformat(),
        } for d in drivers]
    }

@app.get("/api/drivers/{driver_id}/block-status")
async def get_block_status(driver_id: str):
    try:
        driver = await users_collection.find_one({"_id": ObjectId(driver_id)})
    except:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    return {
        "is_blocked": driver.get("is_blocked", False),
        "reason": driver.get("block_reason")
    }

@app.post("/api/drivers/{driver_id}/block")
async def block_driver(driver_id: str, data: BlockDriver, user: dict = Depends(get_current_user)):
    if user["role"] not in ["admin", "hotel"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        result = await users_collection.update_one(
            {"_id": ObjectId(driver_id), "role": "driver"},
            {"$set": {"is_blocked": True, "block_reason": data.reason, "blocked_at": datetime.utcnow()}}
        )
    except:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    # Remove from queue
    await queue_collection.delete_many({"driver_id": driver_id})
    
    return {"message": "Driver blocked"}

@app.post("/api/drivers/{driver_id}/unblock")
async def unblock_driver(driver_id: str, user: dict = Depends(get_current_user)):
    if user["role"] not in ["admin", "hotel"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        result = await users_collection.update_one(
            {"_id": ObjectId(driver_id), "role": "driver"},
            {"$set": {"is_blocked": False}, "$unset": {"block_reason": "", "blocked_at": ""}}
        )
    except:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    return {"message": "Driver unblocked"}

@app.get("/api/drivers/blocked")
async def get_blocked_drivers(user: dict = Depends(get_current_user)):
    if user["role"] not in ["admin", "hotel"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    drivers = await users_collection.find({"role": "driver", "is_blocked": True}).to_list(100)
    
    return {
        "blocked_drivers": [{
            "driver_id": str(d["_id"]),
            "name": d["name"],
            "email": d["email"],
            "reason": d.get("block_reason"),
            "blocked_at": d.get("blocked_at", datetime.utcnow()).isoformat(),
        } for d in drivers]
    }

@app.put("/api/drivers/{driver_id}/vehicle-class")
async def update_vehicle_class(driver_id: str, data: VehicleClassUpdate, user: dict = Depends(get_current_user)):
    if user["role"] not in ["admin", "hotel"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        result = await users_collection.update_one(
            {"_id": ObjectId(driver_id), "role": "driver"},
            {"$set": {"vehicle_class": data.vehicle_classes}}
        )
    except:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    return {"message": "Vehicle classes updated"}

# ============ QUEUE ROUTES ============

@app.post("/api/queue/join")
async def join_queue(data: QueueJoin, user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    if user.get("is_blocked"):
        raise HTTPException(status_code=403, detail="You are blocked")
    
    driver_id = str(user["_id"])
    
    # Check if already in queue
    existing = await queue_collection.find_one({"driver_id": driver_id, "zone_id": data.zone_id})
    if existing:
        raise HTTPException(status_code=400, detail="Already in queue")
    
    # Check if in zone
    zone = await zones_collection.find_one({"zone_id": data.zone_id})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    # Check polygon if exists
    if zone.get("polygon") and len(zone["polygon"]) >= 3:
        point = {"lat": data.latitude, "lng": data.longitude}
        if not point_in_polygon(point, zone["polygon"]):
            raise HTTPException(status_code=400, detail="You are not in the pickup zone")
    
    # Get current max position
    last_entry = await queue_collection.find_one(
        {"zone_id": data.zone_id},
        sort=[("position", -1)]
    )
    next_position = (last_entry["position"] + 1) if last_entry else 1
    
    vehicle_class = user.get("vehicle_class", ["E-Class"])
    if isinstance(vehicle_class, list):
        vehicle_class = vehicle_class[0] if vehicle_class else "E-Class"
    
    queue_entry = {
        "zone_id": data.zone_id,
        "driver_id": driver_id,
        "driver_name": user["name"],
        "vehicle_class": vehicle_class,
        "position": next_position,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "entered_at": datetime.utcnow(),
    }
    
    result = await queue_collection.insert_one(queue_entry)
    
    return {
        "queue_id": str(result.inserted_id),
        "position": next_position,
        "message": f"You are now #{next_position} in queue"
    }

@app.post("/api/queue/leave")
async def leave_queue(data: QueueLeave, user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    driver_id = str(user["_id"])
    
    entry = await queue_collection.find_one_and_delete({
        "driver_id": driver_id,
        "zone_id": data.zone_id
    })
    
    if not entry:
        raise HTTPException(status_code=404, detail="Not in queue")
    
    # Reorder positions
    await queue_collection.update_many(
        {"zone_id": data.zone_id, "position": {"$gt": entry["position"]}},
        {"$inc": {"position": -1}}
    )
    
    return {"message": "Left queue"}

@app.get("/api/queue/{zone_id}")
async def get_queue(zone_id: str, user: dict = Depends(get_current_user)):
    entries = await queue_collection.find({"zone_id": zone_id}).sort("position", 1).to_list(100)
    
    return {
        "zone_id": zone_id,
        "queue": [{
            "queue_id": str(e["_id"]),
            "driver_id": e["driver_id"],
            "driver_name": e["driver_name"],
            "vehicle_class": e["vehicle_class"],
            "position": e["position"],
            "entered_at": e["entered_at"].isoformat(),
        } for e in entries]
    }

@app.get("/api/queue/my/status")
async def get_my_queue_status(user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    driver_id = str(user["_id"])
    entry = await queue_collection.find_one({"driver_id": driver_id})
    
    if not entry:
        return {"in_queue": False}
    
    return {
        "in_queue": True,
        "entry": {
            "queue_id": str(entry["_id"]),
            "zone_id": entry["zone_id"],
            "position": entry["position"],
            "entered_at": entry["entered_at"].isoformat(),
        }
    }

# ============ RIDE ROUTES ============

@app.post("/api/rides/request")
async def request_ride(data: RideRequest, user: dict = Depends(get_current_user)):
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Find driver
    if data.driver_id:
        # Specific driver requested
        queue_entry = await queue_collection.find_one({
            "zone_id": data.zone_id,
            "driver_id": data.driver_id
        })
    elif data.vehicle_class:
        # Find first driver of class
        queue_entry = await queue_collection.find_one(
            {"zone_id": data.zone_id, "vehicle_class": data.vehicle_class},
            sort=[("position", 1)]
        )
    else:
        # Find first driver
        queue_entry = await queue_collection.find_one(
            {"zone_id": data.zone_id},
            sort=[("position", 1)]
        )
    
    if not queue_entry:
        raise HTTPException(status_code=404, detail="No drivers available")
    
    # Create ride
    ride_doc = {
        "zone_id": data.zone_id,
        "hotel_id": str(user["_id"]),
        "driver_id": queue_entry["driver_id"],
        "driver_name": queue_entry["driver_name"],
        "vehicle_class": queue_entry["vehicle_class"],
        "customer_info": data.customer_info,
        "status": "requested",
        "requested_at": datetime.utcnow(),
    }
    
    result = await rides_collection.insert_one(ride_doc)
    
    # Remove from queue
    await queue_collection.delete_one({"_id": queue_entry["_id"]})
    
    # Reorder queue
    await queue_collection.update_many(
        {"zone_id": data.zone_id, "position": {"$gt": queue_entry["position"]}},
        {"$inc": {"position": -1}}
    )
    
    return {
        "ride_id": str(result.inserted_id),
        "driver_id": queue_entry["driver_id"],
        "driver_name": queue_entry["driver_name"],
        "vehicle_class": queue_entry["vehicle_class"],
        "status": "requested"
    }

@app.post("/api/rides/{ride_id}/accept")
async def accept_ride(ride_id: str, user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    try:
        result = await rides_collection.update_one(
            {"_id": ObjectId(ride_id), "driver_id": str(user["_id"]), "status": "requested"},
            {"$set": {"status": "accepted", "accepted_at": datetime.utcnow()}}
        )
    except:
        raise HTTPException(status_code=404, detail="Ride not found")
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Ride not found or already processed")
    
    return {"message": "Ride accepted"}

@app.post("/api/rides/{ride_id}/decline")
async def decline_ride(ride_id: str, user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    try:
        ride = await rides_collection.find_one({"_id": ObjectId(ride_id), "driver_id": str(user["_id"])})
    except:
        raise HTTPException(status_code=404, detail="Ride not found")
    
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    
    # Cancel this ride
    await rides_collection.update_one(
        {"_id": ObjectId(ride_id)},
        {"$set": {"status": "declined", "declined_at": datetime.utcnow()}}
    )
    
    return {"message": "Ride declined. It has been passed to the next driver."}

@app.post("/api/rides/{ride_id}/start")
async def start_ride(ride_id: str, data: RideStart, user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    try:
        result = await rides_collection.update_one(
            {"_id": ObjectId(ride_id), "driver_id": str(user["_id"]), "status": "accepted"},
            {"$set": {
                "status": "in_progress",
                "destination": data.destination,
                "started_at": datetime.utcnow()
            }}
        )
    except:
        raise HTTPException(status_code=404, detail="Ride not found")
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Ride not found or not accepted")
    
    return {"message": "Ride started"}

@app.post("/api/rides/{ride_id}/complete")
async def complete_ride(ride_id: str, data: RideComplete, user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    try:
        result = await rides_collection.update_one(
            {"_id": ObjectId(ride_id), "driver_id": str(user["_id"]), "status": "in_progress"},
            {"$set": {
                "status": "completed",
                "price": data.price,
                "commission": data.commission,
                "ride_type": data.ride_type,
                "destination": data.destination,
                "end_latitude": data.latitude,
                "end_longitude": data.longitude,
                "completed_at": datetime.utcnow()
            }}
        )
    except:
        raise HTTPException(status_code=404, detail="Ride not found")
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Ride not found or not in progress")
    
    return {"message": "Ride completed", "commission": data.commission}

@app.get("/api/rides/active/driver")
async def get_active_ride_driver(user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    ride = await rides_collection.find_one({
        "driver_id": str(user["_id"]),
        "status": {"$in": ["requested", "accepted", "in_progress"]}
    })
    
    if not ride:
        return None
    
    return {
        "ride_id": str(ride["_id"]),
        "zone_id": ride["zone_id"],
        "customer_info": ride.get("customer_info"),
        "destination": ride.get("destination"),
        "status": ride["status"]
    }

@app.get("/api/rides/active/hotel")
async def get_active_rides_hotel(user: dict = Depends(get_current_user)):
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    rides = await rides_collection.find({
        "hotel_id": str(user["_id"]),
        "status": {"$in": ["requested", "accepted", "in_progress"]}
    }).to_list(50)
    
    return [{
        "ride_id": str(r["_id"]),
        "driver_id": r["driver_id"],
        "driver_name": r["driver_name"],
        "vehicle_class": r["vehicle_class"],
        "status": r["status"]
    } for r in rides]

@app.get("/api/rides/history/hotel")
async def get_ride_history_hotel(user: dict = Depends(get_current_user)):
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    rides = await rides_collection.find({
        "hotel_id": str(user["_id"]),
        "status": "completed"
    }).sort("completed_at", -1).to_list(100)
    
    return [{
        "ride_id": str(r["_id"]),
        "driver_name": r["driver_name"],
        "vehicle_class": r["vehicle_class"],
        "destination": r.get("destination"),
        "price": r.get("price", 0),
        "commission": r.get("commission", 0),
        "ride_type": r.get("ride_type"),
        "completed_at": r.get("completed_at", datetime.utcnow()).isoformat(),
    } for r in rides]

# ============ STATS ROUTES ============

@app.get("/api/stats/driver/monthly")
async def get_driver_monthly_stats(user: dict = Depends(get_current_user)):
    if user["role"] != "driver":
        raise HTTPException(status_code=403, detail="Not a driver")
    
    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1)
    
    rides = await rides_collection.find({
        "driver_id": str(user["_id"]),
        "status": "completed",
        "completed_at": {"$gte": month_start}
    }).to_list(1000)
    
    total_rides = len(rides)
    total_revenue = sum(r.get("price", 0) for r in rides)
    total_commission = sum(r.get("commission", 0) for r in rides)
    
    by_type = {"airport": {"rides": 0, "commission": 0}, "city": {"rides": 0, "commission": 0}, "other": {"rides": 0, "commission": 0}}
    for r in rides:
        rt = r.get("ride_type", "city")
        if rt in by_type:
            by_type[rt]["rides"] += 1
            by_type[rt]["commission"] += r.get("commission", 0)
    
    return {
        "month": now.strftime("%B %Y"),
        "total_rides": total_rides,
        "total_revenue": total_revenue,
        "total_commission_paid": total_commission,
        "by_type": by_type
    }

@app.get("/api/stats/hotel/monthly")
async def get_hotel_monthly_stats(user: dict = Depends(get_current_user)):
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1)
    
    rides = await rides_collection.find({
        "hotel_id": str(user["_id"]),
        "status": "completed",
        "completed_at": {"$gte": month_start}
    }).to_list(1000)
    
    total_rides = len(rides)
    total_revenue = sum(r.get("price", 0) for r in rides)
    total_commission = sum(r.get("commission", 0) for r in rides)
    
    return {
        "month": now.strftime("%B %Y"),
        "total_rides": total_rides,
        "total_revenue": total_revenue,
        "total_commission_collected": total_commission
    }

# ============ NOTIFICATION ROUTES ============

@app.post("/api/notifications/register")
async def register_notification(data: NotificationRegister):
    await notifications_collection.update_one(
        {"user_id": data.user_id},
        {"$set": {
            "push_token": data.push_token,
            "platform": data.platform,
            "updated_at": datetime.utcnow()
        }},
        upsert=True
    )
    return {"message": "Push token registered"}

# ============ HEALTH CHECK ============

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)