from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, Depends
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Union
import uuid
from datetime import datetime, timezone, timedelta
import httpx
import bcrypt
from jose import jwt, JWTError
import math

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'taxi_queue_db')]

# JWT Config
SECRET_KEY = os.environ.get('SECRET_KEY', 'taxi-queue-secret-key-2025')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

# Vehicle classes
VEHICLE_CLASSES = ["E-Class", "S-Class", "V-Class"]

# Create the main app
app = FastAPI(title="Taxi Queue API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

class UserBase(BaseModel):
    email: EmailStr
    name: str
    role: str
    picture: Optional[str] = None

class DriverRegister(BaseModel):
    email: EmailStr
    name: str
    password: str
    vehicle_class: Optional[str] = None
    vehicle_classes: Optional[List[str]] = None
    photo: str

class HotelRegister(BaseModel):
    email: EmailStr
    name: str
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    user_id: str
    created_at: datetime
    vehicle_class: Optional[str] = None
    driver_photo: Optional[str] = None

class DriverLocation(BaseModel):
    driver_id: str
    driver_name: str
    latitude: float
    longitude: float
    vehicle_class: str
    photo: Optional[str] = None
    in_queue: bool
    queue_position: Optional[int] = None
    last_updated: datetime

class PolygonPoint(BaseModel):
    lat: float
    lng: float

class HotelZoneBase(BaseModel):
    name: str
    latitude: float
    longitude: float
    radius: Optional[float] = None
    polygon: Optional[List[PolygonPoint]] = None

class HotelZoneCreate(HotelZoneBase):
    pass

class HotelZone(HotelZoneBase):
    zone_id: str
    hotel_user_id: str
    created_at: datetime

class DriverQueueEntry(BaseModel):
    queue_id: str
    driver_id: str
    driver_name: str
    vehicle_class: str
    zone_id: str
    position: int
    entered_at: datetime
    status: str
    left_zone_at: Optional[datetime] = None

class DriverLocationUpdate(BaseModel):
    latitude: float
    longitude: float
    timestamp: datetime

class RideBase(BaseModel):
    destination: Optional[str] = None
    destination_lat: Optional[float] = None
    destination_lng: Optional[float] = None

class RideCreate(BaseModel):
    zone_id: str
    customer_info: Optional[str] = None
    driver_id: Optional[str] = None
    vehicle_class: Optional[str] = None

class PushTokenRegister(BaseModel):
    push_token: str

class Ride(BaseModel):
    ride_id: str
    driver_id: str
    driver_name: str
    vehicle_class: Union[str, List[str]]
    zone_id: str
    hotel_user_id: str
    customer_info: Optional[str] = None
    destination: Optional[str] = None
    destination_lat: Optional[float] = None
    destination_lng: Optional[float] = None
    price: Optional[float] = None
    status: str
    driver_location: Optional[DriverLocationUpdate] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class BlockedDriver(BaseModel):
    driver_id: str
    driver_name: str
    driver_email: str
    vehicle_class: Optional[str] = None
    blocked_by: str
    blocked_at: datetime
    reason: Optional[str] = None

class BlockDriverRequest(BaseModel):
    driver_id: str
    reason: Optional[str] = None

# ==================== AUTH HELPERS ====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode = {"sub": user_id, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def point_in_polygon(lat: float, lng: float, polygon: List[dict]) -> bool:
    if not polygon or len(polygon) < 3:
        return False
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]['lat'], polygon[i]['lng']
        xj, yj = polygon[j]['lat'], polygon[j]['lng']
        if ((yi > lng) != (yj > lng)) and (lat < (xj - xi) * (lng - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def is_driver_in_zone(lat: float, lng: float, zone: dict) -> bool:
    if zone.get('polygon') and len(zone.get('polygon', [])) >= 3:
        return point_in_polygon(lat, lng, zone['polygon'])
    if zone.get('radius'):
        from math import radians, cos, sin, sqrt, atan2
        R = 6371000
        lat1, lon1 = radians(zone['latitude']), radians(zone['longitude'])
        lat2, lon2 = radians(lat), radians(lng)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        return distance <= zone['radius']
    return False

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def calculate_eta_minutes(distance_meters: float, avg_speed_kmh: float = 30) -> int:
    if distance_meters <= 0:
        return 0
    distance_km = distance_meters / 1000
    time_hours = distance_km / avg_speed_kmh
    time_minutes = int(time_hours * 60)
    return max(1, time_minutes)

async def find_nearest_online_driver(zone: dict, exclude_driver_ids: list = None, vehicle_class: str = None) -> dict:
    if exclude_driver_ids is None:
        exclude_driver_ids = []
    query = {
        "role": "driver",
        "is_online": True,
        "user_id": {"$nin": exclude_driver_ids},
        "current_location": {"$ne": None},
        "is_blocked": {"$ne": True}
    }
    if vehicle_class:
        query["$or"] = [{"vehicle_class": vehicle_class}, {"vehicle_classes": vehicle_class}]
    drivers = await db.users.find(query, {"_id": 0, "password_hash": 0}).to_list(100)
    if not drivers:
        return None
    queue_driver_ids = set()
    queue_entries = await db.driver_queue.find({"zone_id": zone["zone_id"], "status": "waiting"}, {"driver_id": 1}).to_list(100)
    for entry in queue_entries:
        queue_driver_ids.add(entry["driver_id"])
    hotel_lat = zone["latitude"]
    hotel_lng = zone["longitude"]
    drivers_with_distance = []
    for driver in drivers:
        if driver["user_id"] in queue_driver_ids:
            continue
        loc = driver.get("current_location", {})
        if not loc or not loc.get("latitude") or not loc.get("longitude"):
            continue
        distance = calculate_distance(hotel_lat, hotel_lng, loc["latitude"], loc["longitude"])
        eta = calculate_eta_minutes(distance)
        drivers_with_distance.append({**driver, "distance_meters": distance, "eta_minutes": eta})
    if not drivers_with_distance:
        return None
    drivers_with_distance.sort(key=lambda d: d["distance_meters"])
    return drivers_with_distance[0]

async def get_current_user(request: Request) -> dict:
    token = None
    token = request.cookies.get("session_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    session = await db.user_sessions.find_one({"session_token": token}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    expires_at = session.get("expires_at")
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    user = await db.users.find_one({"user_id": session["user_id"]}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def recalculate_positions(zone_id: str):
    entries = await db.driver_queue.find({"zone_id": zone_id, "status": "waiting"}, {"_id": 0}).sort("entered_at", 1).to_list(100)
    for i, entry in enumerate(entries):
        await db.driver_queue.update_one({"queue_id": entry["queue_id"]}, {"$set": {"position": i + 1}})

async def send_push_notification(push_token: str, title: str, body: str, data: dict = None):
    try:
        message = {"to": push_token, "sound": "default", "title": title, "body": body, "data": data or {}, "priority": "high", "channelId": "ride-requests"}
        async with httpx.AsyncClient() as client:
            response = await client.post("https://exp.host/--/api/v2/push/send", json=message, headers={"Content-Type": "application/json"})
            logger.info(f"Push notification sent: {response.status_code}")
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Error sending push notification: {e}")
        return False
# ==================== AUTH ENDPOINTS ====================

@api_router.post("/auth/register/driver")
async def register_driver(user_data: DriverRegister, response: Response):
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    vehicle_classes = user_data.vehicle_classes or ([user_data.vehicle_class] if user_data.vehicle_class else [])
    for vc in vehicle_classes:
        if vc not in VEHICLE_CLASSES:
            raise HTTPException(status_code=400, detail=f"Vehicle class '{vc}' must be one of: {VEHICLE_CLASSES}")
    user_id = f"driver_{uuid.uuid4().hex[:12]}"
    hashed_password = hash_password(user_data.password)
    user_doc = {"user_id": user_id, "email": user_data.email, "name": user_data.name, "role": "driver", "password_hash": hashed_password, "vehicle_class": vehicle_classes[0] if vehicle_classes else None, "vehicle_classes": vehicle_classes, "driver_photo": user_data.photo, "picture": None, "created_at": datetime.now(timezone.utc), "is_online": False, "current_location": None}
    await db.users.insert_one(user_doc)
    session_token = f"session_{uuid.uuid4().hex}"
    session_doc = {"user_id": user_id, "session_token": session_token, "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS), "created_at": datetime.now(timezone.utc)}
    await db.user_sessions.insert_one(session_doc)
    response.set_cookie(key="session_token", value=session_token, httponly=True, secure=True, samesite="none", path="/", max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60)
    return {"user_id": user_id, "email": user_data.email, "name": user_data.name, "role": "driver", "vehicle_class": user_data.vehicle_class, "session_token": session_token}

@api_router.post("/auth/register/hotel")
async def register_hotel(user_data: HotelRegister, response: Response):
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = f"hotel_{uuid.uuid4().hex[:12]}"
    hashed_password = hash_password(user_data.password)
    user_doc = {"user_id": user_id, "email": user_data.email, "name": user_data.name, "role": "hotel", "password_hash": hashed_password, "picture": None, "created_at": datetime.now(timezone.utc)}
    await db.users.insert_one(user_doc)
    session_token = f"session_{uuid.uuid4().hex}"
    session_doc = {"user_id": user_id, "session_token": session_token, "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS), "created_at": datetime.now(timezone.utc)}
    await db.user_sessions.insert_one(session_doc)
    response.set_cookie(key="session_token", value=session_token, httponly=True, secure=True, samesite="none", path="/", max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60)
    return {"user_id": user_id, "email": user_data.email, "name": user_data.name, "role": "hotel", "session_token": session_token}

@api_router.post("/auth/login")
async def login(credentials: UserLogin, response: Response):
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Please use Google login for this account")
    if not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    session_token = f"session_{uuid.uuid4().hex}"
    session_doc = {"user_id": user["user_id"], "session_token": session_token, "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS), "created_at": datetime.now(timezone.utc)}
    await db.user_sessions.insert_one(session_doc)
    response.set_cookie(key="session_token", value=session_token, httponly=True, secure=True, samesite="none", path="/", max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60)
    return {"user_id": user["user_id"], "email": user["email"], "name": user["name"], "role": user["role"], "vehicle_class": user.get("vehicle_class"), "driver_photo": user.get("driver_photo"), "session_token": session_token}

@api_router.get("/auth/me")
async def get_me(request: Request):
    user = await get_current_user(request)
    return {"user_id": user["user_id"], "email": user["email"], "name": user["name"], "role": user["role"], "vehicle_class": user.get("vehicle_class"), "driver_photo": user.get("driver_photo"), "picture": user.get("picture")}

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("session_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    if token:
        session = await db.user_sessions.find_one({"session_token": token})
        if session:
            await db.users.update_one({"user_id": session["user_id"]}, {"$set": {"is_online": False}})
        await db.user_sessions.delete_one({"session_token": token})
    response.delete_cookie(key="session_token", path="/")
    return {"message": "Logged out"}

# ==================== ZONE ENDPOINTS ====================

@api_router.get("/zones")
async def get_zones(request: Request):
    user = await get_current_user(request)
    if user["role"] == "hotel":
        zones = await db.hotel_zones.find({"hotel_user_id": user["user_id"]}, {"_id": 0}).to_list(10)
    else:
        zones = await db.hotel_zones.find({}, {"_id": 0}).to_list(50)
    return zones

@api_router.get("/zones/{zone_id}")
async def get_zone(zone_id: str, request: Request):
    await get_current_user(request)
    zone = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return zone

@api_router.post("/zones")
async def create_zone(zone_data: HotelZoneCreate, request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can create zones")
    zone_id = f"zone_{uuid.uuid4().hex[:12]}"
    zone_doc = {"zone_id": zone_id, "hotel_user_id": user["user_id"], "name": zone_data.name, "latitude": zone_data.latitude, "longitude": zone_data.longitude, "radius": zone_data.radius, "polygon": [p.dict() for p in zone_data.polygon] if zone_data.polygon else None, "created_at": datetime.now(timezone.utc)}
    await db.hotel_zones.insert_one(zone_doc)
    return {"zone_id": zone_id, "message": "Zone created"}

@api_router.put("/zones/{zone_id}")
async def update_zone(zone_id: str, zone_data: HotelZoneCreate, request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can update zones")
    zone = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    update_data = {"name": zone_data.name, "latitude": zone_data.latitude, "longitude": zone_data.longitude, "radius": zone_data.radius, "polygon": [p.dict() for p in zone_data.polygon] if zone_data.polygon else None}
    await db.hotel_zones.update_one({"zone_id": zone_id}, {"$set": update_data})
    return {"message": "Zone updated"}

# ==================== DRIVER ENDPOINTS ====================

@api_router.post("/drivers/go-online")
async def driver_go_online(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can go online")
    body = await request.json()
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    await db.users.update_one({"user_id": user["user_id"]}, {"$set": {"is_online": True, "current_location": {"latitude": latitude, "longitude": longitude, "updated_at": datetime.now(timezone.utc)}}})
    return {"message": "You are now online", "is_online": True}

@api_router.post("/drivers/go-offline")
async def driver_go_offline(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can go offline")
    await db.users.update_one({"user_id": user["user_id"]}, {"$set": {"is_online": False}})
    return {"message": "You are now offline", "is_online": False}

@api_router.post("/drivers/update-location")
async def update_driver_location(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can update location")
    body = await request.json()
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    await db.users.update_one({"user_id": user["user_id"]}, {"$set": {"is_online": True, "current_location": {"latitude": latitude, "longitude": longitude, "updated_at": datetime.now(timezone.utc)}}})
    return {"message": "Location updated"}

@api_router.get("/drivers/all")
async def get_all_drivers(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can view all drivers")
    drivers = await db.users.find({"role": "driver", "is_online": True}, {"_id": 0, "password_hash": 0}).to_list(100)
    result = []
    for driver in drivers:
        queue_entry = await db.driver_queue.find_one({"driver_id": driver["user_id"], "status": "waiting"}, {"_id": 0})
        location = driver.get("current_location", {})
        result.append({"driver_id": driver["user_id"], "driver_name": driver["name"], "vehicle_class": driver.get("vehicle_class", "Unknown"), "photo": driver.get("driver_photo"), "latitude": location.get("latitude"), "longitude": location.get("longitude"), "last_updated": location.get("updated_at"), "in_queue": queue_entry is not None, "queue_position": queue_entry["position"] if queue_entry else None, "queue_zone_id": queue_entry["zone_id"] if queue_entry else None})
    return {"drivers": result, "count": len(result)}

@api_router.get("/drivers/all-with-distance/{zone_id}")
async def get_all_drivers_with_distance(zone_id: str, request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can view all drivers")
    zone = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    hotel_lat = zone["latitude"]
    hotel_lng = zone["longitude"]
    drivers = await db.users.find({"role": "driver", "is_online": True}, {"_id": 0, "password_hash": 0}).to_list(100)
    queue_entries = await db.driver_queue.find({"zone_id": zone_id, "status": "waiting"}, {"_id": 0}).to_list(100)
    queue_by_driver = {q["driver_id"]: q for q in queue_entries}
    in_zone_drivers = []
    outside_zone_drivers = []
    for driver in drivers:
        location = driver.get("current_location", {})
        lat = location.get("latitude")
        lng = location.get("longitude")
        distance_meters = None
        eta_minutes = None
        if lat and lng:
            distance_meters = calculate_distance(hotel_lat, hotel_lng, lat, lng)
            eta_minutes = calculate_eta_minutes(distance_meters)
        queue_entry = queue_by_driver.get(driver["user_id"])
        in_queue = queue_entry is not None
        in_zone = False
        if lat and lng:
            in_zone = is_driver_in_zone(lat, lng, zone)
        driver_info = {"driver_id": driver["user_id"], "driver_name": driver["name"], "vehicle_class": driver.get("vehicle_class", "Unknown"), "photo": driver.get("driver_photo"), "latitude": lat, "longitude": lng, "last_updated": location.get("updated_at"), "in_queue": in_queue, "queue_position": queue_entry["position"] if queue_entry else None, "in_zone": in_zone, "distance_meters": round(distance_meters) if distance_meters else None, "distance_km": round(distance_meters / 1000, 1) if distance_meters else None, "eta_minutes": eta_minutes}
        if in_zone or in_queue:
            in_zone_drivers.append(driver_info)
        else:
            outside_zone_drivers.append(driver_info)
    in_zone_drivers.sort(key=lambda d: d.get("queue_position") or 999)
    outside_zone_drivers.sort(key=lambda d: d.get("distance_meters") or 999999)
    return {"zone_id": zone_id, "zone_name": zone.get("name", "Unknown Zone"), "in_zone_drivers": in_zone_drivers, "outside_zone_drivers": outside_zone_drivers, "total_in_zone": len(in_zone_drivers), "total_outside_zone": len(outside_zone_drivers), "total_online": len(in_zone_drivers) + len(outside_zone_drivers)}

@api_router.get("/drivers/{driver_id}/block-status")
async def get_driver_block_status(driver_id: str, request: Request):
    user = await get_current_user(request)
    if user["role"] == "driver" and user["user_id"] != driver_id:
        raise HTTPException(status_code=403, detail="Cannot check other driver's status")
    block_record = await db.blocked_drivers.find_one({"driver_id": driver_id}, {"_id": 0})
    is_blocked = block_record is not None
    return {"driver_id": driver_id, "is_blocked": is_blocked, "blocked_at": block_record["blocked_at"] if block_record else None, "reason": block_record.get("reason") if block_record else None}

@api_router.post("/drivers/{driver_id}/block")
async def block_driver(driver_id: str, request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel or admin can block drivers")
    driver = await db.users.find_one({"user_id": driver_id, "role": "driver"}, {"_id": 0})
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    existing = await db.blocked_drivers.find_one({"driver_id": driver_id})
    if existing:
        raise HTTPException(status_code=400, detail="Driver is already blocked")
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    reason = body.get("reason", "")
    block_doc = {"driver_id": driver_id, "driver_name": driver["name"], "driver_email": driver["email"], "vehicle_class": driver.get("vehicle_class"), "blocked_by": user["user_id"], "blocked_by_name": user["name"], "blocked_at": datetime.now(timezone.utc), "reason": reason}
    await db.blocked_drivers.insert_one(block_doc)
    await db.driver_queue.delete_many({"driver_id": driver_id})
    await db.users.update_one({"user_id": driver_id}, {"$set": {"is_online": False, "is_blocked": True}})
    return {"message": f"Driver {driver['name']} has been blocked", "driver_id": driver_id}

@api_router.post("/drivers/{driver_id}/unblock")
async def unblock_driver(driver_id: str, request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel or admin can unblock drivers")
    block_record = await db.blocked_drivers.find_one({"driver_id": driver_id})
    if not block_record:
        raise HTTPException(status_code=400, detail="Driver is not blocked")
    await db.blocked_drivers.delete_one({"driver_id": driver_id})
    await db.users.update_one({"user_id": driver_id}, {"$set": {"is_blocked": False}})
    driver = await db.users.find_one({"user_id": driver_id}, {"_id": 0, "name": 1})
    return {"message": f"Driver {driver['name']} has been unblocked", "driver_id": driver_id}

# ==================== QUEUE ENDPOINTS ====================

@api_router.get("/queue/{zone_id}")
async def get_queue(zone_id: str, request: Request):
    await get_current_user(request)
    entries = await db.driver_queue.find({"zone_id": zone_id, "status": "waiting"}, {"_id": 0}).sort("position", 1).to_list(50)
    return {"zone_id": zone_id, "queue": entries, "count": len(entries)}

@api_router.post("/queue/join")
async def join_queue(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can join queue")
    body = await request.json()
    zone_id = body.get("zone_id")
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    zone = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    existing = await db.driver_queue.find_one({"driver_id": user["user_id"], "zone_id": zone_id, "status": "waiting"})
    if existing:
        raise HTTPException(status_code=400, detail="Already in queue")
    max_pos = await db.driver_queue.find_one({"zone_id": zone_id, "status": "waiting"}, sort=[("position", -1)])
    position = (max_pos["position"] + 1) if max_pos else 1
    queue_id = f"queue_{uuid.uuid4().hex[:12]}"
    queue_doc = {"queue_id": queue_id, "driver_id": user["user_id"], "driver_name": user["name"], "vehicle_class": user.get("vehicle_class", "Unknown"), "photo": user.get("driver_photo"), "zone_id": zone_id, "position": position, "entered_at": datetime.now(timezone.utc), "status": "waiting", "left_zone_at": None}
    await db.driver_queue.insert_one(queue_doc)
    await db.users.update_one({"user_id": user["user_id"]}, {"$set": {"is_online": True, "current_location": {"latitude": latitude, "longitude": longitude, "updated_at": datetime.now(timezone.utc)}}})
    return {"queue_id": queue_id, "position": position, "message": "Joined queue"}

@api_router.post("/queue/leave")
async def leave_queue(request: Request):
    user = await get_current_user(request)
    body = await request.json()
    zone_id = body.get("zone_id")
    result = await db.driver_queue.delete_one({"driver_id": user["user_id"], "zone_id": zone_id, "status": "waiting"})
    if result.deleted_count == 0:
        raise HTTPException(status_code=400, detail="Not in queue")
    await recalculate_positions(zone_id)
    return {"message": "Left queue"}

@api_router.get("/queue/my/status")
async def get_my_queue_status(request: Request):
    user = await get_current_user(request)
    entry = await db.driver_queue.find_one({"driver_id": user["user_id"], "status": "waiting"}, {"_id": 0})
    if entry:
        return {"in_queue": True, "entry": entry}
    return {"in_queue": False, "entry": None}

@api_router.post("/queue/update-location")
async def update_queue_location(request: Request):
    user = await get_current_user(request)
    body = await request.json()
    zone_id = body.get("zone_id")
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    zone = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    entry = await db.driver_queue.find_one({"driver_id": user["user_id"], "zone_id": zone_id, "status": "waiting"}, {"_id": 0})
    if not entry:
        return {"in_queue": False, "message": "Not in queue"}
    in_zone = is_driver_in_zone(latitude, longitude, zone)
    await db.users.update_one({"user_id": user["user_id"]}, {"$set": {"current_location": {"latitude": latitude, "longitude": longitude, "updated_at": datetime.now(timezone.utc)}}})
    if in_zone:
        await db.driver_queue.update_one({"queue_id": entry["queue_id"]}, {"$set": {"left_zone_at": None}})
        return {"in_queue": True, "in_zone": True, "position": entry["position"]}
    else:
        left_at = entry.get("left_zone_at")
        if not left_at:
            await db.driver_queue.update_one({"queue_id": entry["queue_id"]}, {"$set": {"left_zone_at": datetime.now(timezone.utc)}})
            left_at = datetime.now(timezone.utc)
        if isinstance(left_at, str):
            left_at = datetime.fromisoformat(left_at)
        if left_at.tzinfo is None:
            left_at = left_at.replace(tzinfo=timezone.utc)
        grace_period = 180
        elapsed = (datetime.now(timezone.utc) - left_at).total_seconds()
        remaining = grace_period - elapsed
        if remaining <= 0:
            await db.driver_queue.delete_one({"queue_id": entry["queue_id"]})
            await recalculate_positions(zone_id)
            return {"in_queue": False, "in_zone": False, "message": "Removed from queue - left zone too long"}
        return {"in_queue": True, "in_zone": False, "position": entry["position"], "seconds_remaining": int(remaining)}

# ==================== RIDE ENDPOINTS ====================
@api_router.post("/rides/request")
async def request_ride(ride_data: RideCreate, request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can request rides")
    zone = await db.hotel_zones.find_one({"zone_id": ride_data.zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    driver = None
    queue_entry = None
    is_cascade_request = False
    if ride_data.driver_id:
        driver = await db.users.find_one({"user_id": ride_data.driver_id, "role": "driver"}, {"_id": 0, "password_hash": 0})
        if not driver:
            raise HTTPException(status_code=404, detail="Driver not found")
        queue_entry = await db.driver_queue.find_one({"driver_id": ride_data.driver_id, "zone_id": ride_data.zone_id, "status": "waiting"}, {"_id": 0})
    else:
        query = {"zone_id": ride_data.zone_id, "status": "waiting"}
        if ride_data.vehicle_class:
            if ride_data.vehicle_class not in VEHICLE_CLASSES:
                raise HTTPException(status_code=400, detail=f"Invalid vehicle class. Must be one of: {VEHICLE_CLASSES}")
            query["vehicle_class"] = ride_data.vehicle_class
        queue_entry = await db.driver_queue.find_one(query, {"_id": 0}, sort=[("position", 1)])
        if queue_entry:
            driver = await db.users.find_one({"user_id": queue_entry["driver_id"]}, {"_id": 0, "password_hash": 0})
        else:
            nearest_driver = await find_nearest_online_driver(zone=zone, exclude_driver_ids=[], vehicle_class=ride_data.vehicle_class)
            if nearest_driver:
                driver = nearest_driver
                is_cascade_request = True
            else:
                vehicle_msg = f" with {ride_data.vehicle_class}" if ride_data.vehicle_class else ""
                raise HTTPException(status_code=404, detail=f"No drivers{vehicle_msg} available - neither in queue nor online nearby")
    ride_id = f"ride_{uuid.uuid4().hex[:12]}"
    ride_doc = {"ride_id": ride_id, "driver_id": driver["user_id"], "driver_name": driver["name"], "vehicle_class": driver.get("vehicle_class", "Unknown"), "zone_id": ride_data.zone_id, "hotel_user_id": user["user_id"], "customer_info": ride_data.customer_info, "requested_vehicle_class": ride_data.vehicle_class, "destination": None, "price": None, "status": "requested", "driver_location": None, "created_at": datetime.now(timezone.utc), "completed_at": None, "is_cascade_request": is_cascade_request, "driver_distance_meters": driver.get("distance_meters"), "driver_eta_minutes": driver.get("eta_minutes"), "declined_by": []}
    await db.rides.insert_one(ride_doc)
    if queue_entry:
        await db.driver_queue.update_one({"queue_id": queue_entry["queue_id"]}, {"$set": {"status": "assigned"}})
        await recalculate_positions(ride_data.zone_id)
    if driver.get("push_token"):
        await send_push_notification(driver["push_token"], "Neue Fahrtanfrage!", "Sie haben eine neue Fahrtanfrage. Tippen Sie zum Annehmen oder Ablehnen.", {"ride_id": ride_id, "type": "ride_request"})
    return Ride(**ride_doc)

@api_router.get("/rides/active/driver")
async def get_driver_active_ride(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can access this")
    ride = await db.rides.find_one({"driver_id": user["user_id"], "status": {"$in": ["requested", "accepted", "in_progress"]}}, {"_id": 0})
    if not ride:
        return None
    return Ride(**ride)

@api_router.get("/rides/active/hotel")
async def get_hotel_active_rides(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can access this")
    rides = await db.rides.find({"hotel_user_id": user["user_id"], "status": {"$in": ["requested", "accepted", "in_progress"]}}, {"_id": 0}).to_list(10)
    return [Ride(**r) for r in rides]

@api_router.post("/rides/{ride_id}/accept")
async def accept_ride(ride_id: str, request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can accept rides")
    ride = await db.rides.find_one({"ride_id": ride_id}, {"_id": 0})
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    if ride["driver_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your ride")
    if ride["status"] != "requested":
        raise HTTPException(status_code=400, detail="Ride already accepted or completed")
    await db.rides.update_one({"ride_id": ride_id}, {"$set": {"status": "accepted"}})
    return {"message": "Ride accepted"}

@api_router.post("/rides/{ride_id}/decline")
async def decline_ride(ride_id: str, request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can decline rides")
    ride = await db.rides.find_one({"ride_id": ride_id}, {"_id": 0})
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    if ride["driver_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your ride")
    if ride["status"] != "requested":
        raise HTTPException(status_code=400, detail="Can only decline requested rides")
    declined_by = ride.get("declined_by", [])
    if user["user_id"] not in declined_by:
        declined_by.append(user["user_id"])
    zone = await db.hotel_zones.find_one({"zone_id": ride["zone_id"]}, {"_id": 0})
    query = {"zone_id": ride["zone_id"], "status": "waiting", "driver_id": {"$ne": user["user_id"]}}
    if ride.get("requested_vehicle_class"):
        query["vehicle_class"] = ride["requested_vehicle_class"]
    next_queue_entry = await db.driver_queue.find_one(query, {"_id": 0}, sort=[("position", 1)])
    next_driver = None
    is_cascade = False
    if next_queue_entry:
        next_driver = await db.users.find_one({"user_id": next_queue_entry["driver_id"]}, {"_id": 0, "password_hash": 0})
        await db.driver_queue.update_one({"queue_id": next_queue_entry["queue_id"]}, {"$set": {"status": "assigned"}})
    else:
        if zone:
            nearest_driver = await find_nearest_online_driver(zone=zone, exclude_driver_ids=declined_by, vehicle_class=ride.get("requested_vehicle_class"))
            if nearest_driver:
                next_driver = nearest_driver
                is_cascade = True
    if next_driver:
        update_data = {"driver_id": next_driver["user_id"], "driver_name": next_driver["name"], "vehicle_class": next_driver.get("vehicle_class", "Unknown"), "declined_by": declined_by, "is_cascade_request": is_cascade}
        if is_cascade:
            update_data["driver_distance_meters"] = next_driver.get("distance_meters")
            update_data["driver_eta_minutes"] = next_driver.get("eta_minutes")
        await db.rides.update_one({"ride_id": ride_id}, {"$set": update_data})
        current_queue = await db.driver_queue.find_one({"driver_id": user["user_id"], "zone_id": ride["zone_id"]}, {"_id": 0})
        if current_queue:
            max_pos_entry = await db.driver_queue.find_one({"zone_id": ride["zone_id"]}, {"_id": 0}, sort=[("position", -1)])
            new_pos = (max_pos_entry["position"] + 1) if max_pos_entry else 1
            await db.driver_queue.update_one({"queue_id": current_queue["queue_id"]}, {"$set": {"status": "waiting", "position": new_pos}})
        await recalculate_positions(ride["zone_id"])
        if next_driver.get("push_token"):
            await send_push_notification(next_driver["push_token"], "Neue Fahrtanfrage!", "Sie haben eine neue Fahrtanfrage. Tippen Sie zum Annehmen oder Ablehnen.", {"ride_id": ride_id, "type": "ride_request"})
        return {"message": f"Ride declined, passed to {'nearby' if is_cascade else 'next'} driver", "next_driver": next_driver["name"], "is_cascade": is_cascade}
    else:
        await db.rides.update_one({"ride_id": ride_id}, {"$set": {"status": "cancelled", "declined_by": declined_by}})
        current_queue = await db.driver_queue.find_one({"driver_id": user["user_id"], "zone_id": ride["zone_id"]}, {"_id": 0})
        if current_queue:
            await db.driver_queue.update_one({"queue_id": current_queue["queue_id"]}, {"$set": {"status": "waiting"}})
            await recalculate_positions(ride["zone_id"])
        return {"message": "Ride declined, no more drivers available - ride cancelled", "is_cascade": False}

@api_router.post("/rides/{ride_id}/start")
async def start_ride(ride_id: str, request: Request):
    user = await get_current_user(request)
    body = await request.json()
    ride = await db.rides.find_one({"ride_id": ride_id}, {"_id": 0})
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    if ride["driver_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your ride")
    await db.rides.update_one({"ride_id": ride_id}, {"$set": {"status": "in_progress", "destination": body.get("destination")}})
    return {"message": "Ride started"}

@api_router.post("/rides/{ride_id}/complete")
async def complete_ride(ride_id: str, request: Request):
    user = await get_current_user(request)
    body = await request.json()
    ride = await db.rides.find_one({"ride_id": ride_id}, {"_id": 0})
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    if ride["driver_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your ride")
    price = body.get("price")
    commission = body.get("commission", 0)
    ride_type = body.get("ride_type", "other")
    final_destination = body.get("destination", ride.get("destination"))
    update_data = {"status": "completed", "completed_at": datetime.now(timezone.utc), "ride_type": ride_type, "commission": float(commission) if commission else 0, "commission_paid": False}
    if price is not None:
        update_data["price"] = float(price)
    if final_destination:
        update_data["destination"] = final_destination
    await db.rides.update_one({"ride_id": ride_id}, {"$set": update_data})
    return {"message": "Ride completed", "price": price, "commission": commission, "ride_type": ride_type}

@api_router.get("/rides/history/driver")
async def get_driver_ride_history(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can access this")
    rides = await db.rides.find({"driver_id": user["user_id"], "status": "completed"}, {"_id": 0}).sort("completed_at", -1).to_list(50)
    return [Ride(**r) for r in rides]

@api_router.get("/rides/history/hotel")
async def get_hotel_ride_history(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can access this")
    rides = await db.rides.find({"hotel_user_id": user["user_id"], "status": "completed"}, {"_id": 0}).sort("completed_at", -1).to_list(50)
    return [Ride(**r) for r in rides]

# ==================== STATISTICS ====================

@api_router.get("/stats/driver/monthly")
async def get_driver_monthly_stats(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can access this")
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    rides = await db.rides.find({"driver_id": user["user_id"], "status": "completed", "completed_at": {"$gte": month_start}}, {"_id": 0}).to_list(500)
    total_rides = len(rides)
    total_commission = sum(r.get("commission", 0) or 0 for r in rides)
    total_revenue = sum(r.get("price", 0) or 0 for r in rides)
    airport_rides = [r for r in rides if r.get("ride_type") == "airport"]
    city_rides = [r for r in rides if r.get("ride_type") == "city"]
    other_rides = [r for r in rides if r.get("ride_type") not in ["airport", "city"]]
    return {"month": now.strftime("%B %Y"), "total_rides": total_rides, "total_commission_paid": total_commission, "total_revenue": total_revenue, "by_type": {"airport": {"rides": len(airport_rides), "commission": sum(r.get("commission", 0) or 0 for r in airport_rides)}, "city": {"rides": len(city_rides), "commission": sum(r.get("commission", 0) or 0 for r in city_rides)}, "other": {"rides": len(other_rides), "commission": sum(r.get("commission", 0) or 0 for r in other_rides)}}}

@api_router.get("/stats/hotel/monthly")
async def get_hotel_monthly_stats(request: Request):
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can access this")
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    rides = await db.rides.find({"hotel_user_id": user["user_id"], "status": "completed", "completed_at": {"$gte": month_start}}, {"_id": 0}).to_list(500)
    total_rides = len(rides)
    total_commission = sum(r.get("commission", 0) or 0 for r in rides)
    total_revenue = sum(r.get("price", 0) or 0 for r in rides)
    return {"month": now.strftime("%B %Y"), "total_rides": total_rides, "total_commission_received": total_commission, "total_revenue": total_revenue}

# ==================== PUSH NOTIFICATIONS ====================

@api_router.post("/notifications/register-token")
async def register_push_token(token_data: PushTokenRegister, request: Request):
    user = await get_current_user(request)
    await db.users.update_one({"user_id": user["user_id"]}, {"$set": {"push_token": token_data.push_token}})
    return {"message": "Push token registered"}

# ==================== GENERAL ====================

@api_router.get("/")
async def root():
    return {"message": "Taxi Queue API is running"}

@api_router.get("/health")
async def health():
    return {"status": "healthy"}

@api_router.get("/vehicle-classes")
async def get_vehicle_classes():
    return {"vehicle_classes": VEHICLE_CLASSES}

# ==================== STARTUP ====================

@app.on_event("startup")
async def init_default_zone():
    default_polygon = [{"lat": 47.367800793449014, "lng": 8.53916359557819}, {"lat": 47.366525972979694, "lng": 8.540306219777566}, {"lat": 47.36693151709968, "lng": 8.541814116331826}, {"lat": 47.36814364875325, "lng": 8.541799408882968}]
    avg_lat = sum(p['lat'] for p in default_polygon) / len(default_polygon)
    avg_lng = sum(p['lng'] for p in default_polygon) / len(default_polygon)
    existing = await db.hotel_zones.find_one({"zone_id": "zone_baur_au_lac"})
    if not existing:
        default_zone = {"zone_id": "zone_baur_au_lac", "hotel_user_id": "hotel_bauraulac_001", "name": "Baur au Lac", "latitude": avg_lat, "longitude": avg_lng, "radius": None, "polygon": default_polygon, "created_at": datetime.now(timezone.utc)}
        await db.hotel_zones.insert_one(default_zone)

@app.on_event("startup")
async def init_default_admin():
    admin_email = "admin@bauraulac.ch"
    existing = await db.users.find_one({"email": admin_email})
    if not existing:
        hashed_password = bcrypt.hashpw("Zurich2025".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        admin_doc = {"user_id": "admin_bauraulac_001", "email": admin_email, "name": "Baur au Lac Admin", "role": "admin", "password_hash": hashed_password, "picture": None, "created_at": datetime.now(timezone.utc)}
        await db.users.insert_one(admin_doc)

@app.on_event("startup")
async def init_default_hotel():
    hotel_email = "taxi@bauraulac.ch"
    existing = await db.users.find_one({"email": hotel_email})
    if not existing:
        hashed_password = bcrypt.hashpw("Marguita2025".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        hotel_doc = {"user_id": "hotel_bauraulac_001", "email": hotel_email, "name": "Baur au Lac Hotel", "role": "hotel", "password_hash": hashed_password, "picture": None, "created_at": datetime.now(timezone.utc)}
        await db.users.insert_one(hotel_doc)

app.include_router(api_router)

app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
