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
    role: str  # "driver" or "hotel"
    picture: Optional[str] = None

class DriverRegister(BaseModel):
    email: EmailStr
    name: str
    password: str
    vehicle_class: Optional[str] = None  # Single class (legacy)
    vehicle_classes: Optional[List[str]] = None  # Multiple classes
    photo: str  # Base64 encoded photo

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
    latitude: float  # Center point (for backward compatibility)
    longitude: float
    radius: Optional[float] = None  # Deprecated - use polygon instead
    polygon: Optional[List[PolygonPoint]] = None  # New polygon boundary

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
    status: str  # "waiting", "assigned", "left"
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
    driver_id: Optional[str] = None  # Optional: specify driver directly
    vehicle_class: Optional[str] = None  # Optional: filter by vehicle class (E-Class, S-Class, V-Class)

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
    price: Optional[float] = None  # Price in CHF
    status: str  # "requested", "accepted", "in_progress", "completed"
    driver_location: Optional[DriverLocationUpdate] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class BlockedDriver(BaseModel):
    driver_id: str
    driver_name: str
    driver_email: str
    vehicle_class: Optional[str] = None
    blocked_by: str  # user_id of hotel/admin who blocked
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
    """
    Check if a point is inside a polygon using ray casting algorithm.
    polygon is a list of dicts with 'lat' and 'lng' keys.
    """
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
    """
    Check if driver is in zone - supports both polygon and radius methods.
    """
    # If zone has polygon, use polygon check
    if zone.get('polygon') and len(zone.get('polygon', [])) >= 3:
        return point_in_polygon(lat, lng, zone['polygon'])
    
    # Fallback to radius check for backward compatibility
    if zone.get('radius'):
        from math import radians, cos, sin, sqrt, atan2
        R = 6371000  # Earth radius in meters
        
        lat1, lon1 = radians(zone['latitude']), radians(zone['longitude'])
        lat2, lon2 = radians(lat), radians(lng)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        return distance <= zone['radius']
    
    return False

async def get_current_user(request: Request) -> dict:
    """Get current user from session token (cookie or header)"""
    token = None
    
    # Try cookie first
    token = request.cookies.get("session_token")
    
    # Then try Authorization header
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Check session in database
    session = await db.user_sessions.find_one({"session_token": token}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check expiry
    expires_at = session.get("expires_at")
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Get user
    user = await db.users.find_one({"user_id": session["user_id"]}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

# ==================== AUTH ENDPOINTS ====================

@api_router.post("/auth/register/driver")
async def register_driver(user_data: DriverRegister, response: Response):
    """Register a new driver with photo and vehicle class(es)"""
    # Check if user exists
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Get vehicle classes (support both single and multiple)
    vehicle_classes = user_data.vehicle_classes or ([user_data.vehicle_class] if user_data.vehicle_class else [])
    
    # Validate vehicle classes
    for vc in vehicle_classes:
        if vc not in VEHICLE_CLASSES:
            raise HTTPException(status_code=400, detail=f"Vehicle class '{vc}' must be one of: {VEHICLE_CLASSES}")
    
    # Create user
    user_id = f"driver_{uuid.uuid4().hex[:12]}"
    hashed_password = hash_password(user_data.password)
    
    user_doc = {
        "user_id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "role": "driver",
        "password_hash": hashed_password,
        "vehicle_class": vehicle_classes[0] if vehicle_classes else None,  # Primary class (for backward compat)
        "vehicle_classes": vehicle_classes,  # All classes the driver can operate
        "driver_photo": user_data.photo,  # Base64 photo
        "picture": None,
        "created_at": datetime.now(timezone.utc),
        "is_online": False,
        "current_location": None,
        "is_approved": False  # Driver must be approved by hotel/admin before going online
    }
    
    await db.users.insert_one(user_doc)
    
    # Create session
    session_token = f"session_{uuid.uuid4().hex}"
    session_doc = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
        "created_at": datetime.now(timezone.utc)
    }
    await db.user_sessions.insert_one(session_doc)
    
    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    )
    
    return {
        "user_id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "role": "driver",
        "vehicle_class": user_data.vehicle_class,
        "session_token": session_token
    }

@api_router.post("/auth/register/hotel")
async def register_hotel(user_data: HotelRegister, response: Response):
    """Register a new hotel (master interface)"""
    # Check if user exists
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user_id = f"hotel_{uuid.uuid4().hex[:12]}"
    hashed_password = hash_password(user_data.password)
    
    user_doc = {
        "user_id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "role": "hotel",
        "password_hash": hashed_password,
        "picture": None,
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.users.insert_one(user_doc)
    
    # Create session
    session_token = f"session_{uuid.uuid4().hex}"
    session_doc = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
        "created_at": datetime.now(timezone.utc)
    }
    await db.user_sessions.insert_one(session_doc)
    
    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    )
    
    return {
        "user_id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "role": "hotel",
        "session_token": session_token
    }

@api_router.post("/auth/register")
async def register(request: Request, response: Response):
    """Legacy register endpoint - routes to appropriate registration"""
    body = await request.json()
    role = body.get("role", "driver")
    
    if role == "hotel":
        user_data = HotelRegister(**body)
        return await register_hotel(user_data, response)
    else:
        # For drivers without photo/vehicle, create basic account
        existing = await db.users.find_one({"email": body.get("email")})
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        user_id = f"driver_{uuid.uuid4().hex[:12]}"
        hashed_password = hash_password(body.get("password"))
        
        user_doc = {
            "user_id": user_id,
            "email": body.get("email"),
            "name": body.get("name"),
            "role": "driver",
            "password_hash": hashed_password,
            "vehicle_class": body.get("vehicle_class"),
            "driver_photo": body.get("photo"),
            "picture": None,
            "created_at": datetime.now(timezone.utc),
            "is_online": False,
            "current_location": None
        }
        
        await db.users.insert_one(user_doc)
        
        session_token = f"session_{uuid.uuid4().hex}"
        session_doc = {
            "user_id": user_id,
            "session_token": session_token,
            "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
            "created_at": datetime.now(timezone.utc)
        }
        await db.user_sessions.insert_one(session_doc)
        
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=True,
            samesite="none",
            path="/",
            max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        return {
            "user_id": user_id,
            "email": body.get("email"),
            "name": body.get("name"),
            "role": "driver",
            "vehicle_class": body.get("vehicle_class"),
            "session_token": session_token
        }

@api_router.post("/auth/login")
async def login(credentials: UserLogin, response: Response):
    """Login with email/password"""
    user = await db.users.find_one({"email": credentials.email}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.get("password_hash"):
        raise HTTPException(status_code=401, detail="Please use Google login for this account")
    
    if not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create session
    session_token = f"session_{uuid.uuid4().hex}"
    session_doc = {
        "user_id": user["user_id"],
        "session_token": session_token,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
        "created_at": datetime.now(timezone.utc)
    }
    await db.user_sessions.insert_one(session_doc)
    
    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    )
    
    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "vehicle_class": user.get("vehicle_class"),
        "driver_photo": user.get("driver_photo"),
        "session_token": session_token
    }

@api_router.post("/auth/google-session")
async def google_session(request: Request, response: Response):
    """Exchange Google session_id for app session"""
    body = await request.json()
    session_id = body.get("session_id")
    role = body.get("role", "driver")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    # Call Emergent Auth to get user data
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={"X-Session-ID": session_id}
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid Google session")
            
            google_data = resp.json()
        except Exception as e:
            logger.error(f"Error calling Emergent Auth: {e}")
            raise HTTPException(status_code=500, detail="Auth service error")
    
    # Check if user exists
    existing_user = await db.users.find_one({"email": google_data["email"]}, {"_id": 0})
    
    if existing_user:
        user_id = existing_user["user_id"]
        # Update name and picture if changed
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"name": google_data["name"], "picture": google_data.get("picture")}}
        )
        user_role = existing_user["role"]
        vehicle_class = existing_user.get("vehicle_class")
        driver_photo = existing_user.get("driver_photo")
    else:
        # Create new user
        prefix = "hotel_" if role == "hotel" else "driver_"
        user_id = f"{prefix}{uuid.uuid4().hex[:12]}"
        user_doc = {
            "user_id": user_id,
            "email": google_data["email"],
            "name": google_data["name"],
            "role": role,
            "picture": google_data.get("picture"),
            "created_at": datetime.now(timezone.utc),
            "is_online": False if role == "driver" else None,
            "current_location": None if role == "driver" else None
        }
        await db.users.insert_one(user_doc)
        user_role = role
        vehicle_class = None
        driver_photo = None
    
    # Create session
    session_token = f"session_{uuid.uuid4().hex}"
    session_doc = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
        "created_at": datetime.now(timezone.utc)
    }
    await db.user_sessions.insert_one(session_doc)
    
    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    )
    
    return {
        "user_id": user_id,
        "email": google_data["email"],
        "name": google_data["name"],
        "role": user_role,
        "vehicle_class": vehicle_class,
        "driver_photo": driver_photo,
        "picture": google_data.get("picture"),
        "session_token": session_token
    }

@api_router.get("/auth/me")
async def get_me(request: Request):
    """Get current authenticated user"""
    user = await get_current_user(request)
    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "vehicle_class": user.get("vehicle_class"),
        "driver_photo": user.get("driver_photo"),
        "picture": user.get("picture")
    }

@api_router.post("/auth/update-profile")
async def update_profile(request: Request):
    """Update driver profile (photo, vehicle class)"""
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can update profile")
    
    body = await request.json()
    update_data = {}
    
    if "vehicle_class" in body:
        if body["vehicle_class"] not in VEHICLE_CLASSES:
            raise HTTPException(status_code=400, detail=f"Vehicle class must be one of: {VEHICLE_CLASSES}")
        update_data["vehicle_class"] = body["vehicle_class"]
    
    if "photo" in body:
        update_data["driver_photo"] = body["photo"]
    
    if "name" in body:
        update_data["name"] = body["name"]
    
    if update_data:
        await db.users.update_one(
            {"user_id": user["user_id"]},
            {"$set": update_data}
        )
    
    return {"message": "Profile updated", "updated_fields": list(update_data.keys())}

@api_router.post("/auth/register/admin")
async def register_admin(request: Request, response: Response):
    """Register an admin account (has access to both interfaces)"""
    body = await request.json()
    admin_code = body.get("admin_code")
    
    # Simple admin code verification (you can change this)
    if admin_code != "BAURAULAC2025":
        raise HTTPException(status_code=403, detail="Invalid admin code")
    
    email = body.get("email")
    password = body.get("password")
    name = body.get("name")
    
    if not all([email, password, name]):
        raise HTTPException(status_code=400, detail="Email, password, and name are required")
    
    # Check if user exists
    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create admin user
    user_id = f"admin_{uuid.uuid4().hex[:12]}"
    hashed_password = hash_password(password)
    
    user_doc = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "role": "admin",
        "password_hash": hashed_password,
        "vehicle_class": body.get("vehicle_class"),  # Optional for admin
        "driver_photo": body.get("photo"),  # Optional for admin
        "picture": None,
        "created_at": datetime.now(timezone.utc),
        "is_online": False,
        "current_location": None
    }
    
    await db.users.insert_one(user_doc)
    
    # Create session
    session_token = f"session_{uuid.uuid4().hex}"
    session_doc = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
        "created_at": datetime.now(timezone.utc)
    }
    await db.user_sessions.insert_one(session_doc)
    
    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    )
    
    return {
        "user_id": user_id,
        "email": email,
        "name": name,
        "role": "admin",
        "session_token": session_token
    }

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Logout and clear session"""
    token = request.cookies.get("session_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    if token:
        # Mark driver as offline
        session = await db.user_sessions.find_one({"session_token": token})
        if session:
            await db.users.update_one(
                {"user_id": session["user_id"]},
                {"$set": {"is_online": False}}
            )
        await db.user_sessions.delete_one({"session_token": token})
    
    response.delete_cookie(key="session_token", path="/")
    return {"message": "Logged out"}

# ==================== DRIVER LOCATION TRACKING ====================

@api_router.post("/drivers/go-online")
async def driver_go_online(request: Request):
    """Driver goes online - starts tracking"""
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can go online")
    
    # Check if driver is approved (admins are always approved)
    if user["role"] == "driver" and not user.get("is_approved", False):
        raise HTTPException(
            status_code=403, 
            detail="Warte auf Freigabe durch Hotel. Bitte kontaktiere das Hotel-Personal."
        )
    
    body = await request.json()
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": {
            "is_online": True,
            "current_location": {
                "latitude": latitude,
                "longitude": longitude,
                "updated_at": datetime.now(timezone.utc)
            }
        }}
    )
    
    return {"message": "You are now online", "is_online": True}

@api_router.post("/drivers/go-offline")
async def driver_go_offline(request: Request):
    """Driver goes offline - stops tracking"""
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can go offline")
    
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": {"is_online": False}}
    )
    
    return {"message": "You are now offline", "is_online": False}

@api_router.post("/drivers/update-location")
async def update_driver_location(request: Request):
    """Update driver's current location (called periodically when online)"""
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can update location")
    
    body = await request.json()
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": {
            "is_online": True,
            "current_location": {
                "latitude": latitude,
                "longitude": longitude,
                "updated_at": datetime.now(timezone.utc)
            }
        }}
    )
    
    return {"message": "Location updated"}

@api_router.get("/drivers/all")
async def get_all_drivers(request: Request):
    """Get all online drivers with their locations (for hotel/master interface)"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can view all drivers")
    
    # Get all online drivers
    drivers = await db.users.find(
        {"role": "driver", "is_online": True},
        {"_id": 0, "password_hash": 0}
    ).to_list(100)
    
    # Get queue status for each driver
    result = []
    for driver in drivers:
        driver_id = driver.get("user_id")
        if not driver_id:
            continue
        queue_entry = await db.driver_queue.find_one(
            {"driver_id": driver_id, "status": "waiting"},
            {"_id": 0}
        )
        
        location = driver.get("current_location", {})
        result.append({
            "driver_id": driver["user_id"],
            "driver_name": driver["name"],
            "vehicle_class": driver.get("vehicle_class", "Unknown"),
            "photo": driver.get("driver_photo"),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "last_updated": location.get("updated_at"),
            "in_queue": queue_entry is not None,
            "queue_position": queue_entry["position"] if queue_entry else None,
            "queue_zone_id": queue_entry["zone_id"] if queue_entry else None
        })
    
    return {"drivers": result, "count": len(result)}

@api_router.get("/drivers/by-class/{vehicle_class}")
async def get_drivers_by_class(vehicle_class: str, request: Request):
    """Get all online drivers of a specific vehicle class"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can view drivers")
    
    if vehicle_class not in VEHICLE_CLASSES:
        raise HTTPException(status_code=400, detail=f"Vehicle class must be one of: {VEHICLE_CLASSES}")
    
    drivers = await db.users.find(
        {"role": "driver", "is_online": True, "vehicle_class": vehicle_class},
        {"_id": 0, "password_hash": 0}
    ).to_list(100)
    
    result = []
    for driver in drivers:
        queue_entry = await db.driver_queue.find_one(
            {"driver_id": driver["user_id"], "status": "waiting"},
            {"_id": 0}
        )
        
        location = driver.get("current_location", {})
        result.append({
            "driver_id": driver["user_id"],
            "driver_name": driver["name"],
            "vehicle_class": driver.get("vehicle_class"),
            "photo": driver.get("driver_photo"),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "last_updated": location.get("updated_at"),
            "in_queue": queue_entry is not None,
            "queue_position": queue_entry["position"] if queue_entry else None
        })
    
    return {"drivers": result, "count": len(result), "vehicle_class": vehicle_class}

@api_router.get("/drivers/all-with-distance/{zone_id}")
async def get_all_drivers_with_distance(zone_id: str, request: Request):
    """
    Get ALL drivers (online and in queue) with distance and ETA to hotel zone.
    This endpoint is for the Hotel Dashboard to show all drivers at once.
    Includes:
    - Drivers in queue (in zone)
    - Drivers online but outside zone
    """
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can view all drivers")
    
    # Get zone
    zone = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    hotel_lat = zone["latitude"]
    hotel_lng = zone["longitude"]
    
    # Get all online drivers
    drivers = await db.users.find(
        {"role": "driver", "is_online": True},
        {"_id": 0, "password_hash": 0}
    ).to_list(100)
    
    # Get queue entries for this zone
    queue_entries = await db.driver_queue.find(
        {"zone_id": zone_id, "status": "waiting"},
        {"_id": 0}
    ).to_list(100)
    
    queue_by_driver = {q["driver_id"]: q for q in queue_entries}
    
    # Build result with distance and ETA
    in_zone_drivers = []
    outside_zone_drivers = []
    
    for driver in drivers:
        driver_id = driver.get("user_id")
        if not driver_id:
            continue
        location = driver.get("current_location", {})
        lat = location.get("latitude")
        lng = location.get("longitude")
        
        # Calculate distance and ETA
        distance_meters = None
        eta_minutes = None
        
        if lat and lng:
            distance_meters = calculate_distance(hotel_lat, hotel_lng, lat, lng)
            eta_minutes = calculate_eta_minutes(distance_meters)
        
        # Check if in queue
        queue_entry = queue_by_driver.get(driver_id)
        in_queue = queue_entry is not None
        
        # Check if actually in zone (geofence check)
        in_zone = False
        if lat and lng:
            in_zone = is_driver_in_zone(lat, lng, zone)
        
        driver_info = {
            "driver_id": driver_id,
            "driver_name": driver.get("name", "Unknown"),
            "vehicle_class": driver.get("vehicle_class", "Unknown"),
            "photo": driver.get("driver_photo"),
            "latitude": lat,
            "longitude": lng,
            "last_updated": location.get("updated_at"),
            "in_queue": in_queue,
            "queue_position": queue_entry["position"] if queue_entry else None,
            "in_zone": in_zone,
            "distance_meters": round(distance_meters) if distance_meters else None,
            "distance_km": round(distance_meters / 1000, 1) if distance_meters else None,
            "eta_minutes": eta_minutes
        }
        
        if in_zone or in_queue:
            in_zone_drivers.append(driver_info)
        else:
            outside_zone_drivers.append(driver_info)
    
    # Sort in-zone by queue position, outside by distance
    in_zone_drivers.sort(key=lambda d: d.get("queue_position") or 999)
    outside_zone_drivers.sort(key=lambda d: d.get("distance_meters") or 999999)
    
    return {
        "zone_id": zone_id,
        "zone_name": zone.get("name", "Unknown Zone"),
        "in_zone_drivers": in_zone_drivers,
        "outside_zone_drivers": outside_zone_drivers,
        "total_in_zone": len(in_zone_drivers),
        "total_outside_zone": len(outside_zone_drivers),
        "total_online": len(in_zone_drivers) + len(outside_zone_drivers)
    }

# ==================== DRIVER BLOCKING ENDPOINTS ====================

@api_router.post("/drivers/{driver_id}/block")
async def block_driver(driver_id: str, request: Request):
    """Block a driver (hotel/admin only)"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel or admin can block drivers")
    
    # Get driver info
    driver = await db.users.find_one({"user_id": driver_id, "role": "driver"}, {"_id": 0})
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    # Check if already blocked
    existing = await db.blocked_drivers.find_one({"driver_id": driver_id})
    if existing:
        raise HTTPException(status_code=400, detail="Driver is already blocked")
    
    # Get reason from body if provided
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    reason = body.get("reason", "")
    
    # Create block record
    block_doc = {
        "driver_id": driver_id,
        "driver_name": driver["name"],
        "driver_email": driver["email"],
        "vehicle_class": driver.get("vehicle_class"),
        "blocked_by": user["user_id"],
        "blocked_by_name": user["name"],
        "blocked_at": datetime.now(timezone.utc),
        "reason": reason
    }
    await db.blocked_drivers.insert_one(block_doc)
    
    # Remove driver from queue if in queue
    await db.driver_queue.delete_many({"driver_id": driver_id})
    
    # Set driver offline
    await db.users.update_one(
        {"user_id": driver_id},
        {"$set": {"is_online": False, "is_blocked": True}}
    )
    
    logger.info(f"Driver {driver['name']} blocked by {user['name']}")
    return {"message": f"Driver {driver['name']} has been blocked", "driver_id": driver_id}

@api_router.post("/drivers/{driver_id}/unblock")
async def unblock_driver(driver_id: str, request: Request):
    """Unblock a driver (hotel/admin only)"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel or admin can unblock drivers")
    
    # Check if blocked
    block_record = await db.blocked_drivers.find_one({"driver_id": driver_id})
    if not block_record:
        raise HTTPException(status_code=400, detail="Driver is not blocked")
    
    # Remove block
    await db.blocked_drivers.delete_one({"driver_id": driver_id})
    
    # Update driver
    await db.users.update_one(
        {"user_id": driver_id},
        {"$set": {"is_blocked": False}}
    )
    
    driver = await db.users.find_one({"user_id": driver_id}, {"_id": 0, "name": 1})
    logger.info(f"Driver {driver['name']} unblocked by {user['name']}")
    return {"message": f"Driver {driver['name']} has been unblocked", "driver_id": driver_id}

@api_router.get("/drivers/blocked")
async def get_blocked_drivers(request: Request):
    """Get list of blocked drivers (hotel/admin only)"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel or admin can view blocked drivers")
    
    blocked = await db.blocked_drivers.find({}, {"_id": 0}).to_list(100)
    return {"blocked_drivers": blocked, "count": len(blocked)}

@api_router.get("/drivers/{driver_id}/block-status")
async def get_driver_block_status(driver_id: str, request: Request):
    """Check if a driver is blocked"""
    user = await get_current_user(request)
    
    # Driver can check their own status, hotel/admin can check any
    if user["role"] == "driver" and user["user_id"] != driver_id:
        raise HTTPException(status_code=403, detail="Cannot check other driver's status")
    
    block_record = await db.blocked_drivers.find_one({"driver_id": driver_id}, {"_id": 0})
    is_blocked = block_record is not None
    
    return {
        "driver_id": driver_id,
        "is_blocked": is_blocked,
        "blocked_at": block_record["blocked_at"] if block_record else None,
        "reason": block_record.get("reason") if block_record else None
    }

@api_router.put("/drivers/{driver_id}/vehicle-class")
async def update_driver_vehicle_class(driver_id: str, request: Request):
    """Update a driver's vehicle class (hotel/admin only)"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel or admin can update vehicle class")
    
    body = await request.json()
    vehicle_classes = body.get("vehicle_classes", [])
    
    if not vehicle_classes:
        raise HTTPException(status_code=400, detail="At least one vehicle class is required")
    
    valid_classes = ["E-Class", "S-Class", "V-Class"]
    for vc in vehicle_classes:
        if vc not in valid_classes:
            raise HTTPException(status_code=400, detail=f"Invalid vehicle class: {vc}")
    
    # Update the driver
    result = await db.users.update_one(
        {"user_id": driver_id, "role": "driver"},
        {"$set": {"vehicle_class": vehicle_classes}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    driver = await db.users.find_one({"user_id": driver_id}, {"_id": 0, "name": 1})
    logger.info(f"Updated vehicle classes for {driver['name']}: {vehicle_classes}")
    
    return {"message": f"Vehicle classes updated to {', '.join(vehicle_classes)}", "vehicle_classes": vehicle_classes}

@api_router.get("/drivers/all/registered")
async def get_all_registered_drivers(request: Request):
    """Get all registered drivers (hotel/admin only)"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel or admin can view all drivers")
    
    drivers = await db.users.find(
        {"role": "driver"},
        {"_id": 0, "password_hash": 0}
    ).to_list(500)
    
    # Add blocked status to each driver
    blocked_ids = set()
    blocked_drivers = await db.blocked_drivers.find({}, {"driver_id": 1}).to_list(500)
    for bd in blocked_drivers:
        blocked_ids.add(bd["driver_id"])
    
    for driver in drivers:
        driver["is_blocked"] = driver["user_id"] in blocked_ids
    
    return {"drivers": drivers, "count": len(drivers)}

# ==================== DRIVER APPROVAL ENDPOINTS ====================

@api_router.post("/drivers/{driver_id}/approve")
async def approve_driver(driver_id: str, request: Request):
    """Approve a driver to allow them to go online (hotel/admin only)"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel or admin can approve drivers")
    
    # Find the driver
    driver = await db.users.find_one({"user_id": driver_id, "role": "driver"}, {"_id": 0})
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    # Approve the driver
    await db.users.update_one(
        {"user_id": driver_id},
        {"$set": {"is_approved": True, "approved_by": user["user_id"], "approved_at": datetime.now(timezone.utc)}}
    )
    
    logger.info(f"Driver {driver['name']} approved by {user['name']}")
    return {"message": f"Fahrer {driver['name']} wurde freigegeben", "driver_id": driver_id, "is_approved": True}

@api_router.post("/drivers/{driver_id}/revoke-approval")
async def revoke_driver_approval(driver_id: str, request: Request):
    """Revoke a driver's approval (hotel/admin only)"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel or admin can revoke approval")
    
    # Find the driver
    driver = await db.users.find_one({"user_id": driver_id, "role": "driver"}, {"_id": 0})
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    # Revoke approval and set offline
    await db.users.update_one(
        {"user_id": driver_id},
        {"$set": {"is_approved": False, "is_online": False}}
    )
    
    # Remove from queue if in queue
    await db.driver_queue.delete_many({"driver_id": driver_id})
    
    logger.info(f"Driver {driver['name']} approval revoked by {user['name']}")
    return {"message": f"Freigabe für Fahrer {driver['name']} wurde entzogen", "driver_id": driver_id, "is_approved": False}

@api_router.get("/drivers/{driver_id}/approval-status")
async def get_driver_approval_status(driver_id: str, request: Request):
    """Check if a driver is approved"""
    user = await get_current_user(request)
    
    # Driver can check their own status, hotel/admin can check any
    if user["role"] == "driver" and user["user_id"] != driver_id:
        raise HTTPException(status_code=403, detail="Cannot check other driver's status")
    
    driver = await db.users.find_one({"user_id": driver_id, "role": "driver"}, {"_id": 0, "password_hash": 0})
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    return {
        "driver_id": driver_id,
        "driver_name": driver.get("name"),
        "is_approved": driver.get("is_approved", False),
        "approved_by": driver.get("approved_by"),
        "approved_at": driver.get("approved_at")
    }

# ==================== PASSWORD CHANGE ENDPOINT ====================

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

@api_router.post("/auth/change-password")
async def change_password(password_data: PasswordChangeRequest, request: Request):
    """Change password for current user (hotel/admin only for now)"""
    user = await get_current_user(request)
    
    # Get user with password hash
    user_full = await db.users.find_one({"user_id": user["user_id"]}, {"_id": 0})
    if not user_full:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not user_full.get("password_hash"):
        raise HTTPException(status_code=400, detail="Cannot change password for Google-authenticated accounts")
    
    # Verify current password
    if not verify_password(password_data.current_password, user_full["password_hash"]):
        raise HTTPException(status_code=401, detail="Aktuelles Passwort ist falsch")
    
    # Validate new password
    if len(password_data.new_password) < 8:
        raise HTTPException(status_code=400, detail="Neues Passwort muss mindestens 8 Zeichen haben")
    
    # Hash and update new password
    new_hash = hash_password(password_data.new_password)
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": {"password_hash": new_hash, "password_changed_at": datetime.now(timezone.utc)}}
    )
    
    logger.info(f"Password changed for user {user['email']}")
    return {"message": "Passwort wurde erfolgreich geändert"}

# ==================== HOTEL ZONE ENDPOINTS ====================

@api_router.post("/zones", response_model=HotelZone)
async def create_zone(zone_data: HotelZoneCreate, request: Request):
    """Create a hotel zone (hotel users only)"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel users can create zones")
    
    zone_id = f"zone_{uuid.uuid4().hex[:12]}"
    zone_doc = {
        "zone_id": zone_id,
        "hotel_user_id": user["user_id"],
        "name": zone_data.name,
        "latitude": zone_data.latitude,
        "longitude": zone_data.longitude,
        "radius": zone_data.radius,
        "created_at": datetime.now(timezone.utc)
    }
    
    await db.hotel_zones.insert_one(zone_doc)
    return HotelZone(**zone_doc)

@api_router.get("/zones", response_model=List[HotelZone])
async def get_zones(request: Request):
    """Get all hotel zones"""
    zones = await db.hotel_zones.find({}, {"_id": 0}).to_list(100)
    return [HotelZone(**z) for z in zones]

@api_router.get("/zones/my", response_model=Optional[HotelZone])
async def get_my_zone(request: Request):
    """Get zone for current hotel user"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotel users can access this")
    
    zone = await db.hotel_zones.find_one({"hotel_user_id": user["user_id"]}, {"_id": 0})
    if not zone:
        return None
    return HotelZone(**zone)

@api_router.put("/zones/{zone_id}", response_model=HotelZone)
async def update_zone(zone_id: str, zone_data: HotelZoneCreate, request: Request):
    """Update a hotel zone"""
    user = await get_current_user(request)
    
    zone = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    # Allow hotel owner or admin to update
    if zone["hotel_user_id"] != user["user_id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_data = {
        "name": zone_data.name,
        "latitude": zone_data.latitude,
        "longitude": zone_data.longitude,
    }
    
    # Handle polygon or radius
    if zone_data.polygon and len(zone_data.polygon) >= 3:
        update_data["polygon"] = [{"lat": p.lat, "lng": p.lng} for p in zone_data.polygon]
        update_data["radius"] = None  # Clear radius when using polygon
    elif zone_data.radius:
        update_data["radius"] = zone_data.radius
        update_data["polygon"] = None  # Clear polygon when using radius
    
    await db.hotel_zones.update_one(
        {"zone_id": zone_id},
        {"$set": update_data}
    )
    
    updated = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    return HotelZone(**updated)

@api_router.put("/zones/{zone_id}/polygon")
async def update_zone_polygon(zone_id: str, request: Request):
    """Update just the polygon of a hotel zone"""
    user = await get_current_user(request)
    
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    zone = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    body = await request.json()
    polygon = body.get("polygon", [])
    
    if len(polygon) < 3:
        raise HTTPException(status_code=400, detail="Polygon must have at least 3 points")
    
    # Calculate center from polygon
    avg_lat = sum(p['lat'] for p in polygon) / len(polygon)
    avg_lng = sum(p['lng'] for p in polygon) / len(polygon)
    
    await db.hotel_zones.update_one(
        {"zone_id": zone_id},
        {"$set": {
            "polygon": polygon,
            "latitude": avg_lat,
            "longitude": avg_lng,
            "radius": None
        }}
    )
    
    updated = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    logger.info(f"Updated zone {zone_id} with {len(polygon)}-point polygon")
    return {"message": "Polygon updated", "zone": updated}

# ==================== QUEUE ENDPOINTS ====================

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters using Haversine formula"""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def calculate_eta_minutes(distance_meters: float, avg_speed_kmh: float = 30) -> int:
    """
    Calculate estimated time of arrival in minutes based on distance.
    Default avg speed is 30 km/h for city driving in Zurich.
    """
    if distance_meters <= 0:
        return 0
    distance_km = distance_meters / 1000
    time_hours = distance_km / avg_speed_kmh
    time_minutes = int(time_hours * 60)
    return max(1, time_minutes)  # At least 1 minute

async def find_nearest_online_driver(
    zone: dict, 
    exclude_driver_ids: list = None,
    vehicle_class: str = None
) -> dict:
    """
    Find the nearest online driver outside the queue.
    Used when the queue is empty and we need to cascade to nearby drivers.
    
    Returns driver dict with distance_meters and eta_minutes, or None if no driver found.
    """
    if exclude_driver_ids is None:
        exclude_driver_ids = []
    
    # Build query for online drivers not in the exclusion list
    query = {
        "role": "driver",
        "is_online": True,
        "user_id": {"$nin": exclude_driver_ids},
        "current_location": {"$ne": None},
        "is_blocked": {"$ne": True}
    }
    
    # Filter by vehicle class if specified
    if vehicle_class:
        query["$or"] = [
            {"vehicle_class": vehicle_class},
            {"vehicle_classes": vehicle_class}
        ]
    
    # Get all online drivers
    drivers = await db.users.find(query, {"_id": 0, "password_hash": 0}).to_list(100)
    
    if not drivers:
        return None
    
    # Get drivers NOT in the queue for this zone
    queue_driver_ids = set()
    queue_entries = await db.driver_queue.find(
        {"zone_id": zone["zone_id"], "status": "waiting"},
        {"driver_id": 1}
    ).to_list(100)
    for entry in queue_entries:
        queue_driver_ids.add(entry["driver_id"])
    
    # Calculate distance for each driver and sort
    hotel_lat = zone["latitude"]
    hotel_lng = zone["longitude"]
    
    drivers_with_distance = []
    for driver in drivers:
        # Skip drivers in queue
        if driver["user_id"] in queue_driver_ids:
            continue
        
        loc = driver.get("current_location", {})
        if not loc or not loc.get("latitude") or not loc.get("longitude"):
            continue
        
        distance = calculate_distance(
            hotel_lat, hotel_lng,
            loc["latitude"], loc["longitude"]
        )
        eta = calculate_eta_minutes(distance)
        
        drivers_with_distance.append({
            **driver,
            "distance_meters": distance,
            "eta_minutes": eta
        })
    
    if not drivers_with_distance:
        return None
    
    # Sort by distance and return nearest
    drivers_with_distance.sort(key=lambda d: d["distance_meters"])
    return drivers_with_distance[0]

@api_router.post("/queue/join")
async def join_queue(request: Request):
    """Join queue for a hotel zone based on driver's location"""
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can join queue")
    
    # Check if driver is blocked
    is_blocked = await db.blocked_drivers.find_one({"driver_id": user["user_id"]})
    if is_blocked:
        raise HTTPException(status_code=403, detail="You are blocked and cannot join the queue. Please contact the hotel.")
    
    body = await request.json()
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    zone_id = body.get("zone_id")
    
    if not all([latitude, longitude, zone_id]):
        raise HTTPException(status_code=400, detail="latitude, longitude, and zone_id are required")
    
    # Get zone
    zone = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    # Check if driver is within zone
    if not is_driver_in_zone(latitude, longitude, zone):
        if zone.get('polygon'):
            raise HTTPException(status_code=400, detail="You must be within the hotel's pickup zone to join the queue")
        else:
            distance = calculate_distance(latitude, longitude, zone["latitude"], zone["longitude"])
            raise HTTPException(status_code=400, detail=f"You are {int(distance)}m away. Must be within {int(zone['radius'])}m of the hotel")
    
    # Check if already in queue
    existing = await db.driver_queue.find_one({
        "driver_id": user["user_id"],
        "zone_id": zone_id,
        "status": "waiting"
    })
    if existing:
        return {"message": "Already in queue", "position": existing["position"], "queue_id": existing["queue_id"]}
    
    # Get current max position
    last_entry = await db.driver_queue.find_one(
        {"zone_id": zone_id, "status": "waiting"},
        sort=[("position", -1)]
    )
    next_position = (last_entry["position"] + 1) if last_entry else 1
    
    # Add to queue
    queue_id = f"queue_{uuid.uuid4().hex[:12]}"
    queue_doc = {
        "queue_id": queue_id,
        "driver_id": user["user_id"],
        "driver_name": user["name"],
        "vehicle_class": user.get("vehicle_class", "Unknown"),
        "zone_id": zone_id,
        "position": next_position,
        "entered_at": datetime.now(timezone.utc),
        "status": "waiting",
        "left_zone_at": None,
        "last_location": {"latitude": latitude, "longitude": longitude}
    }
    
    await db.driver_queue.insert_one(queue_doc)
    
    return {"message": "Joined queue", "position": next_position, "queue_id": queue_id}

@api_router.post("/queue/update-location")
async def update_queue_location(request: Request):
    """Update driver's location while in queue"""
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can update location")
    
    body = await request.json()
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    zone_id = body.get("zone_id")
    
    # Get zone
    zone = await db.hotel_zones.find_one({"zone_id": zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    # Check if in queue
    queue_entry = await db.driver_queue.find_one({
        "driver_id": user["user_id"],
        "zone_id": zone_id,
        "status": "waiting"
    })
    
    if not queue_entry:
        return {"in_queue": False, "message": "Not in queue"}
    
    # Check if within zone
    distance = calculate_distance(latitude, longitude, zone["latitude"], zone["longitude"])
    in_zone = distance <= zone["radius"]
    
    if in_zone:
        # Update location and clear left_zone_at
        await db.driver_queue.update_one(
            {"queue_id": queue_entry["queue_id"]},
            {"$set": {
                "last_location": {"latitude": latitude, "longitude": longitude},
                "left_zone_at": None
            }}
        )
        return {"in_queue": True, "in_zone": True, "position": queue_entry["position"]}
    else:
        # Driver left zone - start grace period
        now = datetime.now(timezone.utc)
        left_at = queue_entry.get("left_zone_at")
        
        if not left_at:
            # First time leaving
            await db.driver_queue.update_one(
                {"queue_id": queue_entry["queue_id"]},
                {"$set": {"left_zone_at": now}}
            )
            return {"in_queue": True, "in_zone": False, "grace_period": True, "seconds_remaining": 300}
        else:
            # Check if grace period expired (5 minutes)
            if isinstance(left_at, str):
                left_at = datetime.fromisoformat(left_at)
            if left_at.tzinfo is None:
                left_at = left_at.replace(tzinfo=timezone.utc)
            
            elapsed = (now - left_at).total_seconds()
            if elapsed > 300:  # 5 minutes
                # Remove from queue
                await db.driver_queue.update_one(
                    {"queue_id": queue_entry["queue_id"]},
                    {"$set": {"status": "left"}}
                )
                # Recalculate positions
                await recalculate_positions(zone_id)
                return {"in_queue": False, "message": "Removed from queue (grace period expired)"}
            else:
                return {
                    "in_queue": True, 
                    "in_zone": False, 
                    "grace_period": True, 
                    "seconds_remaining": int(300 - elapsed)
                }

@api_router.post("/queue/leave")
async def leave_queue(request: Request):
    """Manually leave the queue"""
    user = await get_current_user(request)
    body = await request.json()
    zone_id = body.get("zone_id")
    
    result = await db.driver_queue.update_one(
        {"driver_id": user["user_id"], "zone_id": zone_id, "status": "waiting"},
        {"$set": {"status": "left"}}
    )
    
    if result.modified_count > 0:
        await recalculate_positions(zone_id)
    
    return {"message": "Left queue"}

@api_router.get("/queue/{zone_id}")
async def get_queue(zone_id: str, request: Request):
    """Get current queue for a zone"""
    await get_current_user(request)  # Just verify auth
    
    queue = await db.driver_queue.find(
        {"zone_id": zone_id, "status": "waiting"},
        {"_id": 0}
    ).sort("position", 1).to_list(100)
    
    return {"queue": queue, "count": len(queue)}

@api_router.get("/queue/my/status")
async def get_my_queue_status(request: Request):
    """Get current driver's queue status"""
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can check queue status")
    
    queue_entry = await db.driver_queue.find_one(
        {"driver_id": user["user_id"], "status": "waiting"},
        {"_id": 0}
    )
    
    if not queue_entry:
        return {"in_queue": False}
    
    return {"in_queue": True, "entry": queue_entry}

async def recalculate_positions(zone_id: str):
    """Recalculate positions after someone leaves"""
    queue = await db.driver_queue.find(
        {"zone_id": zone_id, "status": "waiting"},
        {"_id": 0}
    ).sort("entered_at", 1).to_list(100)
    
    for i, entry in enumerate(queue):
        await db.driver_queue.update_one(
            {"queue_id": entry["queue_id"]},
            {"$set": {"position": i + 1}}
        )

# ==================== RIDE ENDPOINTS ====================

@api_router.post("/rides/request")
async def request_ride(ride_data: RideCreate, request: Request):
    """
    Hotel requests a taxi - uses cascading system:
    1. First tries queue (FIFO)
    2. If queue empty, finds nearest online driver outside zone
    """
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can request rides")
    
    # Get zone
    zone = await db.hotel_zones.find_one({"zone_id": ride_data.zone_id}, {"_id": 0})
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    if zone["hotel_user_id"] != user["user_id"] and zone["hotel_user_id"] != "system":
        raise HTTPException(status_code=403, detail="Not your zone")
    
    driver = None
    queue_entry = None
    is_cascade_request = False  # Track if this is a cascading (out-of-zone) request
    
    if ride_data.driver_id:
        # Specific driver requested
        driver = await db.users.find_one(
            {"user_id": ride_data.driver_id, "role": "driver"},
            {"_id": 0, "password_hash": 0}
        )
        if not driver:
            raise HTTPException(status_code=404, detail="Driver not found")
        
        # Check if driver is in queue for this zone
        queue_entry = await db.driver_queue.find_one(
            {"driver_id": ride_data.driver_id, "zone_id": ride_data.zone_id, "status": "waiting"},
            {"_id": 0}
        )
    else:
        # Get first driver in queue - optionally filter by vehicle class
        query = {"zone_id": ride_data.zone_id, "status": "waiting"}
        
        # If vehicle class is specified, filter queue by it
        if ride_data.vehicle_class:
            if ride_data.vehicle_class not in VEHICLE_CLASSES:
                raise HTTPException(status_code=400, detail=f"Invalid vehicle class. Must be one of: {VEHICLE_CLASSES}")
            query["vehicle_class"] = ride_data.vehicle_class
        
        queue_entry = await db.driver_queue.find_one(
            query,
            {"_id": 0},
            sort=[("position", 1)]
        )
        
        if queue_entry:
            # Found driver in queue
            driver = await db.users.find_one(
                {"user_id": queue_entry["driver_id"]},
                {"_id": 0, "password_hash": 0}
            )
        else:
            # CASCADING: Queue is empty - find nearest online driver outside zone
            logger.info(f"Queue empty for zone {ride_data.zone_id}, searching for nearest online driver...")
            
            nearest_driver = await find_nearest_online_driver(
                zone=zone,
                exclude_driver_ids=[],
                vehicle_class=ride_data.vehicle_class
            )
            
            if nearest_driver:
                driver = nearest_driver
                is_cascade_request = True
                logger.info(f"Found nearest driver: {driver['name']} at {driver['distance_meters']:.0f}m, ETA: {driver['eta_minutes']} min")
            else:
                vehicle_msg = f" with {ride_data.vehicle_class}" if ride_data.vehicle_class else ""
                raise HTTPException(
                    status_code=404, 
                    detail=f"No drivers{vehicle_msg} available - neither in queue nor online nearby"
                )
    
    # Create ride
    ride_id = f"ride_{uuid.uuid4().hex[:12]}"
    ride_doc = {
        "ride_id": ride_id,
        "driver_id": driver["user_id"],
        "driver_name": driver["name"],
        "vehicle_class": driver.get("vehicle_class", ["Unknown"])[0] if isinstance(driver.get("vehicle_class"), list) else driver.get("vehicle_class", "Unknown"),
        "zone_id": ride_data.zone_id,
        "hotel_user_id": user["user_id"],
        "customer_info": ride_data.customer_info,
        "requested_vehicle_class": ride_data.vehicle_class,
        "destination": None,
        "destination_lat": None,
        "destination_lng": None,
        "price": None,
        "status": "requested",
        "driver_location": None,
        "created_at": datetime.now(timezone.utc),
        "completed_at": None,
        "is_cascade_request": is_cascade_request,  # Track if this came from cascading (driver outside zone)
        "driver_distance_meters": driver.get("distance_meters"),  # Distance when cascading
        "driver_eta_minutes": driver.get("eta_minutes"),  # ETA when cascading
        "declined_by": []  # Track drivers who declined (for cascading)
    }
    
    await db.rides.insert_one(ride_doc)
    
    # Update driver queue status if was in queue
    if queue_entry:
        await db.driver_queue.update_one(
            {"queue_id": queue_entry["queue_id"]},
            {"$set": {"status": "assigned"}}
        )
        await recalculate_positions(ride_data.zone_id)
    
    # Send push notification to driver
    if driver.get("push_token"):
        await send_push_notification(
            driver["push_token"],
            "🚕 New Ride Request!",
            f"You have a new ride request. Tap to accept or decline.",
            {"ride_id": ride_id, "type": "ride_request"}
        )
        logger.info(f"Sent push notification to driver {driver['user_id']}")
    
    # Log cascade request details
    if is_cascade_request:
        logger.info(f"Cascade ride request sent to driver {driver['name']} (distance: {driver.get('distance_meters', 0):.0f}m, ETA: {driver.get('eta_minutes', 0)} min)")
    
    return Ride(**ride_doc)

@api_router.get("/rides/active/driver")
async def get_driver_active_ride(request: Request):
    """Get driver's active ride"""
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can access this")
    
    ride = await db.rides.find_one(
        {"driver_id": user["user_id"], "status": {"$in": ["requested", "accepted", "in_progress"]}},
        {"_id": 0}
    )
    
    if not ride:
        return None
    
    return Ride(**ride)

@api_router.get("/rides/active/hotel")
async def get_hotel_active_rides(request: Request):
    """Get hotel's active rides"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can access this")
    
    rides = await db.rides.find(
        {"hotel_user_id": user["user_id"], "status": {"$in": ["requested", "accepted", "in_progress"]}},
        {"_id": 0}
    ).to_list(10)
    
    return [Ride(**r) for r in rides]

@api_router.post("/rides/{ride_id}/accept")
async def accept_ride(ride_id: str, request: Request):
    """Driver accepts a ride"""
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
    
    await db.rides.update_one(
        {"ride_id": ride_id},
        {"$set": {"status": "accepted"}}
    )
    
    return {"message": "Ride accepted"}

@api_router.post("/rides/{ride_id}/decline")
async def decline_ride(ride_id: str, request: Request):
    """
    Driver declines a ride - uses cascading system:
    1. First tries next driver in queue
    2. If queue empty, finds nearest online driver outside zone
    """
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
    
    # Track this driver as having declined
    declined_by = ride.get("declined_by", [])
    if user["user_id"] not in declined_by:
        declined_by.append(user["user_id"])
    
    # Get zone for cascading
    zone = await db.hotel_zones.find_one({"zone_id": ride["zone_id"]}, {"_id": 0})
    
    # Find next driver in queue with same vehicle class (if specified)
    query = {"zone_id": ride["zone_id"], "status": "waiting", "driver_id": {"$ne": user["user_id"]}}
    
    # If original request specified a vehicle class, keep filtering by it
    if ride.get("requested_vehicle_class"):
        query["vehicle_class"] = ride["requested_vehicle_class"]
    
    next_queue_entry = await db.driver_queue.find_one(
        query,
        {"_id": 0},
        sort=[("position", 1)]
    )
    
    next_driver = None
    is_cascade = False
    
    if next_queue_entry:
        # Found driver in queue
        next_driver = await db.users.find_one(
            {"user_id": next_queue_entry["driver_id"]},
            {"_id": 0, "password_hash": 0}
        )
        
        # Update queue statuses
        await db.driver_queue.update_one(
            {"queue_id": next_queue_entry["queue_id"]},
            {"$set": {"status": "assigned"}}
        )
    else:
        # CASCADING: No driver in queue - find nearest online driver outside zone
        if zone:
            logger.info(f"No more drivers in queue for ride {ride_id}, cascading to nearest online driver...")
            
            nearest_driver = await find_nearest_online_driver(
                zone=zone,
                exclude_driver_ids=declined_by,  # Exclude all drivers who already declined
                vehicle_class=ride.get("requested_vehicle_class")
            )
            
            if nearest_driver:
                next_driver = nearest_driver
                is_cascade = True
                logger.info(f"Cascade to driver: {next_driver['name']} at {next_driver.get('distance_meters', 0):.0f}m")
    
    if next_driver:
        # Update ride to next driver
        update_data = {
            "driver_id": next_driver["user_id"],
            "driver_name": next_driver["name"],
            "vehicle_class": next_driver.get("vehicle_class", "Unknown"),
            "declined_by": declined_by,
            "is_cascade_request": is_cascade
        }
        
        # Add distance/ETA for cascade requests
        if is_cascade:
            update_data["driver_distance_meters"] = next_driver.get("distance_meters")
            update_data["driver_eta_minutes"] = next_driver.get("eta_minutes")
        
        await db.rides.update_one(
            {"ride_id": ride_id},
            {"$set": update_data}
        )
        
        # Put declining driver back in queue at end (if was in queue)
        current_queue = await db.driver_queue.find_one(
            {"driver_id": user["user_id"], "zone_id": ride["zone_id"]},
            {"_id": 0}
        )
        if current_queue:
            # Get max position
            max_pos_entry = await db.driver_queue.find_one(
                {"zone_id": ride["zone_id"]},
                {"_id": 0},
                sort=[("position", -1)]
            )
            new_pos = (max_pos_entry["position"] + 1) if max_pos_entry else 1
            
            await db.driver_queue.update_one(
                {"queue_id": current_queue["queue_id"]},
                {"$set": {"status": "waiting", "position": new_pos}}
            )
        
        await recalculate_positions(ride["zone_id"])
        
        # Send push notification to next driver
        if next_driver.get("push_token"):
            await send_push_notification(
                next_driver["push_token"],
                "🚕 New Ride Request!",
                f"You have a new ride request. Tap to accept or decline.",
                {"ride_id": ride_id, "type": "ride_request"}
            )
            logger.info(f"Ride passed to driver: {next_driver['user_id']} (cascade: {is_cascade})")
        
        cascade_info = f" (ETA: {next_driver.get('eta_minutes', '?')} min)" if is_cascade else ""
        return {
            "message": f"Ride declined, passed to {'nearby' if is_cascade else 'next'} driver{cascade_info}", 
            "next_driver": next_driver["name"],
            "is_cascade": is_cascade,
            "eta_minutes": next_driver.get("eta_minutes") if is_cascade else None
        }
    else:
        # No more drivers available anywhere - cancel the ride
        await db.rides.update_one(
            {"ride_id": ride_id},
            {"$set": {"status": "cancelled", "declined_by": declined_by}}
        )
        
        # Put declining driver back in queue (if was in queue)
        current_queue = await db.driver_queue.find_one(
            {"driver_id": user["user_id"], "zone_id": ride["zone_id"]},
            {"_id": 0}
        )
        if current_queue:
            await db.driver_queue.update_one(
                {"queue_id": current_queue["queue_id"]},
                {"$set": {"status": "waiting"}}
            )
            await recalculate_positions(ride["zone_id"])
        
        logger.info(f"Ride {ride_id} cancelled - no drivers available (declined by {len(declined_by)} drivers)")
        return {"message": "Ride declined, no more drivers available - ride cancelled", "is_cascade": False}

@api_router.post("/rides/{ride_id}/start")
async def start_ride(ride_id: str, request: Request):
    """Driver starts the ride (picked up customer)"""
    user = await get_current_user(request)
    body = await request.json()
    
    ride = await db.rides.find_one({"ride_id": ride_id}, {"_id": 0})
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    
    if ride["driver_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your ride")
    
    await db.rides.update_one(
        {"ride_id": ride_id},
        {"$set": {
            "status": "in_progress",
            "destination": body.get("destination"),
            "destination_lat": body.get("destination_lat"),
            "destination_lng": body.get("destination_lng")
        }}
    )
    
    return {"message": "Ride started"}

@api_router.post("/rides/{ride_id}/update-location")
async def update_ride_location(ride_id: str, request: Request):
    """Update driver location during ride"""
    user = await get_current_user(request)
    body = await request.json()
    
    ride = await db.rides.find_one({"ride_id": ride_id}, {"_id": 0})
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    
    if ride["driver_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your ride")
    
    location = {
        "latitude": body.get("latitude"),
        "longitude": body.get("longitude"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    await db.rides.update_one(
        {"ride_id": ride_id},
        {"$set": {"driver_location": location}}
    )
    
    return {"message": "Location updated"}

@api_router.post("/rides/{ride_id}/complete")
async def complete_ride(ride_id: str, request: Request):
    """Complete a ride with price, commission, and final destination"""
    user = await get_current_user(request)
    body = await request.json()
    
    ride = await db.rides.find_one({"ride_id": ride_id}, {"_id": 0})
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    
    if ride["driver_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your ride")
    
    # Get final location, price, commission, and ride type
    price = body.get("price")
    commission = body.get("commission", 0)
    ride_type = body.get("ride_type", "other")  # "airport", "city", "other"
    final_destination = body.get("destination", ride.get("destination"))
    final_lat = body.get("latitude")
    final_lng = body.get("longitude")
    
    update_data = {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc),
        "ride_type": ride_type,
        "commission": float(commission) if commission else 0,
        "commission_paid": False  # Will be set to True after payment
    }
    
    if price is not None:
        update_data["price"] = float(price)
    
    if final_destination:
        update_data["destination"] = final_destination
    
    if final_lat and final_lng:
        update_data["destination_lat"] = final_lat
        update_data["destination_lng"] = final_lng
        update_data["driver_location"] = {
            "latitude": final_lat,
            "longitude": final_lng,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    await db.rides.update_one(
        {"ride_id": ride_id},
        {"$set": update_data}
    )
    
    return {
        "message": "Ride completed", 
        "price": price, 
        "commission": commission,
        "ride_type": ride_type
    }

@api_router.post("/rides/{ride_id}/pay-commission")
async def pay_commission(ride_id: str, request: Request):
    """Mark commission as paid (after SumUp payment)"""
    user = await get_current_user(request)
    body = await request.json()
    
    ride = await db.rides.find_one({"ride_id": ride_id}, {"_id": 0})
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    
    if ride["driver_id"] != user["user_id"]:
        raise HTTPException(status_code=403, detail="Not your ride")
    
    payment_reference = body.get("payment_reference")
    
    await db.rides.update_one(
        {"ride_id": ride_id},
        {"$set": {
            "commission_paid": True,
            "commission_paid_at": datetime.now(timezone.utc),
            "payment_reference": payment_reference
        }}
    )
    
    return {"message": "Commission marked as paid", "ride_id": ride_id}

@api_router.get("/rides/{ride_id}")
async def get_ride(ride_id: str, request: Request):
    """Get ride details"""
    await get_current_user(request)
    
    ride = await db.rides.find_one({"ride_id": ride_id}, {"_id": 0})
    if not ride:
        raise HTTPException(status_code=404, detail="Ride not found")
    
    return Ride(**ride)

@api_router.get("/rides/history/hotel")
async def get_hotel_ride_history(request: Request):
    """Get hotel's completed rides history"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can access this")
    
    rides = await db.rides.find(
        {"hotel_user_id": user["user_id"], "status": "completed"},
        {"_id": 0}
    ).sort("completed_at", -1).to_list(50)
    
    return [Ride(**r) for r in rides]

# ==================== STATISTICS ENDPOINTS ====================

@api_router.get("/stats/driver/monthly")
async def get_driver_monthly_stats(request: Request):
    """Get driver's monthly statistics"""
    user = await get_current_user(request)
    if user["role"] not in ["driver", "admin"]:
        raise HTTPException(status_code=403, detail="Only drivers can access this")
    
    # Get current month start
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    
    # Get all completed rides this month
    rides = await db.rides.find(
        {
            "driver_id": user["user_id"],
            "status": "completed",
            "completed_at": {"$gte": month_start}
        },
        {"_id": 0}
    ).to_list(500)
    
    # Calculate statistics
    total_rides = len(rides)
    total_commission = sum(r.get("commission", 0) or 0 for r in rides)
    total_revenue = sum(r.get("price", 0) or 0 for r in rides)
    
    # Stats by ride type
    airport_rides = [r for r in rides if r.get("ride_type") == "airport"]
    city_rides = [r for r in rides if r.get("ride_type") == "city"]
    other_rides = [r for r in rides if r.get("ride_type") not in ["airport", "city"]]
    
    # Daily breakdown
    daily_stats = {}
    for ride in rides:
        completed_at = ride.get("completed_at")
        if completed_at:
            if isinstance(completed_at, str):
                completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
            day_key = completed_at.strftime("%Y-%m-%d")
            if day_key not in daily_stats:
                daily_stats[day_key] = {"rides": 0, "commission": 0, "revenue": 0}
            daily_stats[day_key]["rides"] += 1
            daily_stats[day_key]["commission"] += ride.get("commission", 0) or 0
            daily_stats[day_key]["revenue"] += ride.get("price", 0) or 0
    
    return {
        "month": now.strftime("%B %Y"),
        "month_start": month_start.isoformat(),
        "total_rides": total_rides,
        "total_commission_paid": total_commission,
        "total_revenue": total_revenue,
        "by_type": {
            "airport": {
                "rides": len(airport_rides),
                "commission": sum(r.get("commission", 0) or 0 for r in airport_rides)
            },
            "city": {
                "rides": len(city_rides),
                "commission": sum(r.get("commission", 0) or 0 for r in city_rides)
            },
            "other": {
                "rides": len(other_rides),
                "commission": sum(r.get("commission", 0) or 0 for r in other_rides)
            }
        },
        "daily_breakdown": daily_stats
    }

@api_router.get("/stats/hotel/monthly")
async def get_hotel_monthly_stats(request: Request):
    """Get hotel's monthly statistics"""
    user = await get_current_user(request)
    if user["role"] not in ["hotel", "admin"]:
        raise HTTPException(status_code=403, detail="Only hotels can access this")
    
    # Get current month start
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    
    # Get all completed rides this month
    rides = await db.rides.find(
        {
            "hotel_user_id": user["user_id"],
            "status": "completed",
            "completed_at": {"$gte": month_start}
        },
        {"_id": 0}
    ).to_list(500)
    
    # Calculate total statistics
    total_rides = len(rides)
    total_commission = sum(r.get("commission", 0) or 0 for r in rides)
    total_revenue = sum(r.get("price", 0) or 0 for r in rides)
    
    # Stats by driver
    driver_stats = {}
    for ride in rides:
        driver_id = ride.get("driver_id")
        driver_name = ride.get("driver_name", "Unknown")
        vehicle_class = ride.get("vehicle_class", "Unknown")
        
        if driver_id not in driver_stats:
            driver_stats[driver_id] = {
                "driver_id": driver_id,
                "driver_name": driver_name,
                "vehicle_class": vehicle_class,
                "rides": 0,
                "commission": 0,
                "revenue": 0
            }
        driver_stats[driver_id]["rides"] += 1
        driver_stats[driver_id]["commission"] += ride.get("commission", 0) or 0
        driver_stats[driver_id]["revenue"] += ride.get("price", 0) or 0
    
    # Stats by vehicle class
    vehicle_stats = {"E-Class": {"rides": 0, "commission": 0, "revenue": 0},
                     "S-Class": {"rides": 0, "commission": 0, "revenue": 0},
                     "V-Class": {"rides": 0, "commission": 0, "revenue": 0}}
    
    for ride in rides:
        vc = ride.get("vehicle_class", "Unknown")
        if vc in vehicle_stats:
            vehicle_stats[vc]["rides"] += 1
            vehicle_stats[vc]["commission"] += ride.get("commission", 0) or 0
            vehicle_stats[vc]["revenue"] += ride.get("price", 0) or 0
    
    # Stats by ride type
    type_stats = {"airport": {"rides": 0, "commission": 0},
                  "city": {"rides": 0, "commission": 0},
                  "other": {"rides": 0, "commission": 0}}
    
    for ride in rides:
        rt = ride.get("ride_type", "other")
        if rt not in type_stats:
            rt = "other"
        type_stats[rt]["rides"] += 1
        type_stats[rt]["commission"] += ride.get("commission", 0) or 0
    
    # Daily breakdown
    daily_stats = {}
    for ride in rides:
        completed_at = ride.get("completed_at")
        if completed_at:
            if isinstance(completed_at, str):
                completed_at = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
            day_key = completed_at.strftime("%Y-%m-%d")
            if day_key not in daily_stats:
                daily_stats[day_key] = {"rides": 0, "commission": 0, "revenue": 0}
            daily_stats[day_key]["rides"] += 1
            daily_stats[day_key]["commission"] += ride.get("commission", 0) or 0
            daily_stats[day_key]["revenue"] += ride.get("price", 0) or 0
    
    return {
        "month": now.strftime("%B %Y"),
        "month_start": month_start.isoformat(),
        "total_rides": total_rides,
        "total_commission_received": total_commission,
        "total_revenue": total_revenue,
        "by_driver": list(driver_stats.values()),
        "by_vehicle_class": vehicle_stats,
        "by_ride_type": type_stats,
        "daily_breakdown": daily_stats
    }

# ==================== PUSH NOTIFICATIONS ====================

@api_router.post("/notifications/register-token")
async def register_push_token(token_data: PushTokenRegister, request: Request):
    """Register a push notification token for a driver"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Update user's push token
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": {"push_token": token_data.push_token}}
    )
    
    return {"message": "Push token registered"}

async def send_push_notification(push_token: str, title: str, body: str, data: dict = None):
    """Send push notification via Expo Push API"""
    try:
        message = {
            "to": push_token,
            "sound": "default",
            "title": title,
            "body": body,
            "data": data or {},
            "priority": "high",
            "channelId": "ride-requests",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://exp.host/--/api/v2/push/send",
                json=message,
                headers={"Content-Type": "application/json"}
            )
            logger.info(f"Push notification sent: {response.status_code}")
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Error sending push notification: {e}")
        return False

# ==================== INITIALIZATION ====================

@api_router.get("/")
async def root():
    return {"message": "Taxi Queue API is running"}

@api_router.get("/health")
async def health():
    return {"status": "healthy"}

@api_router.get("/vehicle-classes")
async def get_vehicle_classes():
    """Get available vehicle classes"""
    return {"vehicle_classes": VEHICLE_CLASSES}

# Initialize default Baur au Lac zone on startup
@app.on_event("startup")
async def init_default_zone():
    """Create or ensure the default Baur au Lac zone exists with the defined polygon"""
    
    # Default polygon zone - defined by hotel management
    # This is the permanent default pickup zone for Baur au Lac
    default_polygon = [
        {"lat": 47.367800793449014, "lng": 8.53916359557819},
        {"lat": 47.366525972979694, "lng": 8.540306219777566},
        {"lat": 47.36693151709968, "lng": 8.541814116331826},
        {"lat": 47.36814364875325, "lng": 8.541799408882968}
    ]
    
    # Calculate center from polygon
    avg_lat = sum(p['lat'] for p in default_polygon) / len(default_polygon)
    avg_lng = sum(p['lng'] for p in default_polygon) / len(default_polygon)
    
    # Delete any test zones or duplicates (keep only zone_baur_au_lac)
    await db.hotel_zones.delete_many({"zone_id": {"$ne": "zone_baur_au_lac"}})
    
    # Check if the main zone exists
    existing = await db.hotel_zones.find_one({"zone_id": "zone_baur_au_lac"})
    
    if existing:
        # Ensure it has the correct polygon and is linked to hotel
        # Only update if explicitly empty (user can modify for special occasions)
        if not existing.get("polygon"):
            await db.hotel_zones.update_one(
                {"zone_id": "zone_baur_au_lac"},
                {"$set": {
                    "polygon": default_polygon,
                    "latitude": avg_lat,
                    "longitude": avg_lng,
                    "radius": None,
                    "hotel_user_id": "hotel_bauraulac_001"
                }}
            )
            logger.info("Updated Baur au Lac zone with default polygon")
    else:
        # Create the default zone
        default_zone = {
            "zone_id": "zone_baur_au_lac",
            "hotel_user_id": "hotel_bauraulac_001",
            "name": "Baur au Lac",
            "latitude": avg_lat,
            "longitude": avg_lng,
            "radius": None,
            "polygon": default_polygon,
            "created_at": datetime.now(timezone.utc)
        }
        await db.hotel_zones.insert_one(default_zone)
        logger.info("Created default Baur au Lac zone with polygon")

@app.on_event("startup")
async def init_default_admin():
    """Create default admin account if it doesn't exist"""
    admin_email = "admin@bauraulac.ch"
    existing = await db.users.find_one({"email": admin_email})
    if not existing:
        # Default admin credentials
        # Email: admin@bauraulac.ch
        # Password: Zurich2025
        hashed_password = bcrypt.hashpw("Zurich2025".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        admin_doc = {
            "user_id": "admin_bauraulac_001",
            "email": admin_email,
            "name": "Baur au Lac Admin",
            "role": "admin",
            "password_hash": hashed_password,
            "picture": None,
            "created_at": datetime.now(timezone.utc)
        }
        await db.users.insert_one(admin_doc)
        logger.info("Created default admin account: admin@bauraulac.ch / Zurich2025")

@app.on_event("startup")
async def init_default_hotel():
    """Create default hotel account if it doesn't exist"""
    hotel_email = "taxi@bauraulac.ch"
    hotel_user_id = "hotel_bauraulac_001"
    default_zone_id = "zone_ea7daf1a6ab0"
    
    # Remove old hotel accounts if exists
    await db.users.delete_one({"email": "hotel@bauraulac.ch"})
    await db.users.delete_one({"email": hotel_email})
    
    # Default hotel credentials
    # Email: taxi@bauraulac.ch
    # Password: Marguita2025
    hashed_password = bcrypt.hashpw("Marguita2025".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    hotel_doc = {
        "user_id": hotel_user_id,
        "email": hotel_email,
        "name": "Baur au Lac Hotel",
        "role": "hotel",
        "password_hash": hashed_password,
        "picture": None,
        "created_at": datetime.now(timezone.utc),
        "zone_id": default_zone_id
    }
    await db.users.insert_one(hotel_doc)
    
    # Update the zone to link to this hotel user
    await db.hotel_zones.update_one(
        {"zone_id": default_zone_id},
        {"$set": {"hotel_user_id": hotel_user_id}}
    )
    
    logger.info("Created default hotel account: taxi@bauraulac.ch / Marguita2025")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
