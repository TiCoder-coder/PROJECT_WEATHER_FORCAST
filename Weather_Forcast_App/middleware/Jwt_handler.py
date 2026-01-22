import jwt
from datetime import datetime, timedelta, timezone
from decouple import config
from rest_framework.exceptions import AuthenticationFailed, ValidationError
from django.conf import settings
from Weather_Forcast_App.db_connection import get_database, create_index_safe

JWT_ALGORITHM = config("JWT_ALGORITHM", default="HS256")
JWT_SECRET = config("JWT_SECRET")
ISSUER = config("JWT_ISSUER", default="weather_api")
AUDIENCE = config("JWT_AUDIENCE", default="weather_web")

ACCESS_TOKEN_EXPIRE_HOURS = int(config("ACCESS_TOKEN_EXPIRE_HOURS", default=3))
REFRESH_TOKEN_EXPIRE_DAYS = int(config("REFRESH_TOKEN_EXPIRE_DAYS", default=1))

db = get_database()
revoked_tokens = db["revoked_tokens"]
create_index_safe(revoked_tokens, "token", unique=True)

SECRET = config("JWT_SECRET", default=settings.SECRET_KEY)

def create_access_token(payload: dict) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

    body = {
        "manager_id": str(payload.get("manager_id")),
        "role": str(payload.get("role", "guest")).lower(),
        "iat": now,
        "exp": exp,
        "iss": ISSUER,
        "aud": AUDIENCE
    }
    token = jwt.encode(body, SECRET, algorithm=JWT_ALGORITHM)
    return token if isinstance(token, str) else token.decode("utf-8")

def is_token_revoked(token: str) -> bool:
    return revoked_tokens.find_one({"token": token}) is not None

def verify_access_token(token: str) -> dict:
    try:
        decoded = jwt.decode(
            token,
            SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer=ISSUER,
            audience=AUDIENCE
        )
        if is_token_revoked(token):
            raise AuthenticationFailed("Token has been revoked")
        return decoded
    except jwt.ExpiredSignatureError:
        raise AuthenticationFailed("Token expired")
    except jwt.InvalidTokenError:
        raise AuthenticationFailed("Invalid token")

def revoke_token(token: str):
    revoked_tokens.insert_one({"token": token, "revoked_at": datetime.now(timezone.utc)})

# Tao ra mot refresh token --- Dung de lay lai mot access token ma khong can login lai
def create_refresh_token(payload: dict) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload["exp"] = exp

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token if isinstance(token, str) else token.decode("utf-8")

# Ham dung de giai ma token
def decode_token(token: str):
    try:
        # Giai ma
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValidationError("Token expired")
    except jwt.InvalidTokenError:
        raise ValidationError("Invalid token")


# Kiem tra xem token co bi thu hoi hay chua
def is_token_revoked(token):
    return revoked_tokens.find_one({"token": token}) is not None

# Giai ma refresh token ---- thu hoi token cu va tao token moi
def refresh_access_token(refresh_token: str):
    decoded = decode_token(refresh_token)                                   # Giai ma token
    revoke_token(refresh_token)                                             # Thu hoi token
    return create_access_token({                                            # Tao token moi
        "manager_id": decoded.get("manager_id"),
        "role": decoded.get("role", "manager")
    })
