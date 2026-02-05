import jwt
import secrets
from datetime import datetime, timedelta, timezone
from decouple import config
from rest_framework.exceptions import AuthenticationFailed, ValidationError
from django.conf import settings
from pymongo.errors import DuplicateKeyError
from Weather_Forcast_App.db_connection import get_database, create_index_safe

JWT_ALGORITHM = config("JWT_ALGORITHM", default="HS256")
JWT_SECRET = config("JWT_SECRET", default=settings.SECRET_KEY)
ISSUER = config("JWT_ISSUER", default="weather_api")
AUDIENCE = config("JWT_AUDIENCE", default="weather_web")
ACCESS_TOKEN_EXPIRE_HOURS = int(config("ACCESS_TOKEN_EXPIRE_HOURS", default=3))
REFRESH_TOKEN_EXPIRE_DAYS = int(config("REFRESH_TOKEN_EXPIRE_DAYS", default=7))
CHECK_ACCESS_REVOKE = config("CHECK_ACCESS_REVOKE", default="true").lower() == "true"

db = get_database()
revoked_tokens = db["revoked_tokens"]
create_index_safe(revoked_tokens, "jti", unique=True)
create_index_safe(revoked_tokens, "expiresAt", expireAfterSeconds=0)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _to_ts(dt: datetime) -> int:
    return int(dt.timestamp())

def is_token_revoked_jti(jti: str) -> bool:
    if not jti:
        return False
    return revoked_tokens.find_one({"jti": jti}, {"_id": 1}) is not None

def revoke_jti(jti: str, expires_at: datetime, token_type: str = "unknown") -> None:
    """
    Revoke theo jti (nhẹ hơn lưu token).
    Upsert để không bị DuplicateKeyError.
    """
    if not jti:
        return
    revoked_tokens.update_one(
        {"jti": jti},
        {
            "$setOnInsert": {
                "jti": jti,
                "type": token_type,
                "revokedAt": _now_utc(),
                "expiresAt": expires_at,
            }
        },
        upsert=True
    )

def _build_token(payload: dict, token_type: str, exp: datetime) -> str:
    now = _now_utc()
    body = {
        "type": token_type,
        "jti": secrets.token_hex(16),  # unique token id
        "manager_id": str(payload.get("manager_id")),
        "role": str(payload.get("role", "guest")).lower(),
        "iat": _to_ts(now),
        "exp": _to_ts(exp),
        "iss": ISSUER,
        "aud": AUDIENCE,
    }
    token = jwt.encode(body, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token if isinstance(token, str) else token.decode("utf-8")


def create_access_token(payload: dict) -> str:
    exp = _now_utc() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return _build_token(payload, token_type="access", exp=exp)

def create_refresh_token(payload: dict) -> str:
    exp = _now_utc() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return _build_token(payload, token_type="refresh", exp=exp)

def verify_access_token(token: str) -> dict:
    try:
        decoded = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer=ISSUER,
            audience=AUDIENCE,
        )
        if decoded.get("type") != "access":
            raise AuthenticationFailed("Invalid token type")

        if CHECK_ACCESS_REVOKE and is_token_revoked_jti(decoded.get("jti")):
            raise AuthenticationFailed("Token has been revoked")

        return decoded

    except jwt.ExpiredSignatureError:
        raise AuthenticationFailed("Token expired")
    except jwt.InvalidTokenError:
        raise AuthenticationFailed("Invalid token")

def decode_refresh_token(token: str) -> dict:
    """
    Decode refresh: luôn check revoke vì refresh là cửa ngõ lấy access mới.
    """
    try:
        decoded = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer=ISSUER,
            audience=AUDIENCE,
        )
        if decoded.get("type") != "refresh":
            raise ValidationError("Invalid token type")

        if is_token_revoked_jti(decoded.get("jti")):
            raise ValidationError("Refresh token revoked")

        return decoded

    except jwt.ExpiredSignatureError:
        raise ValidationError("Token expired")
    except jwt.InvalidTokenError:
        raise ValidationError("Invalid token")

def revoke_token(token: str, token_type_hint: str = "unknown") -> None:
    """
    Revoke token bất kỳ: decode lấy jti + exp, rồi lưu TTL record.
    NOTE: verify_exp=False để vẫn revoke được token đã hết hạn (nếu cần).
    """
    try:
        decoded = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer=ISSUER,
            audience=AUDIENCE,
            options={"verify_exp": False},
        )
        jti = decoded.get("jti")
        exp_ts = decoded.get("exp")
        if not exp_ts:
            return
        expires_at = datetime.fromtimestamp(int(exp_ts), tz=timezone.utc)
        token_type = decoded.get("type", token_type_hint)
        revoke_jti(jti, expires_at, token_type=token_type)
    except jwt.InvalidTokenError:
        return

def refresh_access_token(refresh_token: str) -> str:
    decoded = decode_refresh_token(refresh_token)

    exp_ts = decoded.get("exp")
    expires_at = datetime.fromtimestamp(int(exp_ts), tz=timezone.utc)
    revoke_jti(decoded.get("jti"), expires_at, token_type="refresh")

    return create_access_token(
        {"manager_id": decoded.get("manager_id"), "role": decoded.get("role", "manager")}
    )
