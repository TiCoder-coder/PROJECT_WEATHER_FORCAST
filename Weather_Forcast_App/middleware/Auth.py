#---------------------------------------------- DINH NGHIA CAC HAM DUNG DE PHAN QUYEN LUC CHAY---------------------------------------------------------

from rest_framework.exceptions import PermissionDenied, AuthenticationFailed
from .Jwt_handler import verify_access_token
import jwt
from django.conf import settings
from rest_framework.exceptions import AuthenticationFailed

# Dinh nghia mot ham dung de phan quyen chi co manager moi duoc su dung cac ham nay
def require_manager(user):
    if not user:
        raise PermissionDenied("Authentication required")
    role = getattr(user, "role", None)
    if not role or role.lower() != "manager":
        raise PermissionDenied("You must be manager to perform this actions")

# Dinh nghia mot ham dung de phan quyen ra staff hay manager deu su dung duoc ham nay
def require_staff_or_manager(user):
    if not user:
        raise PermissionDenied("Authentication required.")
    role = getattr(user, "role", None)
    if not role or role.lower() not in ["manager", "staff", "admin"]:
        raise PermissionDenied("You must be staff or manager to perform this action.")

# Dinh nghia mot ham dung de lay thong tin user va kiem tra
def get_user_from_token(request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise AuthenticationFailed("Authorization header missing or invalid.")
    token = auth_header.split(" ", 1)[1].strip()

    payload = verify_access_token(token)

    class UserObj:
        pass

    user = UserObj()
    user.id = payload.get("manager_id")
    user.role = payload.get("role", "guest")
    return user

# Tao mot ham de dinh nghia cac phuong thuc chi co manager va admin duoc sai
def require_manager_or_admin(user):
    if not user or not getattr(user, "role", None):
        raise PermissionDenied("Authentication required.")
    if user.role.lower() not in ["manager", "admin"]:
        raise PermissionDenied("You do not have permission to perform this action.")