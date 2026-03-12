from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from django.contrib.auth.models import AnonymousUser
from .Jwt_handler import verify_access_token

class TokenUser(AnonymousUser):
    def __init__(self, payload):
        self.id = payload.get("manager_id")
        self.role = payload.get("role", "guest")

    @property
    def is_authenticated(self):
        return True

class JWTAuthentication(BaseAuthentication):
    def authenticate(self, request):
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            return None

        token = auth.split(" ", 1)[1].strip()
        payload = verify_access_token(token)
        return (TokenUser(payload), None)
