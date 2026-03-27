import redis
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.Config import Config

class Authentication:

  security = HTTPBearer()
  config = Config()
  redis_client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)

  def create_access_token(self, data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM)
    return encoded_jwt


  def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    if self.redis_client.exists(f"blacklist:{token}"):
        raise HTTPException(status_code=401, detail="Token has been revoked")

    try:
        payload = jwt.decode(token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload

    except JWTError:
        raise HTTPException(status_code=401, detail="Token validation failed")
    
  def revoke_token(self, token: str):
        self.redis_client.setex(
            f"blacklist:{token}",
            self.config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "revoked"
        ) 

   
    