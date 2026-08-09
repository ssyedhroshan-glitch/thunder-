from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserData(BaseModel):
    name: str
    email: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Backend!"}

@app.post("/api/user")
def create_user(user: UserData):
    # Process user logic or save to DB here
    return {
        "status": "success",
        "message": f"User {user.name} created successfully!",
        "data": user
    }

if __name__ == "__main__":
    import uvicorn
    # Host on 0.0.0.0 to accept connections across local network
    uvicorn.run(app, host="0.0.0.0", port=8000)
  
