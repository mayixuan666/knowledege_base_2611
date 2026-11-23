import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #允许所有的源
    allow_credentials=True,  # 允许客户端传递cookie
    allow_methods=["*"], # get，post都行
    allow_headers=["*"]   # 请求头所有行都行
)

@app.get("/api/data")
async def get_data():
    print("请求后端数据服务")
    return {"message":"这是后端数据","state":"success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)