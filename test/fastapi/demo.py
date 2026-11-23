from fastapi import FastAPI
from langsmith import BaseModel
from starlette.responses import JSONResponse, FileResponse, RedirectResponse

# 创建一个 FastAPI 应用实例
# 用uv run uvicorn demo:app
app = FastAPI()

class User(BaseModel):
    username: str
    email: str
    age: int

@app.get("/new-path")
async def new_path():
    return {"message": "这是新接口"}

@app.get("/old-path")
async def redirect_old_path():
    # 重定向到 /new-path，状态码 307 表示临时重定向
    return RedirectResponse(url="/new-path", status_code=307)


@app.get("/download/excel")
async def download_excel():
    excel_path = "D:/test.xls"
    # 返回文件并指定下载文件名
    return FileResponse(
        path=excel_path,
        filename="月度报表.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.get("/api/user")
async def get_user():
    # 等价于直接 return {"name": "张三", "age": 20}（FastAPI 自动转 JSONResponse）
    return JSONResponse(
        content={"name": "张三", "age": 20},
        status_code=200,  # 可选，默认 200
        headers={"X-Custom-Header": "custom-value"}  # 可选，自定义响应头,可用于带着账户名密码
    )

@app.post("/user")
async def create_user(user: User):
    return {"username": user.username, "email": user.email, "age": user.age}

@app.get("/read_root")
def read_root():
    print("read root后端接口被访问")
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int,a:str):
    print("read_item后端有参数被访问")
    return {"item_id": item_id, "a": a}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.44.1", port=8001)