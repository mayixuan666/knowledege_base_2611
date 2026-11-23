from pathlib import Path

from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

# 保留关键字
BASE_DIR = Path(__file__).parent
print(BASE_DIR)
static_dir = BASE_DIR / "static"
print(static_dir)

app = FastAPI()

# 挂在前端资源
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0:内+外网ip
    uvicorn.run(app, host="127.0.0.1", port=8000)