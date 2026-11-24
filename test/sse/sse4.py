import asyncio

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #允许所有的源
    allow_credentials=True,  # 允许客户端传递cookie
    allow_methods=["*"], # get，post都行
    allow_headers=["*"]   # 请求头所有行都行
)

task_queues = {}

class QueryRequest(BaseModel):
    session_id: str
    query: str

# 建立队列
@app.post("/query")
async def query_by_session(query:QueryRequest,background_tasks: BackgroundTasks):
    query_query = query.query
    session_id = query.session_id
    background_tasks.add_task(long_task, session_id,query_query)
    return {"message":"任务已经开始，请耐心等待","session_id":session_id}

# 调用工作流的长耗时方法
async def long_task(session_id: str,query: str):
    print(f"开始对{query}开始长耗时任务")
    queue = asyncio.Queue()
    task_queues[session_id] = queue

    for i in range(10):
        await asyncio.sleep(1)
        msg = f"这是工作流的第{i}条执行结果\n\n"
        await queue.put(msg)
    await queue.put(None)


# 找到队列
@app.get("/stream/{session_id}")
async def stream_by_session(session_id: str):
    print("我来问问出答案了吗")
    return StreamingResponse(
        event_generator(session_id),
        media_type="text/event-stream"
    )

# 封装sse实时输出结果
async def event_generator(session_id: str):
    while session_id not in task_queues:
        await asyncio.sleep(0.1)
    queue = task_queues[session_id]
    while True:
        msg = await queue.get()
        if msg is None:
            break
        yield f"data:{msg}\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

