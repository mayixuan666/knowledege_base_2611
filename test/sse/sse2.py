import asyncio

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #允许所有的源
    allow_credentials=True,  # 允许客户端传递cookie
    allow_methods=["*"], # get，post都行
    allow_headers=["*"]   # 请求头所有行都行
)

@app.get("/stream/{session_id}")
async def stream_by_session(session_id: str):

    async def event_generator():
        for i in range(10):
            yield f"data:会话id{session_id}:这是第{i}条消息\n\n"
            await asyncio.sleep(1)
        yield f"data:[END]\n\n"

    async def error_event_generator():
        # 不要 return，而是 yield 一个符合 SSE 格式的错误消息
        yield "data: 无效会话id\n\n"
        yield f"data: [END]\n\n"  # 结束标记

    if session_id == "123":
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    else:
        return StreamingResponse(
            error_event_generator(),
            media_type="text/event-stream"
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.44.1", port=8001)
