from pymilvus import connections, Collection

# 1. 先连接（host/port 按你项目配置填）
connections.connect(
    alias="default",
    host="127.0.0.1",   # 或你的 Milvus IP，比如 192.168.44.132
    port="19530"        # Milvus 默认端口
)

# 2. 再拿 Collection
col = Collection("kb_chunks")
col.load()

res = col.query(
    expr="file_title like '%华为擎云B530%'",
    output_fields=["chunk_id", "content"],
    limit=3
)

for r in res:
    print(f"chunk_id: {r['chunk_id']}")
    print(f"Contains http://: {'http://' in r['content']}")
    print(f"Contains images/: {'images/' in r['content']}")
    print(r['content'][:300])
    print("-" * 50)