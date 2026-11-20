from  utils.embedding_utils import generate_embeddings,get_bge_m3_ef

model = get_bge_m3_ef()
# 测试1：直接用bge-m3客户端调用向量化方法，反回带有稠密向量和稀疏向量
embeddings = model.encode_documents(["你好"])
print(embeddings)

# 测试2：用封装的generate_embeddings方法，返回带有稠密向量和稀疏向量
embeddings = generate_embeddings(["你好"])
print(embeddings["dense"][0])
print(embeddings["sparse"][0])