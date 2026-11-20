import json

import pymongo

mongo_client = pymongo.MongoClient("mongodb://192.168.44.132:27017")

db = mongo_client["test"]
# 创建集合
# db.create_collection("classes")
# 插入数据
# db["classes"].insert_one({"name": "three", "age": 2})
# 查询数据
find_result = db["classes"].find()
for item in find_result:
    print(item)
    print(item["name"])