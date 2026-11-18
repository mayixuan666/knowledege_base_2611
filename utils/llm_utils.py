from langchain_openai import ChatOpenAI

from config.lm_config import lm_config
# 用字典类型把多个模型封装到一个里边
_llm_client_cache = {}

def get_llm_client(model:str|None=None,json_model:bool=False):
    m = model or lm_config.llm_model
    key = (m,json_model)

    if key in _llm_client_cache:
        return _llm_client_cache[key]

    # 以JSON格式返回
    model_kwargs = {}
    if json_model:
        model_kwargs['response_format'] = {"type":"json_object"}
    # 返回模型
    client = ChatOpenAI(
        model=m,
        temperature=lm_config.llm_temperature,
        base_url=lm_config.base_url,
        api_key=lm_config.api_key,
        model_kwargs=model_kwargs
    )
    # 根据传进来的模型名即key（model）找到模型，返回具体模型对象value（client）
    _llm_client_cache[key] = client
    return client
if __name__ == '__main__':
    client = get_llm_client()
    response = client.invoke("你好")
    print(response)
