import json
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus.milvus_client import milvus_client

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, \
    ITEM_NAME_EXTRACT_TEMPLATE
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search

from utils.mongo_history_util import get_recent_messages, save_chat_message, update_message_item_names


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 1.参数校验
        session_id, original_query = self._step_1_validate_param(state)
        logger.info(f"步骤1：参数校验通过")

        # 2.获取历史记忆
        history = get_recent_messages(session_id)
        logger.info(f"步骤2：获取到 {len(history)} 条历史消息")
        state["history"] = history

        # 3.保存用户信息(存原始问题)
        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"步骤3：用户消息已初始保存, ID: {message_id}")

        # 4.LLM提取:rewritten_query（改写问题）、item_name（模型识别到的设备主体）
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names")
        rewritten_query = extract_res.get("rewritten_query", original_query)
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names
        print("extract_res:", extract_res)

        # 5.向量搜索（搜索知识库）、6.搜索结果对齐（整理）
        align_result = {}
        if item_names:
            query_results = self._step_5_vectorize_and_query(item_names)
            print(f"milvus根据提取到的item_name的匹配结果")
            for query_result in query_results:
                print(query_result)
            align_result = self._step_6_align_item_names(query_results)  #返回一个相似度最高的结果
            # print(f"milvus根据提取到的item_name的匹配结果，并且经过了分数整理，高分：confirmed_item_names，低分：options")
            # names_list = align_result["confirmed_item_names"]
            # options = align_result["options"]
            # print(names_list)
            # print(options)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 7.状态state状态整理（更新state数据），将第六步结果记录到state
        state = self._step_7_check_confirmation(state, align_result, history)

        # 8.保存历史会话(将识别完的问题保存到历史记录)
        self._step_8_write_history(state, session_id, rewritten_query, message_id)
        return state

    def _step_1_validate_param(self, state: QueryGraphState) -> Tuple[str, str]:

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数original_query缺失")

        return session_id, original_query

    def _step_4_extract_info(self, original_query, history) -> Dict:
        print("step4:模型提取意图")

        ai_client = get_llm_client(json_model=True)
        # 拼接上下文：history+original_query
        history_text = ""
        for msg in history:
            role = msg.get("role")
            content = msg.get("text")
            history_text += f"{role}: {content}\n"
        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text,
            query=original_query
        )
        messages = [
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = ai_client.invoke(messages)
        response_content = response.content
        result = json.loads(response_content)
        if "item_names" not in result:
            result["item_names"] = []
            # 确保返回结果包含rewritten_query字段，无则复用原始查询
        if "rewritten_query" not in result:
            result["rewritten_query"] = original_query
        if result["item_names"]:
            result["item_names"] = [
                name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                for name in result["item_names"]
            ]

        return result

    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        print("step5:向量化并检索，返回与item_name对应的向量库中相似的名称列表")

        results: List[Dict] = []  #大模型识别的可能的名称：milvus的匹配结果集合

        # milvus客户端，集合名称
        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection

        # 条件向量化
        embeddings = generate_embeddings(item_names)

        # 相似性匹配
        for i in range(len(item_names)):
            dense_vector = embeddings.get("dense")[i]
            sparse_vector = embeddings.get("sparse")[i]
            reqs = create_hybrid_search_requests(dense_vector=dense_vector, sparse_vector=sparse_vector,limit=5)
            search_res = hybrid_search(
                client=milvus_client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8,0.2),
                norm_score=True,
                output_fields=["item_name"]
            )
            # 结果处理
            matches = []
            if search_res and len(search_res) > 0:
                # print("此处是意图识别向量搜索结果：")
                # print(search_res)
                for hit in search_res[0]:
                    matches.append({
                        "item_name": hit.get("item_name"),
                        "score": hit.get("distance")
                    })
                    pass
            results.append({
                "extracted_name": item_names[i],
                "matches":matches   #相似性匹配结果列表（milvus匹配对象）
            })

        return results


    # 对结果进行分段处理
    def _step_6_align_item_names(self, query_results:List[Dict]) -> dict:
        print("step6:对齐结果，高份结果低份结果整合")
        confirmed_item_names : List[Dict] = []
        options:List[Dict] = []

        for res in query_results:
            extracted_name = (res.get("extracted_name","") or "").strip()  #模型识别的商品名称
            matches = res.get("matches",[]) or []
            if not matches:
                continue
            high_results = [m for m in matches if m.get("score",0) >= 0.8]
            mid_results = [m for m in matches if 0.8 > m.get("score",0) >= 0.6]
            #有高分则取高分，没有高分取中分段
            #########################################################################
            # 高于0.8：高度匹配，直接给item_name
            # 只有一条结果且高于0.8
            if len(high_results) == 1:
                confirmed_item_names.append(high_results[0].get("item_name"))
                continue
            if len(high_results) > 1:
                picked = None
                if extracted_name:
                    for hr in high_results:
                        if hr.get("item_name") == extracted_name:
                            picked = hr
                            break
                if not picked:
                    picked = high_results[0]
                # 确认名称
                confirmed_item_names.append(picked.get("item_name"))
                continue
            ##########################################################################################
            # 高于0.6低于0.8：可能匹配，有可选项options,反问用户
            if len(mid_results) > 0:
                for mr in mid_results[:5]:
                    options.append(mr.get("item_name"))
            ###########################################################################################
            # 低于0.6：无匹配结果,不处理，都是空值


        return{
            "confirmed_item_names":list(set(confirmed_item_names)),  # 确认后商品名称（>0.8分）
            "options":list(set(options))     # 低分商品名称(>0.6),用set是为了去重
        }

    def _step_7_check_confirmation(self, state, align_result:Dict, history):
        print("step7:状态state信息，根据第六步高分低分对齐结果整理")
        confirmed = align_result.get("confirmed_item_names")
        options = align_result.get("options")
        # 1.反问用户（0.6-0.8） 2.直接返回“非常抱歉，没有要找的”（<0.6）
        # 只有item_name里有值才进入下一个节点，否则直接返回
        # 1.有命中（>0.8）
        if confirmed:
            # 更新MongoDB会话记录，将命中结果更新到与本次命中有关的所有会话中（更新所有历史记录的item_name)
            ids_to_update = []
            for msg in history:
                if not msg.get("item_names"):
                    mid = msg.get("_id")
                    if mid:
                        ids_to_update.append(str(mid))
            if ids_to_update:
                update_message_item_names(ids_to_update,confirmed)

            # 封装结果
            state["item_names"] = confirmed
            state["answer"] = ""
            return state
        # 2.有备选
        # 封装结果
        if options:
            state["item_names"] = []
            options_str = "、".join(options)
            state["answer"] = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
            return state
        # 3.没命中（<0.6）
        # 封装结果
        if not confirmed and not options:
            state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
            state["item_names"] = []
        return state

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        
        print("step8:写入历史会话，更新")

        # 若会话状态中有助手答案（分支B/C），写入助手消息到历史
        if state.get("answer"):
            save_chat_message(
                session_id=session_id,  # 会话ID，关联所属会话
                role="assistant",  # 消息角色：助手
                text=state["answer"],  # 消息内容：向用户确认的提示语/无结果提示语
                rewritten_query="",  # 助手消息无需改写查询，设为空
                item_names=state.get("item_names", [])  # 关联的商品名列表（分支B/C均为空）
            )


        # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名）
        save_chat_message(
            session_id=session_id,  # 会话ID，关联所属会话
            role="user",  # 消息角色：用户
            text=state["original_query"],  # 消息内容：用户原始查询
            rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
            item_names=state.get("item_names", []),  # 补充关联的商品名列表
            message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
        )

        # 返回最终会话状态，供下游节点使用
        return state



if __name__ == "__main__":

    # 初始化图状态
    init_state = {
        "original_query": "兄弟HAK180烫金机这个玩应咋搞的啊？",
        "session_id":123
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    # json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logger.info(result)