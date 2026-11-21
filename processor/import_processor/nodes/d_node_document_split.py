import json
import re
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from mineru.backend.analysis.pdf.text import lines

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.import_config import get_config
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):

        # 1.参数处理
        content, file_title = self._step_1_get_inputs(state)
        # 2.标题切（初切）
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)
        # for section in sections:
        #     print(f"{section['title']}")
        #     print(f"{section['content']}")
        #     print("=====================================================================================================")
        # print(f"lines_count:{lines_count}")
        # print(f"title_count:{title_count}")
        # 3.无标题兜底（默认标题）
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # 4.块精细化处理（长切短合）
        sections = self._step_4_refine_chunks(sections)
        for section in sections:
            print(f"{section['title']}")
            print(f"{section['content']}")
            print("=====================================================================================================")
        print(f"lines_count:{lines_count}")
        print(f"title_count:{title_count}")
        # 5.打印日志
        self._step_5_print_stats(lines_count, sections)

        # 6.备份
        self._step_6_backup(state, sections)
        state["chunks"] = sections
        return state

    def _step_1_get_inputs(self, state):
        print("node_document_split:步骤1：参数处理")
        content = state.get("md_content")
        if not content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        # 换行符处理
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        return content, file_title

    def _step_2_split_by_titles(self, content, file_title):
        print("node_document_split:步骤2：标题切分（初切）")
        sections:List[Dict[str,str]] = []
        current_lines = []
        title_count = 0
        lines = content.split("\n")
        in_code_block = False  #记录#是否在代码块中
        current_title = ""

        # 独立封装刷新到下一个块的代码
        def _flush_section():
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                # 每段使用 \n换行区分
                "content": "\n".join(current_lines),
                "file_title": file_title,
            })

        # 切块逻辑(标题切),用#切分
        title_pattern = r'^\s*#{1,6}\s+.+'

        for line in lines:
            stripped_line = line.strip()

            if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue
            if(not in_code_block) and (re.match(title_pattern, line)):
                _flush_section()  #将之前存好的块中的内容封装成一个section块
                # 遇到下一个标题了，进行标题处理
                current_title = stripped_line  #换标题
                current_lines = [current_title] #把这段的标题放入到这段的代码块中
                title_count += 1
                pass
            else:
                current_lines.append(line)

        _flush_section() #处理最后一个块
        return sections, title_count, len(lines)

    def _step_3_handle_no_title(self, content, sections, title_count, file_title):
        print("node_document_split:步骤3：无标题兜底（默认标题）")
        if title_count == 0:
            return [{"title": "无标题", "content": content, "file_title": file_title}]
        return sections

    def _step_4_refine_chunks(self, sections):
        print("node_document_split:步骤4：块精细化处理（长切短合）")
        refined_split = []
        for sec in sections:
            refined_split.extend(self.split_long_section(sec))

        # 短合列表
        final_sections = self.merge_short_sections(refined_split)

        # 安全性代码
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        return final_sections
    # 步骤四方法一：长切
    def split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        content = section.get("content","")
        content_len = len(content)
        if content_len <= get_config().max_content_length:
            return [section]

        # 提取章节标题，用于组装子Chunk前缀（保留标题上下文）
        title = section.get("title", "")
        prefix = f"{title}\n\n" if title else ""
        available_len = self.config.max_content_length - len(prefix)  #切分标准

        # 去重标题,把content里的title去掉
        # 防止body在执行的时候抛异常使content内容遗失
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title)+len(title):].lstrip()
        # 切分器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]
        )
        # 切分结果
        sub_sections = []
        for index, chunk in enumerate(splitter.split_text(body), start=1):
            text = chunk.strip()
            if not text:
                continue

            # 组装子Chunk完整内容 = 标题前缀 + 切分后的正文
            full_text = (prefix + text).strip()

            sub_sections.append({
                "title": f"{title}-{index}" if title else f"chunk-{index}",  # 子Chunk标题（带序号）
                "content": full_text,  # 切分后的完整内容
                "parent_title": title,  # 父章节标题（用于后续合并）
                "part": index,  # 子Chunk序号
                "file_title": section.get("file_title"),  # 所属文件标题
            })
        return sub_sections
    # 步骤四方法二：短合
    def merge_short_sections(self,sections):
        print("node_document_split:步骤四方法二短合")

        # 小于阈值且和前一个同父标题
        """
               【辅助函数】过短章节合并（减少碎片化，提升检索效果）
               核心规则：仅合并「同父标题」且「当前块长度不足阈值」的相邻Chunk，避免跨章节合并
               :param sections: 待合并的Chunk列表（通常是_split_long_section切分后的结果）
               :return: 合并后的Chunk列表，长度适中，保留元信息
               """
        # 边界处理：空列表直接返回，避免后续索引报错
        if not sections:
            self.logger.debug("待合并Chunk列表为空，直接返回")
            return []

        merged_sections = []  # 最终合并结果
        current_chunk = None  # 迭代累加器：保存当前待合并的Chunk

        for sec in sections:
            # 初始化：第一个Chunk直接作为当前待合并块
            if current_chunk is None:
                current_chunk = sec
                continue

            # 合并条件：1.当前块长度不足阈值 2.与下一块同父标题（同属一个原章节）
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

            if is_current_short and is_same_parent:
                # 合并前清理：去掉下一块开头重复的父标题，避免内容冗余
                parent_title = sec.get("parent_title", "")
                next_content = sec["content"]
                if parent_title and next_content.startswith(parent_title):
                    next_content = next_content[len(parent_title):].lstrip()
                # 合并内容：空行分隔，保证格式整洁
                current_chunk["content"] += "\n\n" + next_content
                # 更新子Chunk序号：保留最新序号，便于溯源
                if "part" in sec:
                    current_chunk["part"] = sec["part"]
                self.logger.debug(
                    f"合并短Chunk：{current_chunk.get('parent_title')} → 累计长度{len(current_chunk['content'])}")
            else:
                # 不满足合并条件：将当前块加入结果，切换为新的待合并块
                merged_sections.append(current_chunk)
                current_chunk = sec

        # 循环结束后，将最后一个待合并块加入结果
        if current_chunk is not None:
            merged_sections.append(current_chunk)

        self.logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")
        return merged_sections



    def _step_5_print_stats(self, lines_count, sections):
        print("node_document_split:步骤5：打印日志")
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数：{lines_count}")
        self.logger.info(f"最终生成Chunk数量：{chunk_num}")

    def _step_6_backup(self, state, sections):
        print("node_document_split:步骤6：备份")
        path = Path(state.get("md_path")).parent / f'{state.get("file_title")}_new_chunks.json'
        with open(path,"w",encoding="utf-8") as f:
            json.dump(
                sections,
                f,
                ensure_ascii=False,
                indent=2
            )



if __name__ == "__main__":
    node = NodeDocumentSplit()
    with open(r"E:\AI Agent\g0525\output\H3C LA2608室内无线网关 用户手册-6W100-整本手册\H3C LA2608室内无线网关 用户手册-6W100-整本手册_new.md","r",encoding="utf-8") as f:
        md_content = f.read()
    init_state = {
        "md_path":r"E:\AI Agent\g0525\output\H3C LA2608室内无线网关 用户手册-6W100-整本手册\H3C LA2608室内无线网关 用户手册-6W100-整本手册_new.md",
        "md_content": md_content,
        "file_title":"H3C LA2608室内无线网关 用户手册-6W100-整本手册_new"
    }
    process = node.process(init_state)
    print(f"切分节点执行流程{process}")
