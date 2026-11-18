import re
from typing import List, Dict

from mineru.backend.analysis.pdf.text import lines

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
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

        # 5.打印日志
        self._step_5_print_stats(lines_count, sections)

        # 6.备份
        self._step_6_backup(state, sections)
        state["chunks"] = None
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





        return sections

    def _step_5_print_stats(self, lines_count, sections):
        print("node_document_split:步骤5：打印日志")
        pass

    def _step_6_backup(self, state, sections):
        print("node_document_split:步骤6：备份")
        pass

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
