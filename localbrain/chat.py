"""Explicit chat commands and short pages; no learned tool-selection claims."""
import json

from .assistant import answer, ingest, research

HELP = """ตอนนี้ช่วยค้นเอกสาร ค้นเว็บ เก็บความรู้ และเรียกเครื่องมือได้
/help หรือ ทำอะไรได้บ้าง — ดูคำสั่ง
/calc (8+4)*2 — คำนวณ
/search VRAM — ค้นเฉพาะคลังในเครื่อง
/web Python tutorial — ค้นเว็บและเก็บไม่เกิน 3 หน้า
/ingest data/example.txt — เพิ่มไฟล์ โฟลเดอร์ หรือ URL เข้าคลัง
/time — วันเวลาปัจจุบัน
/stats — จำนวนเอกสารและส่วนข้อความ
/tools — รายชื่อเครื่องมือ
/tool ชื่อเครื่องมือ {"argument":"value"} — เรียกเครื่องมือด้วย JSON
/more — อ่านหน้าถัดไปของคำตอบยาว
/exit — ออกจากแชท
การเลือกคำสั่งเป็นกติกา Python; โมเดลฝึกเองยังเป็นรุ่นทดลอง"""


def response(text, mode="command", sources=None):
    return {"mode": mode, "answer": text, "sources": sources or []}


class Chat:
    def __init__(self, tools, run=None, use_web=False, device="auto", model_only=False):
        self.tools = tools
        self.run = run
        self.use_web = use_web
        self.device = device
        self.predictor = None
        if model_only:
            if not run:
                raise ValueError("Model-only chat requires --run")
            from .inference import Predictor
            self.predictor = Predictor(run, device)
        self.last = None
        self.offset = 0

    def page(self):
        if self.last is None:
            return response("ยังไม่มีคำตอบก่อนหน้า")
        text = self.last["answer"]
        if self.offset >= len(text):
            return response("แสดงคำตอบครบแล้ว")
        block = "".join(text[self.offset:].splitlines(keepends=True)[:12])[:900]
        self.offset += len(block)
        result = {**self.last, "answer": block.strip()}
        if self.offset < len(text):
            result["note"] = (result.get("note", "") + " พิมพ์ /more เพื่ออ่านต่อ").strip()
        return result

    def handle(self, query):
        query = query.strip()
        normalized = query.casefold().rstrip("!?？. ")
        if normalized == "/more":
            return self.page()
        help_queries = {"help", "/help", "ทำอะไรได้บ้าง", "คุณทำอะไรได้บ้าง",
                        "ช่วยอะไรได้บ้าง", "มีคำสั่งอะไรบ้าง", "คำสั่ง", "what can you do"}
        if self.predictor is not None:
            help_queries = {"/help"}
        if normalized in help_queries:
            result = response(HELP, "help")
        elif query.startswith("/"):
            pieces = query.split(maxsplit=1)
            command = pieces[0].casefold()
            argument = pieces[1].strip() if len(pieces) == 2 else ""
            if command in {"/calc", "/search", "/web", "/ingest", "/tool"} and not argument:
                raise ValueError("คำสั่งนี้ต้องมีข้อมูลต่อท้าย ดูตัวอย่างด้วย /help")
            if command == "/calc":
                result = response(str(self.tools.call("calculator", {"expression": argument})))
            elif command == "/time":
                result = response(self.tools.call("get_time", {}))
            elif command == "/stats":
                stats = self.tools.knowledge.stats()
                result = response(f"เอกสาร {stats['documents']} รายการ; ส่วนข้อความ {stats['chunks']} ส่วน")
            elif command == "/tools":
                result = response("เครื่องมือ: " + ", ".join(self.tools.describe())
                                  + "\nใช้ /tool ชื่อเครื่องมือ JSON เพื่อเรียก")
            elif command == "/search":
                hits = self.tools.knowledge.search(argument, 3)
                if not hits:
                    result = response("ยังไม่พบข้อมูลในคลัง")
                else:
                    result = response(
                        "\n\n".join(f"[{i}] {hit['body']}" for i, hit in enumerate(hits, 1)),
                        "retrieved_excerpts",
                        [{"id": i, "title": hit["title"], "source": hit["source"]}
                         for i, hit in enumerate(hits, 1)])
                    result["note"] = "ข้อความคัดจากเอกสาร ยังไม่ได้สรุปหรือตรวจสอบข้อเท็จจริง"
            elif command in {"/web", "/ingest"}:
                values = (research(self.tools.knowledge, argument) if command == "/web"
                          else ingest(self.tools.knowledge, argument))
                lines = ["เก็บความรู้โดยไม่เปลี่ยนน้ำหนักโมเดล:"]
                for item in values:
                    if "error" in item:
                        lines.append(f"ไม่สำเร็จ: {item['source']} — {item['error']}")
                    else:
                        status = "อัปเดตแล้ว" if item["changed"] else "ตรวจแล้ว เนื้อหาเดิม"
                        lines.append(f"{status}: {item['source']}")
                if not values:
                    lines.append("ไม่พบรายการที่เก็บได้")
                result = response("\n".join(lines))
            elif command == "/tool":
                parts = argument.split(maxsplit=1)
                kwargs = json.loads(parts[1]) if len(parts) == 2 else {}
                value = self.tools.call(parts[0], kwargs)
                result = response(value if isinstance(value, str)
                                  else json.dumps(value, ensure_ascii=False, indent=2))
            else:
                raise ValueError("ไม่รู้จักคำสั่งนี้ พิมพ์ /help เพื่อดูคำสั่ง")
        elif self.predictor is not None:
            generated = self.predictor.reply(query)
            result = response(generated or "[โมเดลจบคำตอบโดยไม่สร้างข้อความ]", "learned_model")
            result["note"] = "สร้างจากน้ำหนักโมเดลทดลองโดยตรง อาจผิดหรือแต่งข้อมูล"
        else:
            result = answer(self.tools.knowledge, query, self.run, self.use_web, self.device)
        self.last = result
        self.offset = 0
        return self.page()
