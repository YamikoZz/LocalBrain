# CLAUDE.md — คู่มือสำหรับ Claude ที่ทำงานใน repo นี้

## โปรเจกต์
**LocalBrain** = Vibe Coding AI ที่เทรนเองจากศูนย์
ผู้ใช้สั่งเป็นภาษาอังกฤษ → โมเดลเขียนโค้ด → Agent รันเทสต์ → แก้จนผ่าน
เน้นโค้ดหลายภาษา, คณิต/ฟิสิกส์แก้ด้วยการเขียนโค้ดแล้วรัน (ไม่คิดเลขในหัว)
ไม่ได้ตั้งเป้าแข่งโมเดลใหญ่ — ซื่อตรงเรื่องผลเสมอ (ผลดีก็บอก ผลแย่ก็บอก)

อ่านก่อนเริ่มงานทุกครั้ง: [STATE.md](STATE.md) (สถานะ/ตัวเลขล่าสุด) และ [ROADMAP.md](ROADMAP.md) (แผน)
คู่มืออ่านง่าย: https://github.com/YamikoZz/LocalBrain/wiki (repo แยก `LocalBrain.wiki.git`)

## กฎเหล็ก (ห้ามละเมิด)
1. **ห้ามใช้น้ำหนักโมเดลสำเร็จรูป/โอเพนซอร์ส** (Llama, Qwen, ฯลฯ) ใช้ได้เฉพาะ *ชุดข้อมูล* เทรนเองทุกขั้น
2. **ห้ามรันโค้ดจากชุดข้อมูลภายนอกบนเครื่อง** ตรวจด้วย `ast.parse` / ตัวประเมินแบบ whitelist
   (ดู `scripts/import_v4.py`) หรือรันใน sandbox เท่านั้น
3. **ห้ามใช้คำตอบของ Claude จำนวนมากเป็นข้อมูลเทรน** — สร้างข้อมูลด้วยโปรแกรมที่ตรวจคำตอบได้ หรือข้อมูลสาธารณะที่ license อนุญาต
4. **ห้ามเก็บ token / รหัสผ่าน / ข้อมูลส่วนตัว** ใน repo, commit, wiki หรือ log
5. **ไม่ยุ่งกับการเงิน** (ซื้อ units, สมัครบริการ, ผูกบัตร) — ผู้ใช้ทำเอง
6. `workspace/` และโฟลเดอร์ชุดข้อมูลใหญ่ **ไม่ขึ้น git** (โมเดลที่เทรนแล้วเก็บใน Drive/เครื่องผู้ใช้ ไม่แจกใน repo)
7. ไม่สร้างเนื้อหามัลแวร์/โค้ดโจมตีที่ใช้งานได้จริง

## วิธีทำงานกับผู้ใช้
- ตอบ**ภาษาไทย สั้น เข้าใจง่าย** ผู้ใช้ไม่ใช่สายเทคนิคลึก หลีกเลี่ยงศัพท์ยากหรืออธิบายสั้นๆ
- **ถามก่อนเริ่มงานใหม่หรือเปลี่ยนทิศทาง** ห้ามทำหลายเรื่องเองโดยไม่ได้สั่ง ห้ามกลับไปกลับมา
  ผู้ใช้ตัดสินใจ → ทำทีละรอบ → รายงานผล → รอคำสั่งรอบถัดไป
- ทำอะไรเสร็จ: **commit + push ทันที** และอัปเดต `STATE.md` (และ README/Wiki ถ้าเกี่ยว)
- ตัวเลขที่ไม่ได้วัดเองต้องบอกว่า "ประมาณ" และบอกที่มา

## โครงสร้างหลัก
| ที่ | คืออะไร |
|---|---|
| `localbrain/model.py` | Transformer v2 (RoPE, QK-Norm, RMSNorm, ReLU²) |
| `localbrain/train.py` | เทรน / resume (`--resume`) / เริ่มจาก run อื่น (`--init`), WSD schedule |
| `localbrain/optim.py` | Muon |
| `localbrain/coder.py`, `runner.py`, `tasks.py` | vibe coding: เขียน → รันเทสต์ → แก้ → คลังคำตอบ, โจทย์ที่ตรวจได้ (44 แบบ, Py/JS) |
| `localbrain/knowledge.py`, `webtools.py` | คลัง SQLite FTS5, ค้นเว็บ + แคช |
| `localbrain/projindex.py` | ดัชนีโปรเจกต์ระดับฟังก์ชัน (A1): `index-project`, `find-code` |
| `localbrain/agent.py`, `experience.py` | Agent ครบ 9 ขั้น (`agent`): ค้น→สรุป→patch→เทสต์→วนแก้→จำ→รายงาน; สมองเสียบเปลี่ยนได้ (ModelBrain/ScriptedBrain) (A4) |
| `localbrain/patch.py`, `projtest.py` | แก้ไฟล์ด้วย SEARCH/REPLACE (`apply-patch`) + รันเทสต์โปรเจกต์บนสำเนาชั่วคราว (`run-project-tests`) (A3) |
| `localbrain/codelib.py`, `stackexchange.py` | คลังกลาง CodeSearchNet (`build-codelib`, `find-example`) + Stack Overflow (`ask-so`, แคช 30 วัน) (A2) |
| `scripts/retrain.py` | pipeline corpus → pretrain → anneal → SFT → code-eval (ข้ามขั้นที่เสร็จ, resume ได้) |
| `scripts/make_*.py`, `import_v4.py` | สร้าง/คัดข้อมูล SFT |
| `configs/*-colab.json` | ค่าสำหรับ Colab T4 (ตั้งผ่าน `LOCALBRAIN_PRETRAIN_CONFIG`, `LOCALBRAIN_SFT_CONFIG`) |
| `notebooks/LocalBrain_Colab.ipynb` | เทรนบน Colab |

ทดสอบ: `python -m unittest discover tests`

## เทรนบน Colab (ข้อควรระวังที่เจอจริง)
- resume ต้องใช้ `batch_size` / `precision` เท่าเดิม
- SFT บน Colab ต้องใช้ `configs/code-sft-colab.json` (`code-sft.json` จำกัด VRAM 2.8 GiB → OutOfMemory)
- Colab CLI: session ถูกตัดเมื่อ kernel ว่าง ~15 นาที ต้องมี `exec` ค้างรอ (keepalive)
  ใช้ `pgrep -f '[r]etrain.py'` (ไม่งั้นจับตัวเอง) และอย่า `pkill -f` คำที่อยู่ในสคริปต์ตัวเอง
- T4 = 1.15 compute units/ชม.; เทรนเสร็จต้องปิดเครื่อง (`colab stop`) เสมอ
- Drive ที่มีโฟลเดอร์ชื่อซ้ำ → Colab เห็นเป็น `ชื่อ (1)` ต้อง `mv`

## สถานะ ณ 2026-11-07
- โมเดล 20.5M: pretrain val 2.006 → SFT val 1.741
- code-eval pass@1 66.8% แต่ **โจทย์แบบใหม่ ~0%** (ยังจำรูปแบบ ไม่ได้คิดเอง)
- ผู้ใช้ไม่อยากรันงานหนักบนคอมบ้านแล้ว → งานหนักไป Colab (CPU/GPU) และ GitHub Actions

## งานถัดไป (ยังไม่เริ่ม — ถามผู้ใช้ก่อนว่าจะทำข้อไหน)
- **Phase A (Agent ครบ 9 ขั้น):** ~~A1 ดัชนีโปรเจกต์~~ ✅ (`localbrain/projindex.py`) → ~~A2 คลัง CodeSearchNet + Stack Exchange API~~ ✅
  → ~~A3 patch + sandbox~~ ✅ → ~~A4 ลูป สรุป→แก้→ตรวจ→จำ→รายงาน~~ ✅ → A5 ตัวรัน Python/SymPy ในแชท
- **Phase B (~50M):** GQA, context 4,096, sliding/global attention, FIM, BF16, model growth,
  ขยาย `tasks.py` ไป Luau/TS/C++, ข้อมูลโค้ด 1–2B token, เทรนบน Colab L4
- ตัววัดความสำเร็จหลัก: คะแนน **โจทย์แบบใหม่ (unseen)** ใน code-eval
