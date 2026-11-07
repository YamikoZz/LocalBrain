# LocalBrain

**Vibe Coding AI ที่เทรนเองจากศูนย์** บนคอมบ้านหรือ GPU ฟรี
สั่งงานเป็นภาษาอังกฤษ → โมเดลเขียนโค้ด → ระบบรันเทสต์ → แก้จนผ่าน → เก็บคำตอบไว้ใช้ซ้ำ

- ฝึก Transformer และ tokenizer เองจากน้ำหนักสุ่ม ไม่โหลดน้ำหนักสำเร็จรูป ไม่เรียก API ของโมเดล
  (ดาวน์โหลดเฉพาะชุดข้อมูล)
- มีคลังความรู้ SQLite FTS5, ค้น/อ่านเว็บ และเครื่องมือที่เรียกผ่าน CLI
- แผนสถาปัตยกรรม: [ROADMAP.md](ROADMAP.md) · สถานะและตัวเลขล่าสุด: [STATE.md](STATE.md)
- คู่มืออ่านง่าย: [Wiki](https://github.com/YamikoZz/LocalBrain/wiki)

## สถานะปัจจุบัน (2026-11-08)

**กำลังทำรุ่น 200M** (ใหญ่กว่ารุ่นแรก ~10 เท่า) — ตอนนี้อยู่ขั้นสร้าง corpus (ตัดโทเคน) ถัดไปเทรนบน Google Colab L4; สร้างใหม่จากศูนย์ ไม่ได้ต่อจาก 20M

| | รุ่น 20M (เสร็จแล้ว) | รุ่น 200M (กำลังทำ) |
|---|---|---|
| ขนาด | 20.5M พารามิเตอร์, context 512 | 193M พารามิเตอร์ (width 1024, 16 ชั้น, GQA), context 2,048 |
| ข้อมูลเทรนหลัก | 393M tokens | ~4.0B tokens (โค้ด GitHub 110 ไฟล์ + FineWeb-Edu อังกฤษ 8 ไฟล์) |
| เครื่อง | Colab T4 | Colab L4, bf16, Muon; วัดจริง ~23.6 วินาที/step, VRAM 9.5 GiB |
| แผน | — | 7,600 steps ≈ 50 ชม. ≈ 97 compute units, checkpoint เก็บใน Google Drive กู้ต่อข้าม session ได้ |
| ผล | pretrain val 2.006 → SFT val 1.741; code-eval pass@1 66.8% = รวม 196 โจทย์ (แบบที่เคยเห็น 79.9%: Py 76.8%, JS 82.9%; แบบใหม่ 0%) — วัดจากโมเดลอย่างเดียว ไม่มี Agent ช่วยแก้ | ยังไม่มี — จะวัดด้วย HumanEval, MBPP และโจทย์แบบใหม่ |
| ยังทำไม่ได้ | โจทย์แบบใหม่ ~0%, เลขในหัว, เขียน patch แก้โปรเจกต์เอง (Agent ทดสอบด้วย patch ที่เตรียมไว้เท่านั้น) | — |

การตรวจซ้ำ (2026-11-08): ผลเดิม pass@1 66.8% (131/196) ถูกตรวจซ้ำโดยรันคำตอบ greedy ที่บันทึกไว้กับ test cases เดิม และให้ผลตรงกันครบทุกข้อ — pass@8 ของผลเดิมตรวจซ้ำไม่ได้ (ไม่ได้เก็บคำตอบสุ่มไว้; รายงานใหม่เก็บแล้ว), โจทย์แบบใหม่ 0/32

ขั้นของรุ่น 200M: สร้าง corpus (ตัดโทเคน) → pretrain → anneal (python-edu) → SFT (เน้น Python + Lua, JS + TS, ลด C/C++) → ทดสอบ
ภาษาที่เทรน: โค้ด + ภาษาอังกฤษเท่านั้น

ยังเป็นโมเดลทดลอง ผลลัพธ์ต้องผ่านการรันเทสต์ก่อนใช้งานเสมอ

## โหลดไปตอนนี้ได้อะไร

| ได้ ✅ | ยังไม่ได้ ❌ |
|---|---|
| โค้ดทั้งหมด (MIT License): ตัวสร้าง/เทรนโมเดล, tokenizer, Agent, ระบบรันเทสต์, คลังความรู้, ค้นเว็บ | **ตัวโมเดลที่เทรนเสร็จแล้ว** — ยังไม่แจก (จะแจกเมื่อคุณภาพพร้อม) |
| pipeline เทรนครบชุด + notebook Colab + คู่มือ ([Wiki](https://github.com/YamikoZz/LocalBrain/wiki)) | ใช้เป็น AI เขียนโค้ดได้ทันที (ต้องเทรนเองก่อน) |

- **สายเทคนิค:** ศึกษาหรือเทรนเองได้ รุ่น 20M ใช้ Colab T4 ไม่ถึง 50 บาท (ดูขั้นตอนด้านล่าง)
- **คนทั่วไปที่อยากได้ AI ไว้ใช้:** ตอนนี้ยังใช้ไม่ได้ ต้องรอแจกตัวโมเดล

หลักของโปรเจกต์: เทรนเองทุกขั้น ไม่ใช้น้ำหนักโมเดลของคนอื่น, ไม่มีระบบปฏิเสธคำสั่งในตัว,
รุ่นที่รันในเครื่องฟรีและไม่ผูกกับบริษัทใด (อนาคตอาจมี API เสริมสำหรับคนเครื่องไม่แรง — ยังเป็นแค่แผน)

## เป้าหมายขนาดโมเดลและงบ (ประมาณการ)

เทรนจากศูนย์ทุกขั้น งบคิดจากเช่า H100 ~70 บาท/ชม. รวมเผื่อทดลองซ้ำ 1.5–2 เท่า (คลาดได้ ±50%)
HumanEval ในตารางเป็นค่าโดยประมาณของโมเดลสาธารณะขนาดใกล้เคียง ใช้เป็นจุดเทียบ ไม่ใช่ผลของเรา

| ขั้น | ขนาด | ข้อมูล | งบ | เทียบกับ (HumanEval) |
|---|---|---|---|---|
| ตอนนี้ | 20M | 0.4B tokens | < 50 บาท | Codex 12M–25M (~2–3%) |
| 1 | 125M | 2.5B | ~500–1,000 บาท | Codex 85M (~8%) |
| 2 | 350M | 7B | ~2,000–4,000 บาท | CodeGen-350M-mono (~13%) |
| **เป้าหลัก** | **1B** | 20B | ~10,000–20,000 บาท | StarCoderBase-1B ถึง phi-1 (~15–50%) |
| เป้าเสริม | 3B | 60B | ~80,000–150,000 บาท | ~25–45% |

ก่อนขั้น 1B ต้องเพิ่ม: เทรน BF16, หลาย GPU (FSDP), ข้อมูลโค้ดหลักหมื่นล้าน token (เช่น The Stack v2)
5B/10B ไม่อยู่ในแผน: งบหลักแสน–ล้าน และรันบนคอมทั่วไปไม่ได้

สนับสนุนค่า GPU: GitHub Sponsors ของ YamikoZz (รออนุมัติ)

## Pipeline ปัจจุบัน (code model)

```powershell
# 1) สร้าง corpus โดยใช้ tokenizer เดิม (dataset อื่นจะได้ใช้ร่วมกันได้)
.\.venv\Scripts\python.exe -m localbrain build-corpus --code <code-parquet...> --english <fineweb-parquet> --output workspace/dataset-code-big --max-tokens 640000000 --tokenizer workspace/dataset-code/tokenizer.json
# 2) pretrain → anneal → SFT → code-eval ในคำสั่งเดียว (ข้ามขั้นที่เสร็จแล้ว, resume ได้)
.\.venv\Scripts\python.exe scripts/retrain.py
```

`scripts/retrain.py` ใช้ config pretrain ตามตัวแปร `LOCALBRAIN_PRETRAIN_CONFIG`
(ไม่ตั้ง = `configs/code-pretrain-big.json` สำหรับ GPU 4 GB, บน Colab ใช้ `configs/code-pretrain-colab.json`)

### เทรนบน Google Colab
ดู [notebooks/README.md](notebooks/README.md) และ `notebooks/LocalBrain_Colab.ipynb`

### สร้างข้อมูล SFT

| สคริปต์ | ได้อะไร |
|---|---|
| `scripts/make_arithmetic.py` | บวก/ลบ/คูณ/หารที่คำนวณตรวจแล้ว 4,258 ตัวอย่าง |
| `scripts/make_conversation.py` | ทักทาย/แนะนำตัว/คุยเล่น (คำถามไม่ซ้ำกัน เพราะ dedup ตัดตามคำถาม) |
| `scripts/make_hardcore.py` | โค้ด 7 ภาษา + ตอบเลขด้วยโค้ด `print(...)` |
| `scripts/make_oasst_full.py` | OpenAssistant oasst2 แบบหลายรอบสนทนา 11,750 คู่ |
| `scripts/import_v4.py` | คัดตัวอย่างสมดุลจาก Programming Dataset v4 (Python/JS/TS/Luau/C++/SQL) 8,360 แถว ตรวจโดยไม่รันโค้ดจากชุดข้อมูล |

รวมไฟล์ JSONL แล้วสร้าง dataset ด้วย tokenizer เดิม:

```powershell
.\.venv\Scripts\python.exe -m localbrain prepare-instructions workspace/raw-sft/chat-full.jsonl --output workspace/dataset-english-sft-chat --tokenizer workspace/dataset-code/tokenizer.json --config configs/code-sft.json
```

ทดสอบข้อมูลใหม่เร็วๆ บนคอมบ้านก่อนใช้จริง (SFT 500 steps):

```powershell
.\.venv\Scripts\python.exe -m localbrain train --data workspace/dataset-english-sft-chat --run workspace/run-sft-test --config configs/code-sft-test.json --init workspace/run-code-pretrain-big/latest.pt
```

### ใช้งานโมเดล

```powershell
# แชท
.\.venv\Scripts\python.exe -m localbrain.cli playground --run workspace/run-english-sft2
# vibe coding: เขียน → รันเทสต์ → แก้จนผ่าน
.\.venv\Scripts\python.exe -m localbrain.cli code --run workspace/run-english-sft2 --language python --prompt "a function that checks if a number is prime"
```

---

เนื้อหาด้านล่างเป็นเอกสารของระบบรุ่นแรก (โมเดล 6.9M, ข้อมูลภาษาไทย) และเครื่องมือพื้นฐาน
ซึ่งยังใช้งานได้

## โครงสร้าง

```text
.
├── pyproject.toml
├── README.md
├── ROADMAP.md                 # แผนสถาปัตยกรรม
├── STATE.md                   # สถานะและตัวเลขล่าสุด
├── Open-Chat.cmd
├── Open-Model.cmd
├── .gitignore
├── notebooks/
│   ├── LocalBrain_Colab.ipynb # เทรนบน Colab T4
│   └── README.md
├── configs/
│   ├── code-pretrain-big.json   # pretrain GPU 4 GB
│   ├── code-pretrain-colab.json # pretrain Colab T4
│   ├── code-anneal.json
│   ├── code-sft.json
│   ├── code-sft-test.json       # SFT สั้นสำหรับทดสอบข้อมูล
│   ├── default.json
│   ├── smoke.json
│   └── instruction.json
├── data/
│   └── example.txt
├── localbrain/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                 # คำสั่งทั้งหมด
│   ├── config.py
│   ├── model.py               # Transformer v1/v2
│   ├── optim.py               # Muon, CPUAdamW
│   ├── train.py               # เทรน/resume/init, WSD schedule
│   ├── data.py
│   ├── corpus.py              # build-corpus จาก parquet
│   ├── instruction.py         # prepare-instructions (dedup ตามคำถาม)
│   ├── chatdata.py            # ผสมข้อมูลแชท OASST/Dolly/โค้ด
│   ├── tasks.py               # โจทย์โค้ดที่ตรวจได้
│   ├── coder.py               # vibe coding: เขียน → รันเทสต์ → แก้ → คลังคำตอบ
│   ├── runner.py              # รันเทสต์ Python/JS
│   ├── playground.py          # หน้าต่างแชท
│   ├── extra.py
│   ├── personal.py
│   ├── hardware.py
│   ├── knowledge.py           # คลังความรู้ SQLite FTS5
│   ├── documents.py
│   ├── webtools.py            # ค้นเว็บ/อ่านหน้าเว็บ
│   ├── helpers.py
│   ├── sandbox.py
│   ├── tools.py
│   ├── assistant.py
│   ├── chat.py
│   └── inference.py
├── examples/
│   ├── calculator.json
│   └── sandbox.json
├── sandbox/
│   └── Dockerfile
├── scripts/
│   ├── retrain.py             # pipeline pretrain → anneal → SFT → eval
│   ├── make_arithmetic.py
│   ├── make_conversation.py
│   ├── make_hardcore.py
│   ├── make_oasst_full.py
│   └── fetch_instructions.py
├── tests/
│   ├── test_system.py
│   ├── test_chat.py
│   └── test_instruction.py
└── workspace/                 # ผลลัพธ์ระหว่างใช้งาน ไม่ใช่ source code
    ├── dataset-smoke/         # tokenizer, train.bin, val.bin, meta.json
    ├── run-smoke/             # latest.pt, tokenizer, config, metrics
    ├── knowledge.sqlite3
    └── files/                # ขอบเขตเครื่องมืออ่าน/เขียนไฟล์
```

## ติดตั้งและตรวจเครื่อง

เปิด PowerShell ที่โฟลเดอร์โปรเจกต์:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m localbrain doctor
```

เครื่องนี้ติดตั้งใน .venv ให้แล้ว ใช้คำสั่ง doctor และคำสั่งถัดไปได้เลย
เวอร์ชัน CUDA wheel นี้ตรวจบน Python 3.14 และ GTX 1650 SUPER ของเครื่องนี้
สำหรับเครื่องอื่นดู [คู่มือติดตั้ง PyTorch](https://pytorch.org/get-started/locally/)
ไลบรารี tokenizers มี dependency huggingface-hub แต่โค้ดโปรเจกต์ไม่เรียก Hub/API
และไม่ดาวน์โหลด tokenizer หรือโมเดลใด ๆ

## ทดลองฝึกจากศูนย์

```powershell
.\.venv\Scripts\python.exe -m localbrain prepare data/example.txt --config configs/smoke.json --output workspace/dataset-smoke
.\.venv\Scripts\python.exe -m localbrain train --data workspace/dataset-smoke --config configs/smoke.json --run workspace/run-smoke
.\.venv\Scripts\python.exe -m localbrain generate "ผู้ใช้: สวัสดี" --run workspace/run-smoke --max-tokens 40
```

หาก dataset-smoke และ run-smoke มีอยู่แล้วจากการตรวจระบบ ไม่ต้อง prepare ซ้ำ
ดูผลเดิมหรือฝึกต่อด้วยจำนวน steps เป้าหมายที่มากกว่า checkpoint:

```powershell
.\.venv\Scripts\python.exe -m localbrain train --data workspace/dataset-smoke --config configs/smoke.json --run workspace/run-smoke --resume --steps 60
```

โมเดล smoke มี 2 ชั้น กว้าง 64 ใช้ทดสอบทางเทคนิค คำตอบยังไม่เป็นภาษาได้
prepare ปฏิเสธเขียนทับ dataset เดิม และ train ปฏิเสธเขียนทับ checkpoint ถ้าไม่ระบุ --resume
การฝึกต่อจะตรวจ tokenizer, ข้อมูลกำกับชุดฝึก และการตั้งค่าให้ตรงกับ checkpoint

## ฝึกกับข้อมูลจริง

ใส่ไฟล์ข้อความ UTF-8 .txt หรือ .md ใน data/corpus/ แล้วรัน:

```powershell
.\.venv\Scripts\python.exe -m localbrain prepare data/corpus --config configs/default.json --output workspace/dataset
.\.venv\Scripts\python.exe -m localbrain train --config configs/default.json --data workspace/dataset --run workspace/run
```

default ใช้ Decoder-only Transformer 6 ชั้น, width 256, 8 heads, context 256,
คลังโทเคนเป้าหมาย 8192, tied embeddings รวมประมาณ 6.9 ล้านพารามิเตอร์เมื่อคลังเต็ม
จำนวนโทเคนจริงอาจน้อยกว่าเป้าหมายหากข้อมูลน้อย
Byte-level BPE รองรับภาษาไทยและข้อความที่ไม่เคยเห็นผ่าน byte fallback

- แบ่งท้ายแต่ละไฟล์ 10% เป็น validation ก่อนฝึก tokenizer
- ไฟล์แต่ละไฟล์ต้องอย่างน้อย 100 ตัวอักษร และไม่เกิน 64 MiB
- เก็บโทเคนเป็น uint16 memory-mapped บนดิสก์ ไม่โหลด corpus ทั้งหมดลง VRAM
- batch เล็ก + gradient accumulation + activation checkpointing ลด VRAM
- CPUAdamW เก็บ moment tensors สองชุดใน RAM; น้ำหนัก gradient และการคำนวณโมเดลอยู่บน GPU
- ใช้ FP32 เพื่อความเรียบง่ายและเสถียร ไม่ได้ทำ multi-device layer offload หรือรวม RAM เป็น VRAM
- เลือก --device cpu ได้ ส่วนจำนวน CPU threads ตั้งใน config
- ไม่จอง RAM/VRAM ทั้งหมดให้เต็ม เพราะระบบปฏิบัติการและโปรแกรมอื่นต้องใช้งานด้วย
- การ offload มีค่าโอนข้อมูล อาจช้ากว่า AdamW บน GPU ถ้าโมเดลจุ GPU อยู่แล้ว
- ดู train_loss, val_loss, peak_vram_mib ใน metrics.jsonl ไม่ใช้เพียง loss บอกว่าคุยรู้เรื่อง
- หยุด Ctrl+C แล้วฝึกต่อจาก checkpoint ที่บันทึกครบล่าสุด ไม่บันทึก update ที่ถูกขัดจังหวะ

ข้อมูลถามตอบแบบข้อความธรรมดาใช้ ผู้ใช้: ... ผู้ช่วย: ... ได้
แต่สำหรับ loss เฉพาะคำตอบและชุดทดสอบแยก ใช้ prepare-instructions ในหัวข้อถัดไป
การฝึกต่อ checkpoint ต้องใช้ dataset/tokenizer เดิม ไม่เปลี่ยน tokenizer กลางการฝึก

## ฝึกสมองตอบจากน้ำหนักจริง

ใช้ข้อมูล [Han Instruction Dataset โดย Wannaphong / PyThaiNLP](https://github.com/wannaphong/han-instruction-dataset)
ที่มีทั้งตัวอย่างจากคนและข้อมูลสังเคราะห์ ตาม README ของต้นทางใช้ CC-BY-SA 4.0
ดาวน์โหลดเฉพาะข้อมูลจาก commit ที่ตรึงไว้ ไม่ดาวน์โหลดโมเดลหรือเรียก inference API
หลักฐาน URL, commit และ SHA256 อยู่ workspace/raw/SOURCE.json
การปรับข้อมูล: เอาคำถามซ้ำออก ใช้คู่ user/assistant แรก และคัดตัวอย่างที่ยาวเกิน context ออก

```powershell
.\.venv\Scripts\python.exe scripts/fetch_instructions.py
.\.venv\Scripts\python.exe -m localbrain prepare-instructions workspace/raw/han-instruction-dataset.json --config configs/instruction.json --output workspace/dataset-instruction
.\.venv\Scripts\python.exe -m localbrain train --data workspace/dataset-instruction --config configs/instruction.json --run workspace/run-instruction --steps 6000
```

ไฟล์ที่เตรียมแล้วมี train 8,308, validation 499, test 515 ตัวอย่าง
แยกตาม hash ของคำถามที่ normalize แล้วก่อนฝึก tokenizer
คำถามซ้ำตรงกันอยู่ split เดียวกัน แต่ยังไม่ได้ขจัดคำถามที่มีความหมายคล้ายกันทั้งหมด
Tokenizer เรียนจาก train เท่านั้น; loss ไม่คิดส่วน prompt/padding
คำนวณ validation ด้วยตัวอย่างสุ่มชุดเดิม และบันทึก best.pt ตาม validation loss
ชุด test ใช้ประเมินภายหลัง ไม่ใช้เลือก checkpoint

โมเดลใหม่เริ่มจากน้ำหนักสุ่ม 6,895,616 พารามิเตอร์ ไม่ใช้ run-smoke เป็นพื้นฐาน
GPU คำนวณ forward/backward; CPU เตรียม batch; RAM เก็บสถานะ optimizer
RAM กับ VRAM ไม่ได้รวมเป็นหน่วยความจำ GPU ก้อนเดียว

หากไฟล์มีอยู่แล้ว ไม่ต้องดาวน์โหลดหรือ prepare ซ้ำ
ฝึกต่อให้เพิ่ม --resume และระบุ steps เป้าหมายที่มากกว่า checkpoint ล่าสุด

```powershell
.\.venv\Scripts\python.exe -m localbrain model-eval
.\.venv\Scripts\python.exe -m localbrain chat --model-only --run workspace/run-instruction
```

หรือดับเบิลคลิก **Open-Model.cmd** เปิดหน้าต่างไทยชื่อ LocalBrain Learned Model
โหมดนี้คำถามทั่วไปทั้งหมด รวม hello และ ทำอะไรได้บ้าง ส่งเข้าโมเดลจริง
ไม่มีคำตอบตายตัวและไม่ค้นเอกสารอัตโนมัติ; /help และคำสั่ง /... ยังเป็นเครื่องมือที่สั่งเองได้
โมเดลโหลดค้างครั้งเดียวต่อ session เลือก best.pt ถ้ามี ไม่โหลดใหม่ทุกคำถาม
การสร้างคำตอบใช้ greedy decoding พร้อมลดการซ้ำ ไม่ได้เพิ่มความรู้จากกติกาเหล่านี้
ยังเป็นการตอบครั้งละคำถาม ไม่ได้ฝึกความจำหลายรอบสนทนา

ผลรอบ 2026-11-02: ฝึกจริง 6,000 ขั้น; validation loss 8.76487 → 4.27753;
held-out test loss 4.25104 จาก 515 ตัวอย่างครบทุกตัวอย่าง
ตรวจตัวอย่างคำตอบแล้ว **ยังไม่ผ่านคุณภาพสำหรับผู้ช่วยทั่วไป** มีคำเพี้ยนและตอบผิด
ทดสอบโค้ดผ่าน 19 รายการไม่ได้หมายความว่าคำตอบโมเดลผ่าน
Open-Model.cmd จึงมีไว้ทดลองดูผลฝึก ไม่ได้แทนโหมดแชทหลัก

**การมี checkpoint ไม่ได้หมายความว่าฉลาดหรือถูกต้องทุกเรื่อง**
ข้อมูลรอบนี้ยังมีจำนวนและความหลากหลายจำกัด คำตอบอาจซ้ำ ผิด หรือไม่รู้เรื่อง
ดูผล held-out loss และตัวอย่างคำตอบจริงใน workspace/run-instruction/evaluation.json
loss ไม่ใช่คะแนนความถูกต้องของข้อเท็จจริง และไม่ใช่หลักฐานว่าพร้อมใช้เป็นผู้ช่วยทั่วไป

## ฝึกความรู้ด้านความมั่นใจทางไซเบอร์: Malware และชุดเครื่องมือโจมตี

มี source ฝึกเพิ่มเติมแนะนำ (`localbrain/extra.py`) สำหรับ defensive security, FiveM Lua และ offensive security (malware + attack toolkits) จาก subfolder ของเอกสาร markdown

### รวบรวม offensive-security documents (malware analysis + attack toolkits)

```powershell
.\.venv\Scripts\python.exe -m localbrain collect-offensive --root D:\model ai test\security-offensive --output workspace/raw-offensive/offensive.jsonl
```

แต่ละ immediate subfolder ของ root คือ topic kind (เช่น malware, attack-toolkits, red-team, c2, threat-intel)
แล้วเก็บ `*.md` ทั้งหมดเป็น JSONL แบบ `{"path": "...", "text": "..."}` โดย path มี prefix `offensive/...`

### ใส่ offensive source ใน final anneal mix

```powershell
.\.venv\Scripts\python.exe -m localbrain build-anneal --tokenizer workspace/run-code-pretrain/tokenizer.json --code <code-parquet> --pyedu <pyedu-parquet> --english <fineweb-parquet> --output workspace/dataset-anneal-offensive --extra offensive=workspace/raw-offensive/offensive.jsonl --extra-share 0.10
```

แล้วเทรนจาก checkpoint ที่ผ่าน pretrain:

```powershell
.\.venv\Scripts\python.exe -m localbrain train --data workspace/dataset-anneal-offensive --run workspace/run-code-anneal-offensive --config configs/code-anneal.json --init workspace/run-code-pretrain/latest.pt --steps 4000
```

### บทเรียนฝึก (SFT lessons) ด้าน malware / attack tooling

`localbrain/extra.py` มี `offensive_lessons()` ให้สร้าง question/answer pairs สำหรับฝึกสมองตอบ:

```powershell
.\.venv\Scripts\python.exe -c "from localbrain.extra import offensive_lessons; import json; [print(json.dumps(r, ensure_ascii=False)) for r in offensive_lessons()]" > workspace/raw-offensive/offensive-lessons.jsonl
.\.venv\Scripts\python.exe -m localbrain prepare-instructions workspace/raw-offensive/offensive-lessons.jsonl --config configs/instruction.json --tokenizer workspace/run-instruction/tokenizer.json --output workspace/dataset-instruction-offensive
.\.venv\Scripts\python.exe -m localbrain train --data workspace/dataset-instruction-offensive --run workspace/run-instruction-offensive --config configs/instruction.json --steps 6000
```

## คลังความรู้และถามตอบ

หาก PowerShell แสดงภาษาไทยเป็นสี่เหลี่ยม ให้ดับเบิลคลิก **Open-Chat.cmd**
เพื่อเปิดแชทใน IDLE ของ Python ที่ติดตั้งอยู่แล้ว โดยไม่แก้ฟอนต์ระบบ
หน้าต่างชื่อ LocalBrain Thai Chat; พิมพ์หลัง คุณ> แล้วกด Enter
อ่านคำสั่งเปิด shell ได้ที่ [คู่มือ IDLE](https://docs.python.org/3/library/idle.html#command-line-usage)


```powershell
.\.venv\Scripts\python.exe -m localbrain ingest data/example.txt
.\.venv\Scripts\python.exe -m localbrain search "ภาษาไทย"
.\.venv\Scripts\python.exe -m localbrain ask "VRAM"
.\.venv\Scripts\python.exe -m localbrain chat
```

คำสั่งในแชท (พิมพ์ได้หลัง คุณ> โดยไม่ออกไป Terminal):

```text
ทำอะไรได้บ้าง
/help
/calc (8+4)*2
/search VRAM
/web Python tutorial
/ingest data/example.txt
/time
/stats
/tools
/tool calculator {"expression":"2+3"}
/more
/exit
```

คำตอบยาวแบ่งเป็นหน้าละไม่เกิน 900 ตัวอักษร/12 บรรทัด ใช้ /more อ่านต่อ
หากคำสั่งผิดจะแจ้งข้อผิดพลาดและรอคำสั่งถัดไป
/web ค้นและเก็บไม่เกิน 3 หน้า; /ingest เพิ่มข้อมูลเข้าคลัง ไม่ได้ฝึกน้ำหนัก
คำทักทายและคำถามความสามารถใช้กติกา ส่วนคำถามทั่วไปยังเป็นการค้นเอกสาร
คำสั่งในแชทไม่ฝึกโมเดลอัตโนมัติ การฝึกน้ำหนักจริงใช้ train แยกตามหัวข้อด้านบน

ask/chat โดยปริยายแสดงข้อความคัดจากเอกสารพร้อมแหล่งที่มา; ask ส่ง JSON พร้อมวันเก็บข้อมูล
ส่วน chat แสดงข้อความอ่านง่าย คำทักทาย เช่น hello/สวัสดี ตอบด้วยกติกาโดยไม่ค้นเว็บ
ไม่สร้างคำตอบปลอมจากโมเดลที่ยังไม่ได้ฝึก
ถ้าต้องการทดลองให้โมเดลที่ฝึกเองเรียบเรียง:

```powershell
.\.venv\Scripts\python.exe -m localbrain ask "VRAM" --run workspace/run-smoke
```

ผลจากโมเดลทดลองอาจผิดและใช้บริบทได้จำกัดตาม context
การค้นเป็น lexical substring/trigram รองรับไทย ไม่ใช่ semantic embedding
ข้อความที่คล้ายกันแต่ใช้คนละคำอาจค้นไม่เจอหรือเจอส่วนที่ไม่ตรงคำถาม
การนำเอกสารเข้าคลังไม่เปลี่ยนน้ำหนักโมเดล และไม่ฝึกต่ออัตโนมัติ
ข้อมูลแหล่งเดียวกันจะอัปเดตแทนของเก่า ไม่เพิ่มซ้ำ

## ค้นเว็บและเก็บไว้ล่วงหน้า

```powershell
.\.venv\Scripts\python.exe -m localbrain ingest "https://docs.python.org/3/tutorial/index.html"
.\.venv\Scripts\python.exe -m localbrain research "Python tutorial" --limit 3
.\.venv\Scripts\python.exe -m localbrain ask "Python" --web
.\.venv\Scripts\python.exe -m localbrain refresh
```

research ค้น HTML สาธารณะของ Bing แล้วอ่านและเก็บไม่เกิน limit หน้า
ไม่ใช้ Search API และไม่เก็บทั้งอินเทอร์เน็ต ผู้ให้บริการอาจบล็อกหรือแสดง CAPTCHA
ถ้าค้นไม่ได้ใช้ ingest URL โดยตรง ระบบไม่หลบการบล็อก
อ่านเฉพาะ HTML/text สาธารณะ ตรวจ robots.txt จำกัดขนาดและ timeout
ไม่เรนเดอร์ JavaScript ไม่เข้าเว็บที่ต้องล็อกอิน และไม่ใช้ browser automation
URL ตรวจ public IP ก่อนเปิดและก่อน redirect แต่ไม่ได้เป็น network sandbox
แบบป้องกัน DNS rebinding สมบูรณ์ จึงไม่ควรเปิดรับ URL จากผู้ใช้ภายนอกที่ไม่ไว้ใจ

refresh อัปเดตเฉพาะ URL ที่เก็บไว้แล้ว ไม่อัปเดตไฟล์ในเครื่องอัตโนมัติ
หากต้องการทำซ้ำทุกชั่วโมงแบบ foreground ระบุจำนวนรอบ:

```powershell
.\.venv\Scripts\python.exe -m localbrain refresh --cycles 24 --interval 3600
```

ไม่มีการติดตั้ง background service หรือ Windows Scheduled Task

## เครื่องมือ

```powershell
.\.venv\Scripts\python.exe -m localbrain tools
.\.venv\Scripts\python.exe -m localbrain tool calculator --args-file examples/calculator.json
.\.venv\Scripts\python.exe -m localbrain tool get_time
```

มี calculator, get_time, search_local, search_web, read_webpage, save_knowledge,
read_file, write_file, list_files, run_python
ใช้ --args-file path.json แทน --args สำหรับข้อความหลายบรรทัดหรือ shell ที่ escape JSON ต่างกัน

read_file/write_file จำกัดขอบเขตใน workspace/files และไม่เขียนทับไฟล์เดิม
ingest อ่านไฟล์จากตำแหน่งที่คุณระบุโดยตรง
รองรับ TXT, MD, HTML, CSV; PDF ต้องติดตั้งเพิ่ม:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[pdf]"
```

PDF สแกนภาพยังไม่มี OCR
ไม่มี host terminal หรือการรันคำสั่งจากคำตอบโมเดลอัตโนมัติ
ตัวควบคุม Python จัดเส้นทางการค้นตามคำสั่งที่ชัดเจน โมเดลยังไม่ได้ฝึก tool calling

## Code Sandbox

หากเครื่องมี Docker และ daemon พร้อมใช้งาน สร้าง image ก่อน:

```powershell
docker build -t localbrain-sandbox:1 sandbox
.\.venv\Scripts\python.exe -m localbrain tool run_python --args-file examples/sandbox.json
```

รัน Python ใน container ไม่มี network, ไม่มี host mount, filesystem อ่านอย่างเดียว
จำกัด RAM 128 MiB, 1 CPU, 32 processes, เวลาไม่เกิน 30 วินาที และ output 64 KiB
ใช้พื้นที่ชั่วคราวใน container เท่านั้น ไม่ติดตั้งแพ็กเกจตามโค้ดที่รัน
ไม่ดาวน์โหลด image อัตโนมัติระหว่างเรียก tool
ถ้าไม่มี Docker/image จะปฏิเสธรัน **ไม่ย้อนกลับไปรันบนเครื่องโดยตรง**
เครื่องนี้ยังไม่มี Docker จึงยังไม่ได้ตรวจการรันโค้ดใน container จริง

## ตรวจระบบ

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

ตรวจความเป็น causal ของ Transformer, การฝึกและฝึกต่อให้ได้ผลตรงกับรอบต่อเนื่อง,
การเข้ารหัสไทย, CPUAdamW เทียบกับ AdamW มาตรฐาน, ค้น/อัปเดตคลัง,
ขอบเขตไฟล์, calculator, การปิด Sandbox เมื่อไม่มี Docker และการปฏิเสธ URL ภายใน
ผลทดสอบปัจจุบันและเวลาอยู่ใน STATE.md
