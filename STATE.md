# LocalBrain — สถานะล่าสุด 2026-11-06 (เย็น): ย้ายเทรนขึ้น Colab

- เป้าหมายชัดเจนแล้ว: **Vibe Coding AI** สั่งงานภาษาอังกฤษ → เขียนโค้ด → รันตรวจ → แก้จนผ่าน
  เน้นโค้ดหลายภาษา และคณิต/ฟิสิกส์ผ่านโค้ด แผนสถาปัตยกรรมอยู่ใน ROADMAP.md
- โมเดล v2: 20,454,784 พารามิเตอร์, width 384, 8 layers, 6 heads, context 512, vocab 16384,
  Muon + WSD schedule, FP32
- Corpus `dataset-code-big`: 319,223,829 train tokens (โค้ด 8 shards + FineWeb-Edu, โค้ด ~85%)
  ใช้ tokenizer เดิม (sha256 ff5386c5...) ทุก dataset จึงใช้ร่วมกันได้
- เทรนบน Google Colab T4 ฟรี (`notebooks/LocalBrain_Colab.ipynb`, config `code-pretrain-colab.json`)
  - batch 32, 24,000 steps = 393M tokens (~1.2 epoch), ~1.05 วินาที/step, ใช้ VRAM ~8.8/15 GB
  - ลดจาก 48,000 steps เพราะ batch 32 จะเห็นข้อมูลซ้ำ ~2.5 รอบ
  - ล่าสุด step 14,000: val_loss 9.84 → 2.177 ยังลดลงต่อเนื่อง
  - ต่อด้วย anneal 4,000 steps (33M tokens) และ SFT 1,200 steps อัตโนมัติ
  - Colab ฟรีหลุดราว 3.5 ชม. ต่อ session: รัน cell 1-5 ใหม่แล้ว resume จาก checkpoint ใน Drive
- หยุดเทรนบนคอมบ้านแล้ว (run-code-pretrain-big บ้านค้างที่ step 22,250, val_loss 2.377)
  Colab ที่ step 11,500 ได้ val_loss 2.205 ต่ำกว่า เพราะ batch ใหญ่กว่า
- ข้อมูล SFT ปัจจุบันใน Drive `dataset-english-sft`: train 69,763 / val 3,834 / test 3,877
  - Dolly 15k, OpenAssistant oasst2 แบบหลายรอบสนทนา 11,750 คู่ (เดิมใช้แค่รอบแรก 5,284)
  - code skills (Python/JS), arithmetic ที่ตรวจแล้ว 4,258, บทสนทนาทักทาย 115,
    โค้ด 7 ภาษา + math-via-code 230 (`scripts/make_hardcore.py`)
- ทดสอบบนคอมบ้านด้วย SFT สั้น 500 steps: ได้รูปแบบคำตอบ แต่ logic โค้ดและเลขยังผิด
  โค้ดหลายภาษายังไม่ได้ผลที่ขนาด 20M (ข้อมูลภาษาละ ~15 ตัวอย่างน้อยเกินไป)
- Colab CLI (`google-colab-cli`) ล็อกอินได้ แต่ runtime ที่เปิดจากเบราว์เซอร์แสดงเป็น `[?]` orphan
  สั่ง `exec` เข้าไม่ได้ ต้องสร้าง session ผ่าน `colab new` ซึ่ง GPU ต้องมี compute units
- 2026-11-07: ซื้อ Colab Pay As You Go 100 units (343 บาท). T4 กินจริง 1.15 units/ชม.
  สั่งเทรนผ่าน Colab CLI ได้: `colab new -s NAME --gpu T4` → `colab drivemount` (ผู้ใช้กดอนุญาตลิงก์)
  → `colab upload` repo zip → รัน retrain.py แบบ nohup
  ข้อควรระวัง: session จาก CLI ถูกตัดเมื่อ kernel ว่าง ~15 นาที แม้ process เบื้องหลังยังรัน
  ต้องรัน `colab exec --timeout 21600` สคริปต์ที่วนรอจน retrain.py จบ เพื่อให้ kernel busy ตลอด
  resume ล่าสุดจาก step 19,200 (val_loss 2.141 ที่ step 18,500)
- 2026-11-07: สมัคร GitHub Sponsors แล้ว (รออนุมัติ) — เพิ่ม `.github/FUNDING.yml` ไว้แล้ว
  ปุ่ม Sponsor จะแสดงเมื่อโปรไฟล์ได้รับอนุมัติและ repo เป็น public
  เป้างบและขนาดโมเดล (20M → 1B, เป้าเสริม 3B) อยู่ใน README "เป้าหมายขนาดโมเดลและงบ"
- 2026-11-07 Colab รอบ resume: pretrain ครบ 24,000 steps val_loss 2.006 (14,000: 2.177, 22,000: 2.058)
  anneal 4,000 steps เสร็จ (val 2.161 บนชุด anneal, เทียบตรงกับ pretrain ไม่ได้)
  SFT ล้มที่ ~step 500 ด้วย OutOfMemory (code-sft.json จำกัด VRAM 2.8 GiB สำหรับคอมบ้าน, curriculum ทำให้ลำดับยาวขึ้น)
  แก้: configs/code-sft-colab.json + LOCALBRAIN_SFT_CONFIG; retrain.py ข้ามขั้นที่ครบแล้ว และ resume SFT ที่ค้าง
  (เดิม pretrain ครบแล้วสั่งซ้ำจะ error "steps must exceed checkpoint step", SFT ค้างจะถูกข้ามผิด)
  SFT ครบ 1,200 steps: val_loss 1.860 (250) → 1.759 (1,000) → **1.741 (1,200)**
  code-eval (196 โจทย์, 8 ตัวอย่าง/โจทย์): overall pass@1 66.8%, pass@8 69.9%
  - โจทย์แบบที่เคยเห็นตอนเทรน: Python pass@1 76.8%, JavaScript 82.9%
  - โจทย์แบบที่ไม่เคยเห็น: Python pass@1 0% (pass@8 12.5%), JavaScript 0% → ยังจำรูปแบบ ไม่ได้คิดเอง
  ทดสอบแชท: is_even ถูก, Lua add ถูก, "7 + 8 = 9" ผิด, reverse string JS ผิด,
  "123 times 45" เขียนโค้ดแต่ตัวเลขผิด (45 * 45), hello ตอบแปลก
  โมเดลดาวน์โหลดมาไว้ที่ workspace/run-english-sft2 แล้ว (best.pt = step 1,200) และปิดเครื่อง Colab แล้ว
  units คงเหลือ 96.39 (รอบนี้ใช้ ~3.6 units ≈ 12 บาท รวมเวลาเครื่องค้างจากบั๊ก keepalive ~0.8)
  ใช้ ~2.6 units (~9 บาท) สำหรับ pretrain ที่เหลือ + anneal
- 2026-11-07 ตกลงแผน Agent ครบ 9 ขั้น (ค้น → สรุป → แก้ → ตรวจ → จำ → รายงาน), คลัง,
  Local-first/Cloud-ready, คลังกลางพร้อมตัวกรอง, model growth, .gguf — รายละเอียดใน ROADMAP.md
  ยังเป็นแผน ยังไม่ได้เริ่มเขียนโค้ด Agent ใหม่
- 2026-11-07 Phase A รอบ A1 เสร็จ: `localbrain/projindex.py` ดัชนีโปรเจกต์ระดับฟังก์ชัน (SQLite FTS5)
  Python แยกด้วย ast (ไม่รันโค้ด), JS/TS/Lua/C/C++/Go/Rust/Ruby/PHP/Shell/SQL ด้วย regex, ไฟล์ไม่มีฟังก์ชัน = module
  ชื่อตัวแปรถูกแยกคำ (check_password → check password) ให้คำสั่งภาษาอังกฤษค้นเจอ, อัปเดตเฉพาะไฟล์ที่เปลี่ยน, ข้าม node_modules/.venv/workspace
  คำสั่ง: `index-project --root .` และ `find-code "คำขอ"`; เทสต์ใหม่ 8 รายการผ่าน
  ลองกับโค้ด LocalBrain เอง: 49 ไฟล์ 301 ฟังก์ชัน ใน 0.27 วินาที ไฟล์ดัชนี ~1 MB
  ค้น "resume training from a checkpoint" เจอ train.py/checkpoint_save, "split identifier into words" เจอ projindex
  จุดอ่อน: ไฟล์เทสต์มักติดอันดับต้นเพราะมีคำซ้ำเยอะ
- 2026-11-07 Phase A รอบ A2 เสร็จ: คลังกลาง + Stack Overflow
  `localbrain/codelib.py` สร้างคลังจาก CodeSearchNet (HF parquet `code-search-net/code_search_net`; S3 เดิมปิดแล้ว 403)
  สร้างบน Colab CPU ~3 นาที: 571,542 ฟังก์ชัน (Python 441,885 / JS 129,657), ตัดทิ้ง 24,544, ไฟล์ 1.26 GB
  เก็บที่ Drive `localbrain-workspace/codelib.sqlite3` (sha1 5bb3f01e…, ตรวจตรงกัน); ไม่อยู่ใน git
  ค้น "hash a password with bcrypt" → password_hash/bcrypt_check_password, "debounce" → debounce (JS)
  CodeSearchNet ไม่บอก license รายฟังก์ชัน → เก็บ url ต้นทางทุกชิ้น ใช้เป็นตัวอย่างอ้างอิง ไม่ก๊อปทั้งก้อน
  `localbrain/stackexchange.py` (`ask-so`): ค้นคำตอบที่ยอมรับแล้ว แคช 30 วัน, เคารพ backoff, key จาก env ไม่ถูกเก็บ, CC BY-SA
  ลองจริง: error "Unicode-objects must be encoded before hashing" ได้คำตอบคะแนน 441 พร้อม `.encode('utf-8')`
  เทสต์ใหม่ 6 รายการ (รวม A1+A2 = 14 ผ่าน)
- 2026-11-07 Phase A รอบ A3 เสร็จ: แก้ไฟล์ + รันเทสต์
  `localbrain/patch.py` (`apply-patch`): บล็อก SEARCH/REPLACE หลายไฟล์, จับแบบตรงตัว (ต้องต้นบรรทัด) → ไม่สนช่องว่างท้าย
  → ไม่สนย่อหน้า (ปรับย่อหน้าให้), ต้องเจอที่เดียว, ทั้งหมดหรือไม่เลย, คืน backup ให้ย้อนกลับ, ห้าม path นอกโปรเจกต์/.git
  `localbrain/projtest.py` (`run-project-tests`): ก๊อปโปรเจกต์ไปโฟลเดอร์ชั่วคราว รัน unittest/pytest/node --test
  จำกัดเวลา + หน่วยความจำ 1 GB + priority ต่ำ, สรุปจำนวนผ่าน/ผิด + ข้อผิดแรก (ไม่ใช่ security boundary, เน็ตไม่ถูกปิด)
  runner._execute รับค่าจำกัดเป็นพารามิเตอร์ (ค่าเดิมไม่เปลี่ยน; RunnerTests เดิม 4 ข้อผ่าน)
  เดโมลูปจริง: เทสต์ 2/2 ผิด (TypeError: Strings must be encoded) → apply-patch → 2/2 ผ่าน
  เจอและแก้บั๊ก: SEARCH ไม่มีย่อหน้าเคยไปจับกลางบรรทัด; tests/ ไม่มี __init__.py ใช้ `-t .` ไม่ได้
  เทสต์ใหม่ 10 รายการ (A1–A3 รวม 24 ผ่าน)
- 2026-11-07 Phase A รอบ A4 เสร็จ → **Agent ครบ 9 ขั้น** (`localbrain/agent.py`, `experience.py`, คำสั่ง `agent`)
  ลูป: index โปรเจกต์ → เทสต์ก่อนแก้ → ค้นคลังประสบการณ์/คลังกลาง/SO → สรุป (ดึงข้อเท็จจริง ไม่ใช้โมเดล)
  → สมองเขียน SEARCH/REPLACE → เทสต์ → แย่ลงย้อนกลับ ดีขึ้นเก็บ → วนสูงสุด N รอบ → ผ่านแล้วบันทึก → รายงาน
  เก็บการแก้เฉพาะเมื่อเทสต์ผ่าน; ไม่ผ่าน = คืนโปรเจกต์ตามเดิม; โปรเจกต์ไม่มีเทสต์ = ไม่แก้ (เว้นแต่ --allow-unverified)
  สมองเสียบเปลี่ยนได้: ModelBrain (checkpoint ของเรา) / ScriptedBrain (patch ที่เตรียมไว้)
  เดโม A (patch เตรียมไว้): round 1 แย่ลง→ย้อน, round 2 ผ่าน → fixed, บันทึกลงคลังประสบการณ์
  เดโม B (**โมเดล 20M จริง**): ทั้ง 2 รอบ "no SEARCH/REPLACE blocks found" → not-fixed, โปรเจกต์ไม่ถูกแตะ
  → Agent พร้อม แต่สมองยังเขียน patch ไม่เป็น: ต้องโมเดลใหญ่ขึ้น + ข้อมูล SFT สอนเขียน patch (Phase B)
  เทสต์ใหม่ 5 รายการ (A1–A4 รวม 29 ผ่าน + RunnerTests เดิม 4)
- ข้อควรระวัง Drive: ถ้ามี folder ชื่อซ้ำ Colab จะเห็นเป็น `ชื่อ (1)` ต้อง `mv` กลับให้ชื่อตรง
- Programming Dataset v4 (ภายนอก, 2.6M แถว / 100M คำ) มีโจทย์จริงแค่ ~38 แบบ (สูตรเลข 28 + SQL)
  `scripts/import_v4.py` คัดมา 8,360 แถว (0.35M คำ) ตรวจ Python ด้วย ast + ตัวคำนวณแบบจำกัด ไม่รันโค้ดจากชุดข้อมูล
  ได้ `dataset-english-sft-v4` train 76,822 (ยังไม่ได้ใช้บน Colab)
- ทดสอบ A/B (base Colab step 12,000, SFT 500 steps, ต่างกันแค่ข้อมูล):
  มี v4 แล้วเขียนถูกภาษา (Luau, C++, SQL) แต่ logic ติดเทมเพลต เช่นบวกเลขได้ `a + b - a * b`
  TypeScript ยังออกมาเป็น JavaScript ไม่มี type → ทางแก้ที่ดีกว่าคือขยายโจทย์จริงใน tasks.py ไป Luau/TS/C++

# LocalBrain — สถานะ 2026-11-06 (เช้า)

- เพิ่ม offensive-security training content: `localbrain/extra.py` มี `offensive_collect` (รวบรวม malware analysis + attack toolkits จาก subfolder ของ markdown) และ `offensive_lessons` (บทเรียน question/answer สำหรับ malware + ATT&CK/C2/Metasploit/reverse shell/threat modeling)
- CLI ใหม่: `localbrain collect-offensive --root <folder> --output <jsonl>`; `build-anneal` รับ `--extra NAME=PATH --extra-share` สำหรับใส่ offensive source ใน final anneal mix
- tests ใหม่ `tests/test_extra.py` ผ่านทั้งหมด 36 รายการ (exit 0)

# LocalBrain — สถานะล่าสุด 2026-11-02

- เป้าหมาย: โมเดลฝึกเองที่สร้างคำตอบจากน้ำหนักจริง พร้อมเครื่องมือและคลังความรู้ในเครื่อง
- อนุมัติล่าสุด: ผู้ใช้สั่งให้เลือกวิธีและฝึกเอง ไม่ใช้ chatbot ตอบด้วยกติกาแทนโมเดล
  ดำเนินการฝึกจากศูนย์ถึง 6,000 ขั้นแล้ว ไม่มีงานฝึกค้าง ไม่มีการอัปโหลด/ใช้โมเดลสำเร็จรูป
- ผลคุณภาพ: **ยังไม่สำเร็จตามเป้าหมายผู้ช่วยทั่วไป**; คำตอบหลายข้อไม่รู้เรื่องและข้อเท็จจริงผิด
  ไม่เปลี่ยน Open-Chat.cmd ให้ใช้โมเดลที่ยังไม่ผ่าน; เก็บ checkpoint ให้ทดลอง/ศึกษาต่อ
- เครื่อง: Python 3.14.8, torch 2.14.1+cu126, GTX 1650 SUPER 4 GiB, RAM 31.83 GiB
- โมเดลใหม่ 6,895,616 พารามิเตอร์, Transformer 6 ชั้น width 256,
  context 256, vocabulary 8192, batch 8, FP32; เริ่มน้ำหนักสุ่ม ไม่ใช้ run-smoke
- ข้อมูล: Han Instruction Dataset (Wannaphong / PyThaiNLP), README ต้นทาง CC-BY-SA 4.0
  https://github.com/wannaphong/han-instruction-dataset
  commit 99c6aeda56b2dd9d0f25217208bc2560254d1bf2
  raw SHA256 8f4db6b33e0e8ac44ce647ac8be331871410ed0b02f512976cb2180f3cfc2a7e
  ดาวน์โหลดเฉพาะข้อมูล มีทั้ง human/synthetic; ไม่เรียก inference API หรือโหลด pretrained weights
- เตรียม: deduplicate คำถาม normalize; ใช้คู่ user/assistant แรก; ไม่ตัดคำตอบกลางประโยค
  คัดตัวอย่างที่ยาวเกิน context ออก; train 8308, val 499, test 515
  exact-question split ก่อนฝึก tokenizer; ยังไม่กำจัดความหมายซ้ำข้าม split ทั้งหมด
  supervised loss เฉพาะ answer/EOS; prompt และ padding ใช้ ignore_index
- ฝึกด้วย GPU; CPU เตรียม batch; CPUAdamW เก็บ moments ใน RAM
  peak PyTorch allocated VRAM 315.1 MiB ไม่ใช่หน่วยความจำรวม driver/โปรแกรมอื่น
- checkpoint: workspace/run-instruction/latest.pt และ best.pt ที่ขั้น 6000
  validation loss 8.76487 → 4.27753; held-out test loss 4.25104 ครบทั้ง 515 ตัวอย่าง
  loss ไม่ใช่ factual accuracy; ตัวอย่างคำตอบจริงไม่ผ่านการตรวจคุณภาพทั่วไป
- ผลทดสอบ: workspace/run-instruction/evaluation.json; แหล่งข้อมูล workspace/raw/SOURCE.json
- เวลาที่วัด: จุดเริ่มบันทึก 15:37:11; อ่านผลหลักเสร็จ 15:51:43 +07:00
  ช่วงงาน 14 นาที 32 วินาที รวมพัฒนาและฝึกที่ทับซ้อนกัน
  เวลาฝึกสะสมจาก checkpoint 604.34 วินาที (~10 นาที 4 วินาที); ไม่บวกซ้ำกับเวลางาน
  ดาวน์โหลดข้อมูล 15:38:52; เริ่มบันทึก metrics 15:40:33
- ตรวจ 2026-11-02: unittest 19 รายการผ่าน; pip check ผ่าน
  รวม assistant masking, split/dedup, Unicode JSONL, train/checkpoint/load/evaluate,
  model-only ไม่ใช้ greeting/help ธรรมชาติหรือ retrieval และชุดเดิม
- โหมดทดสอบน้ำหนักจริง: Open-Model.cmd หรือ
  .\.venv\Scripts\python.exe -m localbrain chat --model-only --run workspace/run-instruction
  Predictor โหลด best.pt ครั้งเดียว; natural-language ทุกคำถามเข้าโมเดลตรง
  /help และ /tool เป็นคำสั่งชัดเจนของผู้ใช้ ไม่ใช่ learned tool calling
- โหมดค้นเอกสารเดิม: Open-Chat.cmd; /help ดูคำสั่ง; /more อ่านคำตอบยาว
- Sandbox ยังปิดเพราะไม่มี Docker; ไม่ fallback ไปรันโค้ดบน host
- ข้อจำกัด/งานที่เหลือ: ต้องเพิ่มข้อมูลภาษา/ความรู้ที่มีคุณภาพและหลากหลายมากกว่านี้
  และฝึก/ประเมินต่ออย่างมีแผน; ฝึกเพิ่มด้วยข้อมูลเดิมอย่างเดียวไม่รับประกันคุณภาพ
  ยังไม่มี pretraining corpus ขนาดใหญ่, learned tool calling หรือความจำหลายรอบสนทนา
- รายละเอียดและคำสั่งครบ: README.md; รุ่น smoke เดิมยังอยู่ workspace/run-smoke/
