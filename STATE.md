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
