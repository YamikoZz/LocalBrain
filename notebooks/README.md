# เทรน LocalBrain บน Google Colab

`LocalBrain_Colab.ipynb` ย้ายการเทรนขึ้น Colab T4 ฟรี (15 GB VRAM) แทนคอมบ้าน (GTX 1650, 4 GB)

## เตรียมครั้งเดียว
1. สร้างโฟลเดอร์ `Google Drive ▸ MyDrive ▸ localbrain-workspace/`
2. อัปโหลด dataset ที่สร้างไว้จากคอมบ้าน (เฉพาะโฟลเดอร์ที่ tokenize แล้ว ไม่ต้องขน raw parquet):
   - `dataset-code-big/` (~615 MB) สำหรับ pretrain
   - `dataset-anneal/` (~155 MB) สำหรับ anneal
   - `dataset-english-sft/` (~82 MB) สำหรับ SFT
3. เปิด `LocalBrain_Colab.ipynb` ใน Colab → **Runtime ▸ Change runtime type ▸ T4 GPU**
4. รัน cell 1 → 5 ตามลำดับ (cell 3 ต้องใช้ GitHub Personal Access Token สิทธิ์ `repo`)

## จุดสำคัญ
- **Colab ฟรีหลุดราว 3.5 ชม. ต่อ session** (เมื่อไม่มี compute units) ให้ Connect ใหม่
  แล้วรัน cell 1-5 ซ้ำ pipeline จะ resume จาก checkpoint ใน Drive เอง (เซฟทุก 200 steps)
- config: `configs/code-pretrain-colab.json` (batch 32, 24,000 steps = 393M tokens ~1.2 epoch, vram 14 GB)
- ถ้า cell 4 เตือนว่าขาด dataset หลังเพิ่งอัปหรือเปลี่ยนชื่อใน Drive ให้รอ sync สักครู่แล้วรัน cell 4 ใหม่
- เทรนเสร็จแล้วดาวน์โหลด `run-english-sft2/` (best.pt, config.json, tokenizer.json) กลับมาใช้ที่บ้านได้
- **ห้ามกด Save ในหน้า Change runtime type ระหว่างเทรน** จะรีสตาร์ทเครื่อง

## ดูสถานะ (ใช้ Terminal ของ Colab)
cell 5 เทรนค้างอยู่ cell อื่นจะรอคิว ให้ดูสถานะผ่าน Terminal แทน

```bash
# GPU + step ล่าสุด
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader && tail -2 /content/localbrain/workspace/run-code-pretrain-big/metrics.jsonl
# เครื่องทั้งหมด: GPU / CPU / RAM / Disk
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv && nproc && free -h && df -h /content
# ข้อมูล SFT ใน Drive ตัวไหน (ตัวล่าสุดต้องได้ train 69763)
grep -o '"train": *[0-9]*' /content/localbrain/workspace/dataset-english-sft/meta.json | head -1
```

ใช้ `tail -f` ดูสดได้ กด Ctrl+C เพื่อหยุดดู (การเทรนไม่หยุดตาม)

## เปลี่ยนข้อมูล SFT ระหว่างเทรน
pipeline อ่าน `dataset-english-sft` ตอนเริ่มขั้น SFT จึงเปลี่ยนได้ก่อนถึงขั้นนั้น
1. ใน Drive เปลี่ยนชื่อตัวเดิมเป็น `dataset-english-sft-old` ก่อน
2. อัปโฟลเดอร์ใหม่ แล้วเปลี่ยนชื่อเป็น `dataset-english-sft`
3. ถ้าเคยมีชื่อซ้ำกัน Colab อาจเห็นเป็น `dataset-english-sft (1)` ให้แก้ใน Terminal:
   ```bash
   mv "/content/localbrain/workspace/dataset-english-sft (1)" /content/localbrain/workspace/dataset-english-sft
   ```
ทุก dataset ต้องสร้างด้วย tokenizer เดิม (sha256 `ff5386c5...`)

## เทรนต่อจากงานบนคอมบ้าน
อัป `run-code-pretrain-big/` ไป Drive แล้วลบบรรทัด `LOCALBRAIN_PRETRAIN_CONFIG` ในโน้ตบุ๊ก
ต้องใช้ config เดิม (batch 8) เพราะ `--resume` ไม่ยอมให้ batch ต่างจาก checkpoint

## Colab CLI
`google-colab-cli` ล็อกอินได้ แต่ runtime ที่เปิดจากเบราว์เซอร์จะแสดงเป็น `[?]` (orphan)
และสั่ง `colab exec` เข้าไม่ได้ ต้องสร้าง session ด้วย `colab new` ซึ่ง GPU ต้องมี compute units
