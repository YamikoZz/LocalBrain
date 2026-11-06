# เทรน LocalBrain บน Google Colab

`LocalBrain_Colab.ipynb` — ย้ายการเทรนขึ้น Colab T4 ฟรี (16GB VRAM) แทนคอมบ้าน (GTX 1650, 4GB).

## เตรียมครั้งเดียว
1. สร้างโฟลเดอร์ `Google Drive ▸ MyDrive ▸ localbrain-workspace/`
2. อัปโหลด dataset ที่สร้างไว้จากคอมบ้าน (เอาเฉพาะ dir ที่ tokenize แล้ว ไม่ต้องขน raw parquet):
   - `dataset-code-big/` (~615MB)
   - `dataset-anneal/` (~155MB)
   - `dataset-english-sft/` (~71MB)
3. เปิด `LocalBrain_Colab.ipynb` ใน Colab → **Runtime ▸ Change runtime type ▸ T4 GPU**
4. รัน cell 1 → 6 ตามลำดับ

## จุดสำคัญ
- **ตัดทุก ~12 ชม.** — ต่อ runtime ใหม่ รัน cell 1-5 ซ้ำ, pipeline resume จาก checkpoint ใน Drive เอง
- checkpoint เซฟลง Drive ทุก 200 step → ดาวน์โหลด `run-english-sft2/` กลับมาใช้ที่บ้านได้
- config T4: `configs/code-pretrain-colab.json` (batch 32, vram 14GB)
- จะเทรน **ต่อจากงานบ้าน** ก็ได้ — อัป `run-code-pretrain-big/` ไป Drive แล้วลบ env `LOCALBRAIN_PRETRAIN_CONFIG`
  ในโน้ตบุ๊ก (ต้องใช้ config เดิม batch 8 ไม่งั้น `--resume` จะ error เพราะ batch ไม่ตรง)

การเทรนที่บ้านไม่ต้องหยุด — Colab เป็นอีกสำเนาแยกกัน.
