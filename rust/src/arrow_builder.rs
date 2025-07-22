use anyhow::{Result, Context};
use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;
use crate::cuda::{CudaContext, CudaBuffer};

pub struct ArrowStringBuilder {
    offsets: Vec<i32>,
    data: Vec<u8>,
    current_offset: i32,
}

pub struct GpuArrowBuffers {
    pub offsets_gpu: CudaBuffer,
    pub data_gpu: CudaBuffer,
    pub row_count: usize,
}

impl ArrowStringBuilder {
    pub fn new() -> Self {
        Self {
            offsets: vec![0], // 最初のオフセットは常に0
            data: Vec::new(),
            current_offset: 0,
        }
    }

    pub fn add_string(&mut self, s: &[u8]) {
        self.data.extend_from_slice(s);
        self.current_offset += s.len() as i32;
        self.offsets.push(self.current_offset);
    }

    pub fn add_null(&mut self) {
        // NULLの場合もオフセットは進めない
        self.offsets.push(self.current_offset);
    }

    pub fn transfer_to_gpu(self, cuda_ctx: &CudaContext) -> Result<GpuArrowBuffers> {
        let row_count = self.offsets.len() - 1;

        // オフセット配列をGPUに転送
        let offsets_bytes: Vec<u8> = self.offsets.iter()
            .flat_map(|&offset| offset.to_le_bytes())
            .collect();
        let offsets_gpu = cuda_ctx.allocate(offsets_bytes.len())?;
        cuda_ctx.copy_to_device(&offsets_bytes, &offsets_gpu)?;

        // データ配列をGPUに転送
        let data_gpu = if !self.data.is_empty() {
            let buf = cuda_ctx.allocate(self.data.len())?;
            cuda_ctx.copy_to_device(&self.data, &buf)?;
            buf
        } else {
            cuda_ctx.allocate(1)? // 空の場合でも最小1バイト確保
        };

        Ok(GpuArrowBuffers {
            offsets_gpu,
            data_gpu,
            row_count,
        })
    }
}

pub fn parse_postgres_copy_binary_header(data: &[u8]) -> Result<usize> {
    if data.len() < 11 {
        anyhow::bail!("Data too short for PostgreSQL COPY BINARY header");
    }

    // シグネチャチェック
    let expected_sig = b"PGCOPY\n\xff\r\n\0";
    if &data[0..11] != expected_sig {
        anyhow::bail!("Invalid PostgreSQL COPY BINARY signature");
    }

    let mut cursor = Cursor::new(&data[11..]);

    // フラグフィールド（4バイト）
    let _flags = cursor.read_u32::<BigEndian>()
        .context("Failed to read flags")?;

    // 拡張領域長（4バイト）
    let ext_len = cursor.read_u32::<BigEndian>()
        .context("Failed to read extension length")? as usize;

    // ヘッダーサイズ = 基本(11) + フラグ(4) + 拡張長フィールド(4) + 拡張領域
    Ok(11 + 4 + 4 + ext_len)
}

pub fn parse_postgres_row(data: &[u8], column_count: usize) -> Result<Vec<Option<&[u8]>>> {
    let mut cursor = Cursor::new(data);

    // フィールド数
    let field_count = cursor.read_u16::<BigEndian>()
        .context("Failed to read field count")? as usize;

    if field_count != column_count {
        anyhow::bail!("Field count mismatch: expected {}, got {}", column_count, field_count);
    }

    let mut fields = Vec::with_capacity(field_count);

    for _ in 0..field_count {
        let field_len = cursor.read_i32::<BigEndian>()
            .context("Failed to read field length")?;

        if field_len == -1 {
            // NULL値
            fields.push(None);
        } else {
            let pos = cursor.position() as usize;
            let field_data = &data[pos..pos + field_len as usize];
            cursor.set_position((pos + field_len as usize) as u64);
            fields.push(Some(field_data));
        }
    }

    Ok(fields)
}
