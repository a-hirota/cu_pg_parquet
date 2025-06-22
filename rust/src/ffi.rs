use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::runtime::Runtime;
use anyhow::Result;

use crate::postgres::PostgresConnection;
use crate::cuda::CudaContext;
use crate::arrow_builder::{ArrowStringBuilder, parse_postgres_copy_binary_header, parse_postgres_row};

/// PostgreSQLからCOPY BINARYデータを取得し、GPUに転送する
/// 
/// Returns: dict with keys:
/// - "data_ptr": GPU device pointer (u64)
/// - "data_size": データサイズ (usize)
/// - "offsets_ptr": オフセット配列のGPUポインタ (u64) 
/// - "offsets_size": オフセット配列サイズ (usize)
/// - "row_count": 行数 (usize)
#[pyfunction]
#[pyo3(signature = (dsn, query, column_index=0))]
pub fn fetch_postgres_copy_binary_to_gpu(
    py: Python<'_>,
    dsn: String,
    query: String,
    column_index: usize,
) -> PyResult<Bound<'_, PyDict>> {
    let rt = Runtime::new().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let result = rt.block_on(async {
        fetch_and_transfer_internal(dsn, query, column_index).await
    }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let dict = PyDict::new_bound(py);
    dict.set_item("data_ptr", result.data_ptr)?;
    dict.set_item("data_size", result.data_size)?;
    dict.set_item("offsets_ptr", result.offsets_ptr)?;
    dict.set_item("offsets_size", result.offsets_size)?;
    dict.set_item("row_count", result.row_count)?;
    
    Ok(dict)
}

/// 既存のNumba実装と連携するための簡易版
/// バイナリデータ全体をGPUに転送し、ポインタを返す
#[pyfunction]
pub fn transfer_to_gpu_numba<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let cuda_ctx = CudaContext::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let gpu_buffer = cuda_ctx.allocate(data.len())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    cuda_ctx.copy_to_device(data, &gpu_buffer)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    cuda_ctx.synchronize()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let dict = PyDict::new_bound(py);
    dict.set_item("device_ptr", gpu_buffer.device_ptr())?;
    dict.set_item("size", gpu_buffer.size())?;
    
    // GPU bufferの所有権をPython側に渡す（メモリリークを防ぐため後でfreeが必要）
    std::mem::forget(gpu_buffer);
    
    Ok(dict)
}

struct TransferResult {
    data_ptr: u64,
    data_size: usize,
    offsets_ptr: u64,
    offsets_size: usize,
    row_count: usize,
}

async fn fetch_and_transfer_internal(
    dsn: String,
    query: String,
    column_index: usize,
) -> Result<TransferResult> {
    // PostgreSQL接続
    let conn = PostgresConnection::new(&dsn).await?;
    
    // COPY BINARYデータ取得
    let binary_data = conn.copy_binary(&query).await?;
    
    // ヘッダー解析
    let header_size = parse_postgres_copy_binary_header(&binary_data)?;
    let mut pos = header_size;
    
    // CUDAコンテキスト初期化
    let cuda_ctx = CudaContext::new()?;
    
    // Arrow形式のstring builderを作成
    let mut string_builder = ArrowStringBuilder::new();
    
    // 行ごとに解析
    while pos < binary_data.len() - 2 { // 最後の2バイトは終端マーカー
        if &binary_data[pos..pos+2] == b"\xff\xff" {
            // 終端マーカー
            break;
        }
        
        // 行の解析
        let row_data = &binary_data[pos..];
        let fields = parse_postgres_row(row_data, 1)?; // 仮に1列とする
        
        if column_index < fields.len() {
            match fields[column_index] {
                Some(field_data) => {
                    string_builder.add_string(field_data);
                },
                None => {
                    string_builder.add_null();
                }
            }
        }
        
        // 次の行へ
        // TODO: 実際の行サイズを計算する必要がある
        pos += row_data.len(); // 簡易実装
    }
    
    // GPUに転送
    let gpu_buffers = string_builder.transfer_to_gpu(&cuda_ctx)?;
    
    let result = TransferResult {
        data_ptr: gpu_buffers.data_gpu.device_ptr(),
        data_size: gpu_buffers.data_gpu.size(),
        offsets_ptr: gpu_buffers.offsets_gpu.device_ptr(),
        offsets_size: gpu_buffers.offsets_gpu.size(),
        row_count: gpu_buffers.row_count,
    };
    
    // GPU bufferの所有権をPython側に渡す
    std::mem::forget(gpu_buffers);
    
    Ok(result)
}