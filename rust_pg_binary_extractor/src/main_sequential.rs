use std::time::Instant;
use tokio_postgres::{NoTls, Config};
use futures_util::StreamExt;
use std::fs::File;
use std::io::Write;
use serde::{Serialize, Deserialize};
use serde_json;

const BUFFER_SIZE: usize = 64 * 1024 * 1024;  // 64MBバッファ
const OUTPUT_DIR: &str = "/dev/shm";  // 出力ディレクトリ（高速RAMディスク）

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ColumnMeta {
    name: String,
    data_type: String,
    pg_oid: i32,
    arrow_type: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ChunkResult {
    columns: Vec<ColumnMeta>,
    chunk_id: usize,
    chunk_file: String,
    total_bytes: u64,
    elapsed_seconds: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dsn = std::env::var("GPUPASER_PG_DSN")?;
    let chunk_id: usize = std::env::var("CHUNK_ID")?.parse()?;
    let total_chunks: usize = std::env::var("TOTAL_CHUNKS")?.parse()?;
    
    println!("=== PostgreSQL → /dev/shm シーケンシャル転送 ===");
    println!("チャンク: {} / {}", chunk_id + 1, total_chunks);
    println!("バッファサイズ: {} MB", BUFFER_SIZE / 1024 / 1024);
    
    let config: Config = dsn.parse()?;
    let (client, connection) = config.connect(NoTls).await?;
    
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });
    
    // メタデータ取得
    let rows = client.query("SELECT * FROM lineorder LIMIT 0", &[]).await?;
    let columns: Vec<ColumnMeta> = if !rows.is_empty() {
        rows[0].columns().iter().map(|col| {
            ColumnMeta {
                name: col.name().to_string(),
                data_type: col.type_().name().to_string(),
                pg_oid: col.type_().oid() as i32,
                arrow_type: map_to_arrow_type(col.type_().oid()),
            }
        }).collect()
    } else {
        // 別の方法でメタデータを取得
        let stmt = client.prepare("SELECT * FROM lineorder LIMIT 0").await?;
        stmt.columns().iter().map(|col| {
            ColumnMeta {
                name: col.name().to_string(),
                data_type: col.type_().name().to_string(),
                pg_oid: col.type_().oid() as i32,
                arrow_type: map_to_arrow_type(col.type_().oid()),
            }
        }).collect()
    };
    
    // ctid範囲計算
    let page_count_query = "SELECT relpages FROM pg_class WHERE relname = 'lineorder'";
    let page_count_row = client.query_one(page_count_query, &[]).await?;
    let total_pages: i32 = page_count_row.get(0);
    
    let pages_per_chunk = total_pages as u32 / total_chunks as u32;
    let chunk_start_page = chunk_id as u32 * pages_per_chunk;
    let chunk_end_page = if chunk_id == total_chunks - 1 {
        let max_page_query = "SELECT (ctid::text::point)[0]::int FROM lineorder ORDER BY ctid DESC LIMIT 1";
        let max_page_row = client.query_one(max_page_query, &[]).await?;
        let max_page: i32 = max_page_row.get(0);
        max_page as u32 + 1
    } else {
        (chunk_id + 1) as u32 * pages_per_chunk
    };
    
    println!("ページ範囲: {} - {}", chunk_start_page, chunk_end_page);
    
    // チャンクファイルを作成
    let chunk_path = format!("{}/chunk_{}.bin", OUTPUT_DIR, chunk_id);
    let mut file = File::create(&chunk_path)?;
    
    let start = Instant::now();
    let mut total_bytes = 0u64;
    let mut write_buffer = Vec::with_capacity(BUFFER_SIZE);
    
    // COPY開始（シーケンシャル処理）
    let copy_query = format!(
        "COPY (SELECT * FROM lineorder WHERE ctid >= '({},0)'::tid AND ctid < '({},0)'::tid) TO STDOUT (FORMAT BINARY)",
        chunk_start_page, chunk_end_page
    );
    
    let stream = client.copy_out(&copy_query).await?;
    futures_util::pin_mut!(stream);
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        
        write_buffer.extend_from_slice(&chunk);
        
        if write_buffer.len() >= BUFFER_SIZE {
            // シーケンシャルに書き込み
            file.write_all(&write_buffer)?;
            total_bytes += write_buffer.len() as u64;
            write_buffer.clear();
        }
    }
    
    // 残りのバッファを書き込み
    if !write_buffer.is_empty() {
        file.write_all(&write_buffer)?;
        total_bytes += write_buffer.len() as u64;
    }
    
    file.sync_all()?;
    
    let elapsed = start.elapsed();
    let speed_gbps = total_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;
    
    println!("チャンクサイズ: {:.2} GB", total_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("処理時間: {:.2}秒", elapsed.as_secs_f64());
    println!("速度: {:.2} GB/秒", speed_gbps);
    
    // 結果をJSON出力
    let result = ChunkResult {
        columns,
        chunk_id,
        chunk_file: chunk_path,
        total_bytes,
        elapsed_seconds: elapsed.as_secs_f64(),
    };
    
    let result_json = serde_json::to_string(&result)?;
    println!("\n===CHUNK_RESULT_JSON===");
    println!("{}", result_json);
    println!("===END_CHUNK_RESULT_JSON===");
    
    Ok(())
}

fn map_to_arrow_type(oid: u32) -> String {
    match oid {
        701 => "float8",
        23 => "int4",
        1700 => "decimal",
        25 => "text",
        1043 => "varchar",
        1082 => "timestamp_s",  // date型をtimestamp_sとして扱う
        _ => "unknown",
    }.to_string()
}