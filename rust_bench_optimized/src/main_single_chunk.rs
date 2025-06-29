use std::time::Instant;
use tokio_postgres::{NoTls, Config};
use futures_util::StreamExt;
use tokio::task::JoinSet;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::fs::File;
use serde::{Serialize, Deserialize};
use serde_json;
use std::os::unix::fs::FileExt;

// 環境変数から並列数を取得、デフォルトは16
const DEFAULT_PARALLEL_CONNECTIONS: usize = 16;
const BUFFER_SIZE: usize = 64 * 1024 * 1024;  // 64MBバッファ
const OUTPUT_DIR: &str = "/dev/shm";  // 出力ディレクトリ（高速RAMディスク）

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ColumnMeta {
    name: String,
    data_type: String,
    pg_oid: i32,
    arrow_type: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct WorkerMeta {
    id: usize,
    offset: u64,
    size: u64,
}

#[derive(Serialize, Deserialize, Debug)]
struct ChunkResult {
    columns: Vec<ColumnMeta>,
    chunk_id: usize,
    chunk_file: String,
    workers: Vec<WorkerMeta>,
    total_bytes: u64,
    elapsed_seconds: f64,
}

#[tokio::main(worker_threads = 32)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dsn = std::env::var("GPUPASER_PG_DSN")?;
    let chunk_id: usize = std::env::var("CHUNK_ID")?.parse()?;
    let total_chunks: usize = std::env::var("TOTAL_CHUNKS")?.parse()?;
    let table_name = std::env::var("TABLE_NAME").unwrap_or_else(|_| "lineorder".to_string());
    
    // 環境変数から並列数を取得
    let parallel_connections = std::env::var("RUST_PARALLEL_CONNECTIONS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PARALLEL_CONNECTIONS);
    
    println!("=== PostgreSQL → /dev/shm 単一チャンク転送 ===");
    println!("チャンク: {} / {}", chunk_id + 1, total_chunks);
    println!("並列接続数: {}", parallel_connections);
    println!("バッファサイズ: {} MB", BUFFER_SIZE / 1024 / 1024);
    
    // コンフィグを作成
    let config: Config = dsn.parse()?;
    
    // メタデータを取得
    let (client, connection) = config.connect(NoTls).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });
    
    // テーブルのページ数を取得（正確な値を取得）
    let row = client.query_one(
        &format!("SELECT (pg_relation_size('{}'::regclass) / current_setting('block_size')::int)::int", table_name), 
        &[]
    ).await?;
    let max_page: i32 = row.get(0);
    let max_page = max_page as u32;
    
    // デバッグ情報（テストモード時のみ）
    let is_test_mode = std::env::var("GPUPGPARSER_TEST_MODE").unwrap_or_default() == "1";
    if is_test_mode {
        println!("チャンク{}: 総ページ数: {}", chunk_id, max_page);
    }
    
    // カラム情報を取得（最初のチャンクのみ）
    let columns = if chunk_id == 0 {
        println!("メタデータを取得中...");
        let meta_rows = client.query(
            &format!("SELECT 
                a.attname AS name,
                t.typname AS data_type,
                t.oid AS pg_oid
            FROM pg_attribute a
            JOIN pg_type t ON a.atttypid = t.oid
            WHERE a.attrelid = '{}'::regclass
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum", table_name),
            &[]
        ).await?;
        
        meta_rows.iter().map(|row| {
            let data_type: &str = row.get("data_type");
            let arrow_type = match data_type {
                "int4" => "int32",
                "int8" => "int64",
                "float4" => "float32",
                "float8" => "float64",
                "numeric" => "decimal128",
                "bpchar" | "char" => "string",
                "varchar" | "text" => "string",
                "date" => "date32",
                "timestamp" => "timestamp[ns]",
                "timestamptz" => "timestamp[ns, tz=UTC]",
                "bool" => "bool",
                _ => "string"
            };
            
            ColumnMeta {
                name: row.get("name"),
                data_type: row.get("data_type"),
                pg_oid: row.get::<_, u32>("pg_oid") as i32,
                arrow_type: arrow_type.to_string(),
            }
        }).collect()
    } else {
        Vec::new()
    };
    
    // チャンクの範囲を計算
    let pages_per_chunk = (max_page + 1) / total_chunks as u32;
    let chunk_start_page = chunk_id as u32 * pages_per_chunk;
    let chunk_end_page = if chunk_id == total_chunks - 1 {
        max_page + 1
    } else {
        (chunk_id + 1) as u32 * pages_per_chunk
    };
    
    if is_test_mode {
        println!("ページ範囲: {} - {} (各チャンク: {}ページ)", chunk_start_page, chunk_end_page, pages_per_chunk);
        
        // 実際の行数を事前に確認（デバッグ用）
        if chunk_id == 0 {
            let count_row = client.query_one(
                &format!("SELECT COUNT(*) FROM {} WHERE ctid >= '({},1)'::tid AND ctid < '({},1)'::tid", 
                    table_name, chunk_start_page, chunk_end_page),
                &[]
            ).await?;
            let expected_rows: i64 = count_row.get(0);
            println!("チャンク0の推定行数: {}", expected_rows);
        }
    } else {
        println!("ページ範囲: {} - {}", chunk_start_page, chunk_end_page);
    }
    
    // チャンクファイルを作成
    let chunk_path = format!("{}/chunk_{}.bin", OUTPUT_DIR, chunk_id);
    let file = File::create(&chunk_path)?;
    file.sync_all()?;
    
    let chunk_file = Arc::new(file);
    let worker_offset = Arc::new(AtomicU64::new(0));
    let workers = Arc::new(Mutex::new(Vec::new()));
    
    // 共有カウンタ
    let total_bytes = Arc::new(AtomicU64::new(0));
    
    let start = Instant::now();
    
    // 並列タスクを作成
    let mut tasks = JoinSet::new();
    let pages_per_task = (chunk_end_page - chunk_start_page) / parallel_connections as u32;
    
    for worker_id in 0..parallel_connections {
        let start_page = chunk_start_page + (worker_id as u32 * pages_per_task);
        let end_page = if worker_id == parallel_connections - 1 {
            chunk_end_page
        } else {
            chunk_start_page + ((worker_id + 1) as u32 * pages_per_task)
        };
        
        let config = config.clone();
        let total_bytes_clone = Arc::clone(&total_bytes);
        let chunk_file_clone = Arc::clone(&chunk_file);
        let workers_clone = Arc::clone(&workers);
        let worker_offset_clone = Arc::clone(&worker_offset);
        let table_name_clone = table_name.clone();
        
        tasks.spawn(async move {
            process_range(
                config,
                start_page,
                end_page,
                worker_id,
                chunk_file_clone,
                workers_clone,
                worker_offset_clone,
                total_bytes_clone,
                table_name_clone
            ).await
        });
    }
    
    // 全タスクの完了を待つ
    let mut completed = 0;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => {
                completed += 1;
                if completed % 4 == 0 || completed == parallel_connections {
                    println!("タスク完了: {}/{}", completed, parallel_connections);
                }
            }
            Ok(Err(e)) => eprintln!("タスクエラー: {}", e),
            Err(e) => eprintln!("JoinError: {}", e),
        }
    }
    
    let elapsed = start.elapsed();
    let final_bytes = total_bytes.load(Ordering::Relaxed);
    let speed_gbps = final_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;
    
    println!("チャンクサイズ: {:.2} GB", final_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("処理時間: {:.2}秒", elapsed.as_secs_f64());
    println!("速度: {:.2} GB/秒", speed_gbps);
    
    // 結果をJSON出力
    let result = ChunkResult {
        columns,
        chunk_id,
        chunk_file: chunk_path,
        workers: workers.lock().unwrap().clone(),
        total_bytes: final_bytes,
        elapsed_seconds: elapsed.as_secs_f64(),
    };
    
    let result_json = serde_json::to_string(&result)?;
    println!("\n===CHUNK_RESULT_JSON===");
    println!("{}", result_json);
    println!("===END_CHUNK_RESULT_JSON===");
    
    Ok(())
}

async fn process_range(
    config: Config,
    start_page: u32,
    end_page: u32,
    worker_id: usize,
    chunk_file: Arc<File>,
    workers: Arc<Mutex<Vec<WorkerMeta>>>,
    worker_offset: Arc<AtomicU64>,
    total_bytes: Arc<AtomicU64>,
    table_name: String,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (client, connection) = config.connect(NoTls).await?;
    
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });
    
    // ワーカー用のバッファ
    let mut write_buffer = Vec::with_capacity(BUFFER_SIZE);
    let worker_start_offset = worker_offset.load(Ordering::SeqCst);
    let mut worker_bytes = 0u64;
    
    // COPY開始
    let copy_query = format!(
        "COPY (SELECT * FROM {} WHERE ctid >= '({},1)'::tid AND ctid < '({},1)'::tid) TO STDOUT (FORMAT BINARY)",
        table_name, start_page, end_page
    );
    
    let stream = client.copy_out(&copy_query).await?;
    futures_util::pin_mut!(stream);
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        
        write_buffer.extend_from_slice(&chunk);
        
        if write_buffer.len() >= BUFFER_SIZE {
            let bytes_to_write = write_buffer.len();
            let write_offset = worker_offset.fetch_add(bytes_to_write as u64, Ordering::SeqCst);
            
            chunk_file.write_all_at(&write_buffer, write_offset)?;
            
            worker_bytes += bytes_to_write as u64;
            total_bytes.fetch_add(bytes_to_write as u64, Ordering::Relaxed);
            
            write_buffer.clear();
        }
    }
    
    // 残りのデータを書き込み
    if !write_buffer.is_empty() {
        let bytes_to_write = write_buffer.len();
        let write_offset = worker_offset.fetch_add(bytes_to_write as u64, Ordering::SeqCst);
        
        chunk_file.write_all_at(&write_buffer, write_offset)?;
        
        worker_bytes += bytes_to_write as u64;
        total_bytes.fetch_add(bytes_to_write as u64, Ordering::Relaxed);
    }
    
    // ワーカーメタデータを保存
    {
        let mut w = workers.lock().unwrap();
        w.push(WorkerMeta {
            id: worker_id,
            offset: worker_start_offset,
            size: worker_bytes,
        });
    }
    
    Ok(())
}