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

const PARALLEL_CONNECTIONS: usize = 16;  // 並列接続数
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
struct ChunkMeta {
    id: usize,
    file: String,
    workers: Vec<WorkerMeta>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct WorkerMeta {
    id: usize,
    offset: u64,
    size: u64,
}

#[derive(Serialize, Deserialize, Debug)]
struct Metadata {
    columns: Vec<ColumnMeta>,
    chunks: Vec<ChunkMeta>,
}

#[tokio::main(worker_threads = 32)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dsn = std::env::var("GPUPASER_PG_DSN")?;
    
    // 環境変数から設定を読み込み
    let chunks: usize = std::env::var("RUST_CHUNKS")
        .unwrap_or_else(|_| "4".to_string())
        .parse()?;
    let chunk_offset: usize = std::env::var("RUST_CHUNK_OFFSET")
        .unwrap_or_else(|_| "0".to_string())
        .parse()?;
    
    let meta_path = format!("{}/lineorder_meta_{}.json", OUTPUT_DIR, chunk_offset);
    
    println!("=== PostgreSQL → /dev/shm 高速データ転送（環境変数対応版） ===");
    println!("並列接続数: {}", PARALLEL_CONNECTIONS);
    println!("チャンク数: {}", chunks);
    println!("チャンクオフセット: {}", chunk_offset);
    println!("バッファサイズ: {} MB", BUFFER_SIZE / 1024 / 1024);
    println!("出力ディレクトリ: {}", OUTPUT_DIR);
    
    // コンフィグを作成
    let config: Config = dsn.parse()?;
    
    // メタデータを取得
    let (client, connection) = config.connect(NoTls).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });
    
    // テーブルのページ数とサイズを取得
    let row = client.query_one(
        "SELECT relpages, pg_relation_size('lineorder') FROM pg_class WHERE relname='lineorder'", 
        &[]
    ).await?;
    let max_page: i32 = row.get(0);
    let table_size: i64 = row.get(1);
    let max_page = max_page as u32;
    
    println!("テーブルサイズ: {:.2} GB", table_size as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("最大ページ番号: {}", max_page);
    
    // メタデータを取得（最初のバッチのみ）
    let columns = if chunk_offset == 0 {
        println!("\nメタデータを取得中...");
        let meta_rows = client.query(
            "SELECT 
                a.attname AS name,
                t.typname AS data_type,
                t.oid AS pg_oid
            FROM pg_attribute a
            JOIN pg_type t ON a.atttypid = t.oid
            WHERE a.attrelid = 'lineorder'::regclass
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum",
            &[]
        ).await?;
        
        let columns: Vec<_> = meta_rows.iter().map(|row| {
            let data_type: &str = row.get("data_type");
            let arrow_type = match data_type {
                "int4" => "int32",
                "int8" => "int64",
                "float4" => "float32",
                "float8" => "float64",
                "numeric" => "decimal128",
                "bpchar" | "char" => "string",
                "varchar" | "text" => "string",
                "date" => "timestamp_s",
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
        }).collect();
        
        columns
    } else {
        // 2回目以降は前回のメタデータを読み込む
        let prev_meta_path = format!("{}/lineorder_meta_0.json", OUTPUT_DIR);
        let meta_json = std::fs::read_to_string(&prev_meta_path)?;
        let prev_meta: Metadata = serde_json::from_str(&meta_json)?;
        prev_meta.columns
    };
    
    // チャンクファイルを事前作成（サイズ確保なし - 動的に拡張）
    let mut chunk_files = Vec::new();
    let mut chunk_metas = Vec::new();
    
    for i in 0..chunks {
        let chunk_id = chunk_offset + i;
        let chunk_path = format!("{}/chunk_{}.bin", OUTPUT_DIR, chunk_id);
        let file = File::create(&chunk_path)?;
        file.sync_all()?;
        
        chunk_files.push(Arc::new(file));
        chunk_metas.push(ChunkMeta {
            id: chunk_id,
            file: chunk_path,
            workers: Vec::new(),
        });
        
        println!("チャンクファイル作成: chunk_{}.bin", chunk_id);
    }
    
    // 共有カウンタ
    let total_bytes = Arc::new(AtomicU64::new(0));
    let total_chunks = Arc::new(AtomicU64::new(0));
    
    // チャンクファイルとワーカーオフセットを共有
    let chunk_files = Arc::new(chunk_files);
    let chunk_metas = Arc::new(Mutex::new(chunk_metas));
    let worker_offsets: Arc<Vec<AtomicU64>> = Arc::new(
        (0..chunks).map(|_| AtomicU64::new(0)).collect()
    );
    
    let start = Instant::now();
    
    // 並列タスクを作成
    let mut tasks = JoinSet::new();
    let total_chunks_global = 4;  // 全体のチャンク数は4で固定
    let pages_per_task = max_page / (PARALLEL_CONNECTIONS * total_chunks_global) as u32;
    
    for worker_id in 0..PARALLEL_CONNECTIONS {
        for chunk_idx in 0..chunks {
            let chunk_id = chunk_offset + chunk_idx;
            let global_worker_id = worker_id * total_chunks_global + chunk_id;
            
            let start_page = global_worker_id as u32 * pages_per_task;
            let end_page = if worker_id == PARALLEL_CONNECTIONS - 1 && chunk_idx == chunks - 1 {
                max_page + 1
            } else {
                (global_worker_id + 1) as u32 * pages_per_task
            };
            
            let config = config.clone();
            let total_bytes_clone = Arc::clone(&total_bytes);
            let total_chunks_clone = Arc::clone(&total_chunks);
            let chunk_files_clone = Arc::clone(&chunk_files);
            let chunk_metas_clone = Arc::clone(&chunk_metas);
            let worker_offsets_clone = Arc::clone(&worker_offsets);
            
            tasks.spawn(async move {
                process_range_parallel(
                    config,
                    start_page,
                    end_page,
                    worker_id,
                    chunk_idx,
                    chunk_files_clone,
                    chunk_metas_clone,
                    worker_offsets_clone,
                    total_bytes_clone,
                    total_chunks_clone
                ).await
            });
        }
    }
    
    // 進捗表示タスク
    let total_bytes_monitor = Arc::clone(&total_bytes);
    let start_time = start.clone();
    let monitor_handle = tokio::spawn(async move {
        let mut last_bytes = 0u64;
        let mut max_speed = 0.0f64;
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            let current_bytes = total_bytes_monitor.load(Ordering::Relaxed);
            let elapsed = start_time.elapsed();
            let speed = current_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;
            let delta = current_bytes - last_bytes;
            let delta_speed = delta as f64 / 1024.0 / 1024.0 / 1024.0;
            
            if delta_speed > max_speed {
                max_speed = delta_speed;
            }
            
            println!("進捗: {:.2} GB | 速度: {:.2} GB/秒 | 瞬間: {:.2} GB/秒 | 最大: {:.2} GB/秒", 
                     current_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                     speed,
                     delta_speed,
                     max_speed);
            last_bytes = current_bytes;
        }
    });
    
    // 全タスクの完了を待つ
    let mut completed = 0;
    let total_tasks = PARALLEL_CONNECTIONS * chunks;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => {
                completed += 1;
                if completed % 4 == 0 || completed == total_tasks {
                    println!("タスク完了: {}/{}", completed, total_tasks);
                }
            }
            Ok(Err(e)) => eprintln!("タスクエラー: {}", e),
            Err(e) => eprintln!("JoinError: {}", e),
        }
    }
    
    monitor_handle.abort();
    
    let elapsed = start.elapsed();
    let final_bytes = total_bytes.load(Ordering::Relaxed);
    let speed_gbps = final_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;
    
    println!("\n=== 転送完了 ===");
    println!("転送データ: {:.2} GB", final_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("転送時間: {:.2}秒", elapsed.as_secs_f64());
    println!("平均速度: {:.2} GB/秒", speed_gbps);
    
    // メタデータを出力
    let metadata = Metadata {
        columns,
        chunks: chunk_metas.lock().unwrap().clone(),
    };
    
    let meta_json = serde_json::to_string_pretty(&metadata)?;
    std::fs::write(&meta_path, meta_json)?;
    println!("\nメタデータ保存: {}", meta_path);
    
    // 完了フラグファイルを作成
    let ready_file = format!("{}/lineorder_data_{}.ready", OUTPUT_DIR, chunk_offset);
    File::create(&ready_file)?;
    
    Ok(())
}

async fn process_range_parallel(
    config: Config,
    start_page: u32,
    end_page: u32,
    worker_id: usize,
    chunk_idx: usize,
    chunk_files: Arc<Vec<Arc<File>>>,
    chunk_metas: Arc<Mutex<Vec<ChunkMeta>>>,
    worker_offsets: Arc<Vec<AtomicU64>>,
    total_bytes: Arc<AtomicU64>,
    total_chunks: Arc<AtomicU64>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (client, connection) = config.connect(NoTls).await?;
    
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });
    
    // ワーカー用のバッファ
    let mut write_buffer = Vec::with_capacity(BUFFER_SIZE);
    let worker_start_offset = worker_offsets[chunk_idx].load(Ordering::SeqCst);
    let mut worker_bytes = 0u64;
    
    // COPY開始
    let copy_query = format!(
        "COPY (SELECT * FROM lineorder WHERE ctid >= '({},0)'::tid AND ctid < '({},0)'::tid) TO STDOUT (FORMAT BINARY)",
        start_page, end_page
    );
    
    let stream = client.copy_out(&copy_query).await?;
    futures_util::pin_mut!(stream);
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        
        write_buffer.extend_from_slice(&chunk);
        
        if write_buffer.len() >= BUFFER_SIZE {
            let bytes_to_write = write_buffer.len();
            let write_offset = worker_offsets[chunk_idx].fetch_add(bytes_to_write as u64, Ordering::SeqCst);
            
            chunk_files[chunk_idx].write_all_at(&write_buffer, write_offset)?;
            
            worker_bytes += bytes_to_write as u64;
            total_bytes.fetch_add(bytes_to_write as u64, Ordering::Relaxed);
            total_chunks.fetch_add(1, Ordering::Relaxed);
            
            write_buffer.clear();
        }
    }
    
    // 残りのデータを書き込み
    if !write_buffer.is_empty() {
        let bytes_to_write = write_buffer.len();
        let write_offset = worker_offsets[chunk_idx].fetch_add(bytes_to_write as u64, Ordering::SeqCst);
        
        chunk_files[chunk_idx].write_all_at(&write_buffer, write_offset)?;
        
        worker_bytes += bytes_to_write as u64;
        total_bytes.fetch_add(bytes_to_write as u64, Ordering::Relaxed);
        total_chunks.fetch_add(1, Ordering::Relaxed);
    }
    
    // ワーカーメタデータを更新
    {
        let mut metas = chunk_metas.lock().unwrap();
        metas[chunk_idx].workers.push(WorkerMeta {
            id: worker_id,
            offset: worker_start_offset,
            size: worker_bytes,
        });
    }
    
    Ok(())
}