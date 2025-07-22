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
const CHUNKS: usize = 4;  // チャンク数
const BUFFER_SIZE: usize = 64 * 1024 * 1024;  // 64MBバッファ
const OUTPUT_DIR: &str = "/dev/shm";  // 出力ディレクトリ（高速RAMディスク）
const META_PATH: &str = "/dev/shm/lineorder_meta.json";  // メタデータ出力先

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

    println!("=== PostgreSQL → /dev/shm 高速データ転送（順次チャンク版） ===");
    println!("並列接続数: {}", PARALLEL_CONNECTIONS);
    println!("チャンク数: {}", CHUNKS);
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

    // カラム情報を取得
    println!("\nメタデータを取得中...");
    let column_rows = client.query(
        "SELECT column_name, data_type, udt_name
         FROM information_schema.columns
         WHERE table_name = 'lineorder'
         ORDER BY ordinal_position",
        &[]
    ).await?;

    let oid_rows = client.query(
        "SELECT a.attname, t.oid
         FROM pg_attribute a
         JOIN pg_type t ON a.atttypid = t.oid
         JOIN pg_class c ON a.attrelid = c.oid
         WHERE c.relname = 'lineorder' AND a.attnum > 0
         ORDER BY a.attnum",
        &[]
    ).await?;

    let mut columns = Vec::new();
    for (col_row, oid_row) in column_rows.iter().zip(oid_rows.iter()) {
        let arrow_type = match col_row.get::<_, &str>("data_type") {
            "integer" => "int32",
            "bigint" => "int64",
            "real" => "float32",
            "double precision" => "float64",
            "numeric" | "decimal" => "decimal128",
            "character" | "char" => "string",
            "character varying" | "varchar" | "text" => "string",
            "date" => "timestamp_s",
            "timestamp" | "timestamp without time zone" => "timestamp[ns]",
            "timestamp with time zone" => "timestamp[ns, tz=UTC]",
            "boolean" => "bool",
            _ => "string"
        };

        columns.push(ColumnMeta {
            name: col_row.get("column_name"),
            data_type: col_row.get("data_type"),
            pg_oid: oid_row.get::<_, u32>("oid") as i32,
            arrow_type: arrow_type.to_string(),
        });
    }

    // 推定データサイズ
    let estimated_total_size = table_size as u64;
    let estimated_chunk_size = estimated_total_size / CHUNKS as u64;

    println!("推定データサイズ: {:.2} GB", estimated_total_size as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("チャンクあたり: {:.2} GB", estimated_chunk_size as f64 / 1024.0 / 1024.0 / 1024.0);

    let mut all_chunks_meta = Vec::new();
    let pages_per_chunk = (max_page as u32 + 1) / CHUNKS as u32;

    // チャンクを順次処理
    for chunk_id in 0..CHUNKS {
        println!("\n=== チャンク {} / {} を処理中 ===", chunk_id + 1, CHUNKS);

        let chunk_start_page = chunk_id as u32 * pages_per_chunk;
        let chunk_end_page = if chunk_id == CHUNKS - 1 {
            max_page as u32 + 1
        } else {
            (chunk_id + 1) as u32 * pages_per_chunk
        };

        let chunk_path = format!("{}/chunk_{}.bin", OUTPUT_DIR, chunk_id);
        let file = File::create(&chunk_path)?;
        file.sync_all()?;

        let chunk_file = Arc::new(file);
        let worker_offset = Arc::new(AtomicU64::new(0));
        let chunk_meta = Arc::new(Mutex::new(ChunkMeta {
            id: chunk_id,
            file: chunk_path.clone(),
            workers: Vec::new(),
        }));

        // 共有カウンタ
        let total_bytes = Arc::new(AtomicU64::new(0));
        let total_chunks = Arc::new(AtomicU64::new(0));

        let start = Instant::now();

        // 並列タスクを作成
        let mut tasks = JoinSet::new();
        let pages_per_task = (chunk_end_page - chunk_start_page) / PARALLEL_CONNECTIONS as u32;

        for worker_id in 0..PARALLEL_CONNECTIONS {
            let start_page = chunk_start_page + (worker_id as u32 * pages_per_task);
            let end_page = if worker_id == PARALLEL_CONNECTIONS - 1 {
                chunk_end_page
            } else {
                chunk_start_page + ((worker_id + 1) as u32 * pages_per_task)
            };

            let config = config.clone();
            let total_bytes_clone = Arc::clone(&total_bytes);
            let total_chunks_clone = Arc::clone(&total_chunks);
            let chunk_file_clone = Arc::clone(&chunk_file);
            let chunk_meta_clone = Arc::clone(&chunk_meta);
            let worker_offset_clone = Arc::clone(&worker_offset);

            tasks.spawn(async move {
                process_range_for_chunk(
                    config,
                    start_page,
                    end_page,
                    worker_id,
                    chunk_file_clone,
                    chunk_meta_clone,
                    worker_offset_clone,
                    total_bytes_clone,
                    total_chunks_clone
                ).await
            });
        }

        // 全タスクの完了を待つ
        let mut completed = 0;
        while let Some(result) = tasks.join_next().await {
            match result {
                Ok(Ok(())) => {
                    completed += 1;
                    if completed % 4 == 0 {
                        println!("  タスク完了: {}/{}", completed, PARALLEL_CONNECTIONS);
                    }
                }
                Ok(Err(e)) => eprintln!("  タスクエラー: {}", e),
                Err(e) => eprintln!("  JoinError: {}", e),
            }
        }

        let elapsed = start.elapsed();
        let chunk_bytes = total_bytes.load(Ordering::Relaxed);
        let chunk_speed = chunk_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;

        println!("  チャンクサイズ: {:.2} GB", chunk_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
        println!("  処理時間: {:.2}秒", elapsed.as_secs_f64());
        println!("  速度: {:.2} GB/秒", chunk_speed);

        // メタデータを保存
        all_chunks_meta.push(chunk_meta.lock().unwrap().clone());
    }

    // メタデータを出力
    let metadata = Metadata {
        columns,
        chunks: all_chunks_meta,
    };

    let meta_json = serde_json::to_string_pretty(&metadata)?;
    std::fs::write(META_PATH, meta_json)?;

    // 完了フラグファイルを作成（Python側での同期用）
    File::create("/dev/shm/lineorder_data.ready")?;

    println!("\n✅ データ転送完了");

    Ok(())
}

async fn process_range_for_chunk(
    config: Config,
    start_page: u32,
    end_page: u32,
    worker_id: usize,
    chunk_file: Arc<File>,
    chunk_meta: Arc<Mutex<ChunkMeta>>,
    worker_offset: Arc<AtomicU64>,
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
    let worker_start_offset = worker_offset.fetch_add(0, Ordering::SeqCst);
    let mut worker_bytes = 0u64;

    // COPY開始
    let copy_query = format!(
        "COPY (SELECT * FROM lineorder WHERE ctid >= '({},0)'::tid AND ctid < '({},0)'::tid) TO STDOUT (FORMAT BINARY)",
        start_page, end_page
    );

    let stream = client.copy_out(&copy_query).await?;
    futures_util::pin_mut!(stream);

    let mut _chunk_count = 0u64;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;

        write_buffer.extend_from_slice(&chunk);
        _chunk_count += 1;

        if write_buffer.len() >= BUFFER_SIZE {
            let bytes_to_write = write_buffer.len();
            let write_offset = worker_offset.fetch_add(bytes_to_write as u64, Ordering::SeqCst);

            chunk_file.write_all_at(&write_buffer, write_offset)?;

            worker_bytes += bytes_to_write as u64;
            total_bytes.fetch_add(bytes_to_write as u64, Ordering::Relaxed);
            total_chunks.fetch_add(1, Ordering::Relaxed);

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
        total_chunks.fetch_add(1, Ordering::Relaxed);
    }

    // ワーカーメタデータを更新
    {
        let mut meta = chunk_meta.lock().unwrap();
        meta.workers.push(WorkerMeta {
            id: worker_id,
            offset: worker_start_offset,
            size: worker_bytes,
        });
    }

    Ok(())
}
