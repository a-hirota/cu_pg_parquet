use std::time::Instant;
use tokio_postgres::{NoTls, Config};
use futures_util::StreamExt;
use tokio::task::JoinSet;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::fs::{File, OpenOptions};
use std::io::Write;
use serde::{Serialize, Deserialize};
use serde_json;
use std::os::unix::fs::FileExt;

const PARALLEL_CONNECTIONS: usize = 16;  // 並列接続数
const CHUNKS: usize = 8;  // チャンク数（各約6.6GB - GPUメモリに最適）
const BUFFER_SIZE: usize = 64 * 1024 * 1024;  // 64MBバッファ
const OUTPUT_DIR: &str = "/dev/shm";  // 出力ディレクトリ（高速RAMディスク）
const META_PATH: &str = "/dev/shm/lineorder_meta.json";  // メタデータ出力先
// const ESTIMATED_ROWS: u64 = 600_003_000;  // lineorderテーブルの実際の行数

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
    let mut config: Config = dsn.parse()?;

    // PostgreSQL接続設定の最適化
    config.tcp_user_timeout(std::time::Duration::from_secs(60));

    println!("=== PostgreSQL → /dev/shm 高速データ転送 ===");
    println!("並列接続数: {}", PARALLEL_CONNECTIONS);
    println!("チャンク数: {}", CHUNKS);
    println!("バッファサイズ: {} MB", BUFFER_SIZE / 1024 / 1024);
    println!("出力ディレクトリ: {}", OUTPUT_DIR);

    // テーブル情報を取得
    let (client, connection) = config.connect(NoTls).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("Connection error: {}", e);
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

    // 推定データサイズを計算（実際のデータサイズ基準）
    let estimated_total_size = table_size as u64;  // テーブルサイズを使用
    let estimated_chunk_size = estimated_total_size / CHUNKS as u64;
    println!("推定データサイズ: {:.2} GB", estimated_total_size as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("チャンクあたり: {:.2} GB", estimated_chunk_size as f64 / 1024.0 / 1024.0 / 1024.0);

    // メタデータを取得してJSON保存
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

    let mut columns: Vec<ColumnMeta> = Vec::new();
    for row in meta_rows {
        let name: String = row.get(0);
        let data_type: String = row.get(1);
        let pg_oid: u32 = row.get(2);
        let pg_oid = pg_oid as i32;  // JSON用にi32に変換

        // PostgreSQL型からArrow型へのマッピング（簡易版）
        let arrow_type = match data_type.as_str() {
            "int2" => "int16",
            "int4" => "int32",
            "int8" => "int64",
            "float4" => "float32",
            "float8" => "float64",
            "numeric" => "decimal128",
            "text" | "varchar" | "char" | "bpchar" => "utf8",
            "date" => "timestamp_s",
            "timestamp" | "timestamptz" => "timestamp[us]",
            "bool" => "bool",
            _ => "unknown"
        }.to_string();

        columns.push(ColumnMeta {
            name,
            data_type,
            pg_oid,
            arrow_type,
        });
    }

    // チャンクファイルを事前作成（サイズ確保なし - 動的に拡張）
    let mut chunk_files = Vec::new();
    let mut chunk_metas = Vec::new();

    for chunk_id in 0..CHUNKS {
        let chunk_path = format!("{}/chunk_{}.bin", OUTPUT_DIR, chunk_id);
        let file = File::create(&chunk_path)?;
        // file.set_len()を削除 - 動的にサイズ拡張
        file.sync_all()?;

        chunk_files.push(file);
        chunk_metas.push(ChunkMeta {
            id: chunk_id,
            file: chunk_path.clone(),
            workers: Vec::new(),
        });

        println!("チャンクファイル作成: {} ({:.2} GB)", chunk_path,
                 estimated_chunk_size as f64 / 1024.0 / 1024.0 / 1024.0);
    }

    // 共有カウンタ
    let total_bytes = Arc::new(AtomicU64::new(0));
    let total_chunks = Arc::new(AtomicU64::new(0));

    // チャンクファイルとワーカーオフセットを共有
    let chunk_files = Arc::new(chunk_files);
    let chunk_metas = Arc::new(Mutex::new(chunk_metas));
    let worker_offsets: Arc<Vec<AtomicU64>> = Arc::new(
        (0..CHUNKS).map(|_| AtomicU64::new(0)).collect()
    );

    let start = Instant::now();

    // 並列タスクを作成
    let mut tasks = JoinSet::new();
    let pages_per_task = max_page / PARALLEL_CONNECTIONS as u32;

    for worker_id in 0..PARALLEL_CONNECTIONS {
        let start_page = worker_id as u32 * pages_per_task;
        let end_page = if worker_id == PARALLEL_CONNECTIONS - 1 {
            max_page + 1
        } else {
            (worker_id + 1) as u32 * pages_per_task
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
                chunk_files_clone,
                chunk_metas_clone,
                worker_offsets_clone,
                total_bytes_clone,
                total_chunks_clone
            ).await
        });
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

            println!("進捗: {:.2} GB ({:.1}%) | 速度: {:.2} GB/秒 | 瞬間: {:.2} GB/秒 | 最大: {:.2} GB/秒",
                     current_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                     (current_bytes as f64 / table_size as f64) * 100.0,
                     speed,
                     delta_speed,
                     max_speed);
            last_bytes = current_bytes;
        }
    });

    // 全タスクの完了を待つ
    let mut completed = 0;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => {
                completed += 1;
                println!("タスク完了: {}/{}", completed, PARALLEL_CONNECTIONS);
            }
            Ok(Err(e)) => eprintln!("タスクエラー: {}", e),
            Err(e) => eprintln!("JoinError: {}", e),
        }
    }

    monitor_handle.abort();

    let elapsed = start.elapsed();
    let final_bytes = total_bytes.load(Ordering::Relaxed);
    let final_chunks = total_chunks.load(Ordering::Relaxed);
    let speed_gbps = final_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;

    // メタデータを保存
    let metadata = Metadata {
        columns,
        chunks: chunk_metas.lock().unwrap().clone(),
    };
    let meta_json = serde_json::to_string_pretty(&metadata)?;
    let mut meta_file = File::create(META_PATH)?;
    meta_file.write_all(meta_json.as_bytes())?;
    println!("\nメタデータを保存: {}", META_PATH);

    // 各チャンクファイルを実際のサイズにトリム
    for (chunk_id, offset) in worker_offsets.iter().enumerate() {
        let actual_size = offset.load(Ordering::Relaxed);
        let chunk_path = format!("{}/chunk_{}.bin", OUTPUT_DIR, chunk_id);
        let file = OpenOptions::new().write(true).open(&chunk_path)?;
        file.set_len(actual_size)?;
        println!("チャンク{}: {:.2} GB", chunk_id, actual_size as f64 / 1024.0 / 1024.0 / 1024.0);
    }

    println!("\n=== 結果 ===");
    println!("時間: {:.2}秒", elapsed.as_secs_f64());
    println!("データサイズ: {} bytes ({:.2} GB)", final_bytes, final_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("チャンク数: {}", final_chunks);
    println!("平均チャンクサイズ: {:.2} KB", final_bytes as f64 / final_chunks as f64 / 1024.0);
    println!("スループット: {:.2} MB/秒", final_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0);
    println!("読み取り速度: {:.2} GB/秒", speed_gbps);

    // 完了フラグファイルを作成（Python側での同期用）
    File::create("/dev/shm/lineorder_data.ready")?;

    Ok(())
}

async fn process_range_parallel(
    config: Config,
    start_page: u32,
    end_page: u32,
    worker_id: usize,
    chunk_files: Arc<Vec<File>>,
    chunk_metas: Arc<Mutex<Vec<ChunkMeta>>>,
    worker_offsets: Arc<Vec<AtomicU64>>,
    total_bytes: Arc<AtomicU64>,
    total_chunks: Arc<AtomicU64>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (client, connection) = config.connect(NoTls).await?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("Connection error in worker {}: {}", worker_id, e);
        }
    });

    let query = format!(
        "COPY (SELECT * FROM lineorder WHERE ctid >= '({},1)'::tid AND ctid < '({},1)'::tid) TO STDOUT WITH (FORMAT BINARY)",
        start_page, end_page
    );

    let stream = client.copy_out(&query).await?;

    // チャンクごとのバッファを用意
    let mut chunk_buffers: Vec<Vec<u8>> = (0..CHUNKS)
        .map(|_| Vec::with_capacity(BUFFER_SIZE))
        .collect();

    // 各チャンクへの最初の書き込みを追跡
    let mut first_write_to_chunk: Vec<bool> = vec![true; CHUNKS];

    let mut worker_bytes = 0u64;
    let mut worker_chunks = 0u64;
    let mut current_chunk = 0;

    tokio::pin!(stream);

    // ストリーミング処理（PostgreSQL COPY BINARYフォーマットをそのまま保持）
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let chunk_size = chunk.len() as u64;

        // 現在のチャンクIDを決定（ラウンドロビン）
        let chunk_id = current_chunk % CHUNKS;
        current_chunk += 1;

        // このチャンクへの最初の書き込みの場合、0xFFFFを追加
        if first_write_to_chunk[chunk_id] {
            chunk_buffers[chunk_id].push(0xFF);
            chunk_buffers[chunk_id].push(0xFF);
            first_write_to_chunk[chunk_id] = false;
        }

        // チャンクバッファに追加
        chunk_buffers[chunk_id].extend_from_slice(&chunk);

        // バッファが一定サイズに達したら書き込み
        if chunk_buffers[chunk_id].len() >= BUFFER_SIZE {
            let offset = worker_offsets[chunk_id].fetch_add(
                chunk_buffers[chunk_id].len() as u64,
                Ordering::Relaxed
            );

            // write_all_atで並列書き込み（ロック不要）
            chunk_files[chunk_id].write_all_at(&chunk_buffers[chunk_id], offset)?;

            // ワーカーメタデータを更新
            {
                let mut metas = chunk_metas.lock().unwrap();
                metas[chunk_id].workers.push(WorkerMeta {
                    id: worker_id,
                    offset,
                    size: chunk_buffers[chunk_id].len() as u64,
                });
            }

            chunk_buffers[chunk_id].clear();
        }

        worker_bytes += chunk_size;
        worker_chunks += 1;

        // 共有カウンタを更新
        if worker_chunks % 100 == 0 {
            total_bytes.fetch_add(worker_bytes, Ordering::Relaxed);
            total_chunks.fetch_add(worker_chunks, Ordering::Relaxed);
            worker_bytes = 0;
            worker_chunks = 0;
        }
    }

    // 最後の残りバッファをすべて書き込み
    for chunk_id in 0..CHUNKS {
        if !chunk_buffers[chunk_id].is_empty() {
            // このチャンクへの最初の書き込みの場合、0xFFFFを追加
            if first_write_to_chunk[chunk_id] {
                // バッファの先頭に0xFFFFを挿入
                let mut new_buffer = Vec::with_capacity(chunk_buffers[chunk_id].len() + 2);
                new_buffer.push(0xFF);
                new_buffer.push(0xFF);
                new_buffer.extend_from_slice(&chunk_buffers[chunk_id]);
                chunk_buffers[chunk_id] = new_buffer;
                first_write_to_chunk[chunk_id] = false;
            }

            let offset = worker_offsets[chunk_id].fetch_add(
                chunk_buffers[chunk_id].len() as u64,
                Ordering::Relaxed
            );

            chunk_files[chunk_id].write_all_at(&chunk_buffers[chunk_id], offset)?;

            // ワーカーメタデータを更新
            {
                let mut metas = chunk_metas.lock().unwrap();
                metas[chunk_id].workers.push(WorkerMeta {
                    id: worker_id,
                    offset,
                    size: chunk_buffers[chunk_id].len() as u64,
                });
            }
        }
    }

    // 残りのカウントを更新
    if worker_bytes > 0 {
        total_bytes.fetch_add(worker_bytes, Ordering::Relaxed);
        total_chunks.fetch_add(worker_chunks, Ordering::Relaxed);
    }

    println!("Worker {}: 完了", worker_id);

    Ok(())
}
