use std::time::Instant;
use tokio_postgres::{NoTls, Config};
use futures_util::StreamExt;
use tokio::task::JoinSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const PARALLEL_CONNECTIONS: usize = 16;  // 並列接続数
const CHUNKS_PER_CONNECTION: usize = 4;  // 各接続のチャンク数

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dsn = std::env::var("GPUPASER_PG_DSN")?;
    let config: Config = dsn.parse()?;
    
    println!("=== PostgreSQL 並列COPY BINARY ベンチマーク ===");
    println!("並列接続数: {}", PARALLEL_CONNECTIONS);
    println!("チャンク数/接続: {}", CHUNKS_PER_CONNECTION);
    println!("総タスク数: {}", PARALLEL_CONNECTIONS * CHUNKS_PER_CONNECTION);
    
    // まずテーブル情報を取得
    let (client, connection) = config.connect(NoTls).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("Connection error: {}", e);
        }
    });
    
    // テーブルのページ数を取得
    let row = client.query_one("SELECT relpages FROM pg_class WHERE relname='lineorder'", &[]).await?;
    let max_page: i32 = row.get(0);
    let max_page = max_page as u32;
    
    println!("最大ページ番号: {}", max_page);
    
    // 共有カウンタ
    let total_bytes = Arc::new(AtomicU64::new(0));
    let total_chunks = Arc::new(AtomicU64::new(0));
    
    let start = Instant::now();
    
    // 並列タスクを作成
    let mut tasks = JoinSet::new();
    let pages_per_task = max_page / (PARALLEL_CONNECTIONS * CHUNKS_PER_CONNECTION) as u32;
    
    for conn_idx in 0..PARALLEL_CONNECTIONS {
        for chunk_idx in 0..CHUNKS_PER_CONNECTION {
            let task_idx = conn_idx * CHUNKS_PER_CONNECTION + chunk_idx;
            let start_page = task_idx as u32 * pages_per_task;
            let end_page = if task_idx == PARALLEL_CONNECTIONS * CHUNKS_PER_CONNECTION - 1 {
                max_page + 1
            } else {
                (task_idx + 1) as u32 * pages_per_task
            };
            
            let config = config.clone();
            let total_bytes_clone = Arc::clone(&total_bytes);
            let total_chunks_clone = Arc::clone(&total_chunks);
            
            tasks.spawn(async move {
                process_range(
                    config,
                    start_page,
                    end_page,
                    task_idx,
                    total_bytes_clone,
                    total_chunks_clone
                ).await
            });
        }
    }
    
    // 進捗表示タスク
    let total_bytes_monitor = Arc::clone(&total_bytes);
    let monitor_handle = tokio::spawn(async move {
        let mut last_bytes = 0u64;
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            let current_bytes = total_bytes_monitor.load(Ordering::Relaxed);
            let elapsed = start.elapsed();
            let speed = current_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;
            let delta = current_bytes - last_bytes;
            let delta_speed = delta as f64 / 1024.0 / 1024.0 / 1024.0;
            println!("進捗: {:.2} GB 読み取り済み | 現在速度: {:.2} GB/秒 | 瞬間速度: {:.2} GB/秒", 
                     current_bytes as f64 / 1024.0 / 1024.0 / 1024.0, 
                     speed,
                     delta_speed);
            last_bytes = current_bytes;
        }
    });
    
    // 全タスクの完了を待つ
    while let Some(result) = tasks.join_next().await {
        if let Err(e) = result {
            eprintln!("タスクエラー: {}", e);
        }
    }
    
    monitor_handle.abort();
    
    let elapsed = start.elapsed();
    let final_bytes = total_bytes.load(Ordering::Relaxed);
    let final_chunks = total_chunks.load(Ordering::Relaxed);
    let speed_gbps = final_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;
    
    println!("\n=== 結果 ===");
    println!("時間: {:.2}秒", elapsed.as_secs_f64());
    println!("データサイズ: {} bytes ({:.2} GB)", final_bytes, final_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("チャンク数: {}", final_chunks);
    println!("平均チャンクサイズ: {} KB", final_bytes / final_chunks / 1024);
    
    println!("\n===== 速度結果 =====");
    println!("読み取り速度: {:.2} GB/秒", speed_gbps);
    println!("目標達成率: {:.1}%", (speed_gbps / 7.0) * 100.0);
    
    if speed_gbps < 7.0 {
        println!("\n目標未達成。改善が必要: あと {:.2} GB/秒", 7.0 - speed_gbps);
        println!("\n改善案:");
        println!("- より大きなバッファサイズを使用");
        println!("- PostgreSQLのshared_buffersを増やす");
        println!("- ネットワーク設定の最適化");
    } else {
        println!("\n✓ 目標達成！");
    }
    
    Ok(())
}

async fn process_range(
    config: Config,
    start_page: u32,
    end_page: u32,
    task_idx: usize,
    total_bytes: Arc<AtomicU64>,
    total_chunks: Arc<AtomicU64>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (client, connection) = config.connect(NoTls).await?;
    
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("Connection error in task {}: {}", task_idx, e);
        }
    });
    
    let query = format!(
        "COPY (SELECT * FROM lineorder WHERE ctid >= '({},0)'::tid AND ctid < '({},0)'::tid) TO STDOUT WITH (FORMAT BINARY)",
        start_page, end_page
    );
    
    let stream = client.copy_out(&query).await?;
    
    let mut task_bytes = 0u64;
    let mut _task_chunks = 0u64;
    
    tokio::pin!(stream);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let chunk_size = chunk.len() as u64;
        task_bytes += chunk_size;
        _task_chunks += 1;
        
        // 共有カウンタを更新
        total_bytes.fetch_add(chunk_size, Ordering::Relaxed);
        total_chunks.fetch_add(1, Ordering::Relaxed);
    }
    
    println!("タスク {} 完了: {:.2} MB", task_idx, task_bytes as f64 / 1024.0 / 1024.0);
    
    Ok(())
}