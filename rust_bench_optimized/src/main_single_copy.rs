use std::time::Instant;
use tokio_postgres::{NoTls, Config};
use futures_util::StreamExt;
use std::fs::File;
use std::io::Write;

const OUTPUT_DIR: &str = "/dev/shm";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dsn = std::env::var("GPUPASER_PG_DSN")?;
    let table_name = std::env::var("TABLE_NAME").unwrap_or_else(|_| "customer".to_string());
    
    println!("=== PostgreSQL 一括COPY実験 ===");
    println!("テーブル: {}", table_name);
    
    // コンフィグを作成
    let config: Config = dsn.parse()?;
    
    // 接続
    let (client, connection) = config.connect(NoTls).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });
    
    // まず行数を確認
    let row = client.query_one(&format!("SELECT COUNT(*) FROM {}", table_name), &[]).await?;
    let expected_rows: i64 = row.get(0);
    println!("期待される行数: {}", expected_rows);
    
    // 出力ファイル
    let output_path = format!("{}/{}_single_copy.bin", OUTPUT_DIR, table_name);
    let mut file = File::create(&output_path)?;
    
    let start = Instant::now();
    let mut total_bytes = 0u64;
    
    // 一括COPYを実行（チャンク分割なし）
    println!("COPY開始...");
    let copy_query = format!("COPY {} TO STDOUT (FORMAT BINARY)", table_name);
    
    let stream = client.copy_out(&copy_query).await?;
    futures_util::pin_mut!(stream);
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        total_bytes += chunk.len() as u64;
        file.write_all(&chunk)?;
    }
    
    file.sync_all()?;
    
    let elapsed = start.elapsed();
    let speed_gbps = total_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;
    
    println!("\n=== 結果 ===");
    println!("ファイル: {}", output_path);
    println!("サイズ: {:.2} GB", total_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("処理時間: {:.2}秒", elapsed.as_secs_f64());
    println!("速度: {:.2} GB/秒", speed_gbps);
    println!("期待される行数: {}", expected_rows);
    
    Ok(())
}