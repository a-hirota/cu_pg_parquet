#!/usr/bin/env python3
"""Rust実装を使った高速COPY BINARYベンチマーク"""
import os
import time
import subprocess

def create_rust_benchmark():
    """高速COPY用のRustプログラムを作成"""
    rust_code = '''use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tokio_postgres::{NoTls, Config};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dsn = std::env::var("GPUPASER_PG_DSN")?;
    let config: Config = dsn.parse()?;
    
    println!("Connecting to PostgreSQL...");
    let (client, connection) = config.connect(NoTls).await?;
    
    // Connection handler
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("Connection error: {}", e);
        }
    });
    
    // Output file
    let output_path = "/dev/shm/lineorder_rust.bin";
    let mut file = File::create(output_path)?;
    
    println!("Starting COPY BINARY...");
    let start = Instant::now();
    
    // Execute COPY
    let stream = client
        .copy_out("COPY lineorder TO STDOUT WITH (FORMAT BINARY)")
        .await?;
    
    let mut total_bytes = 0u64;
    let mut chunks = 0u64;
    
    // Stream processing
    tokio::pin!(stream);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        total_bytes += chunk.len() as u64;
        chunks += 1;
        
        if chunks % 1000 == 0 {
            print!("\\rProcessed: {:.2} GB", total_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
            std::io::stdout().flush()?;
        }
    }
    
    let elapsed = start.elapsed();
    let speed_gbps = total_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0 / 1024.0;
    
    println!("\\n\\nCOPY完了:");
    println!("  時間: {:.2}秒", elapsed.as_secs_f64());
    println!("  データサイズ: {} bytes ({:.2} GB)", total_bytes, total_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("  チャンク数: {}", chunks);
    println!("  読み取り速度: {:.2} GB/秒", speed_gbps);
    
    Ok(())
}'''
    
    # Cargo.tomlを作成
    cargo_toml = '''[package]
name = "pg_fast_copy"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1", features = ["full"] }
tokio-postgres = "0.7"
futures-util = "0.3"'''
    
    # ディレクトリ作成
    os.makedirs("/home/ubuntu/gpupgparser/rust_bench", exist_ok=True)
    
    # ファイル書き込み
    with open("/home/ubuntu/gpupgparser/rust_bench/Cargo.toml", "w") as f:
        f.write(cargo_toml)
    
    os.makedirs("/home/ubuntu/gpupgparser/rust_bench/src", exist_ok=True)
    with open("/home/ubuntu/gpupgparser/rust_bench/src/main.rs", "w") as f:
        f.write(rust_code)
    
    print("Rustプログラム作成完了")

def build_and_run_rust():
    """Rustプログラムをビルドして実行"""
    os.chdir("/home/ubuntu/gpupgparser/rust_bench")
    
    # ビルド
    print("Rustプログラムをビルド中...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ビルドエラー: {result.stderr}")
        return
    
    print("ビルド完了")
    
    # 実行
    print("\n=== Rust高速COPY実行 ===")
    env = os.environ.copy()
    env["GPUPASER_PG_DSN"] = "dbname=postgres user=postgres host=localhost port=5432"
    
    subprocess.run(
        ["./target/release/pg_fast_copy"],
        env=env
    )

def main():
    print("PostgreSQL高速読み取りベンチマーク（Rust版）")
    print("目標: 7GB/秒")
    print("=" * 60)
    
    # Rustがインストールされているか確認
    result = subprocess.run(["which", "cargo"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: Rustがインストールされていません")
        return
    
    create_rust_benchmark()
    build_and_run_rust()

if __name__ == "__main__":
    main()