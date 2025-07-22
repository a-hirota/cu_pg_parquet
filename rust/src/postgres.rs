use anyhow::{Result, Context};
use bytes::BytesMut;
use tokio_postgres::{Client, Config, NoTls};
use std::sync::Arc;

pub struct PostgresConnection {
    client: Arc<Client>,
}

impl PostgresConnection {
    pub async fn new(dsn: &str) -> Result<Self> {
        let config = dsn.parse::<Config>()
            .context("Failed to parse DSN")?;

        let (client, connection) = config.connect(NoTls).await
            .context("Failed to connect to PostgreSQL")?;

        // コネクションをバックグラウンドで管理
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("Connection error: {}", e);
            }
        });

        Ok(Self {
            client: Arc::new(client),
        })
    }

    pub async fn copy_binary(&self, query: &str) -> Result<Vec<u8>> {
        let copy_out_stream = self.client
            .copy_out(query)
            .await
            .context("Failed to execute COPY command")?;

        let mut buffer = BytesMut::new();
        let reader = Box::pin(copy_out_stream);

        use futures_util::StreamExt;
        let mut reader = reader;
        while let Some(chunk) = reader.next().await {
            let chunk = chunk?;
            buffer.extend_from_slice(&chunk);
        }

        Ok(buffer.freeze().to_vec())
    }

    pub async fn copy_binary_streaming<F>(&self, query: &str, mut callback: F) -> Result<()>
    where
        F: FnMut(&[u8]) -> Result<()>,
    {
        let copy_out_stream = self.client
            .copy_out(query)
            .await
            .context("Failed to execute COPY command")?;

        let reader = Box::pin(copy_out_stream);

        use futures_util::StreamExt;
        let mut reader = reader;
        while let Some(chunk) = reader.next().await {
            let chunk = chunk?;
            callback(&chunk)?;
        }

        Ok(())
    }
}
