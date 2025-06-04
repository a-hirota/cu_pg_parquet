# PostgreSQL GPU Parserの処理フロー

## Parquet出力機能の処理フロー図

```mermaid
flowchart TD
    %% メインコンポーネント
    PostgreSQL[PostgreSQL データベース]
    PgConn[PgConnector]
    BinaryParser[BinaryDataParser]
    MemoryManager[GPUMemoryManager]
    GPUDecoder[GPUDecoder]
    OutputHandler[OutputHandler]
    ParquetWriter[ParquetWriter]
    
    %% データフロー
    PostgreSQL --> |COPY バイナリデータ| PgConn
    PgConn --> |バイナリバッファ| PgProcessor
    
    subgraph PgProcessor[PgGpuProcessor]
        direction TB
        LoopStart(チャンク処理ループ開始)
        ParseChunk[チャンクパース]
        InitBuffer[GPUバッファ初期化]
        Decode[GPUデコード]
        CollectResults[結果集約]
        CheckComplete{全行処理完了?}
        LoopEnd(ループ終了)
        
        LoopStart --> ParseChunk
        ParseChunk --> InitBuffer
        InitBuffer --> Decode
        Decode --> CollectResults
        CollectResults --> CheckComplete
        CheckComplete -->|No| LoopStart
        CheckComplete -->|Yes| LoopEnd
    end
    
    BinaryParser -.->|使用| ParseChunk
    MemoryManager -.->|使用| InitBuffer
    GPUDecoder -.->|使用| Decode
    
    PgProcessor --> |チャンク結果| OutputHandler
    OutputHandler --> |データ| ParquetWriter
    
    %% Parquet出力関連
    subgraph ParquetOutput[Parquet出力処理]
        PyArrow[PyArrow Table]
        ParquetFile[Parquetファイル]
        CudfVerify[cuDF検証]
        
        PyArrow --> |write_table| ParquetFile
        ParquetFile --> |読み込み| CudfVerify
    end
    
    ParquetWriter --> PyArrow
    
    %% スタイル定義
    classDef database fill:#f9f,stroke:#333,stroke-width:2px;
    classDef processor fill:#bbf,stroke:#333,stroke-width:1px;
    classDef output fill:#bfb,stroke:#333,stroke-width:1px;
    
    class PostgreSQL database;
    class PgProcessor,BinaryParser,MemoryManager,GPUDecoder processor;
    class OutputHandler,ParquetWriter,ParquetOutput output;
```

## 大規模テーブル処理フロー

```mermaid
flowchart LR
    %% データソースとチャンク処理
    PostgreSQL[(PostgreSQL\nDB)] --> |SQL| ChunkQuery[チャンククエリ\nLIMIT/OFFSET]
    
    subgraph ChunkProcessor[チャンク処理]
        direction TB
        ChunkQuery --> PandasDF[Pandas\nDataFrame]
        PandasDF --> CudfDF[cuDF\nDataFrame]
        CudfDF --> PyArrowTable[PyArrow\nTable]
    end
    
    subgraph ParquetWriter[Parquetファイル書き込み]
        direction TB
        FirstChunk{最初の\nチャンク?}
        CreateFile[新規ファイル\n作成]
        AppendData[データ\n追記]
        
        FirstChunk -->|Yes| CreateFile
        FirstChunk -->|No| AppendData
    end
    
    ChunkProcessor --> PyArrowTable
    PyArrowTable --> ParquetWriter
    
    %% 結果検証
    ParquetWriter --> |書き込み完了| ParquetFile[(Parquet\nファイル)]
    ParquetFile --> Verify[PyArrow/cuDFで検証]
    
    %% スタイル定義
    classDef database fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef output fill:#bfb,stroke:#333,stroke-width:1px;
    
    class PostgreSQL,ParquetFile database;
    class ChunkProcessor process;
    class ParquetWriter output;
```

## モジュール間の関係図

```mermaid
classDiagram
    PgGpuProcessor "1" --> "1" PgConnector : 使用
    PgGpuProcessor "1" --> "1" BinaryDataParser : 使用
    PgGpuProcessor "1" --> "1" GPUMemoryManager : 使用
    PgGpuProcessor "1" --> "1" GPUDecoder : 使用
    PgGpuProcessor "1" --> "1" OutputHandler : 使用
    OutputHandler "1" --> "0..1" ParquetWriter : 使用(オプション)
    
    class PgGpuProcessor {
        +process_table(table_name, limit)
        -conn
        -memory_manager
        -parser
        -gpu_decoder
        -output_handler
        -parquet_output
    }
    
    class PgConnector {
        +connect_to_postgres()
        +get_binary_data(conn, table_name, limit)
    }
    
    class BinaryDataParser {
        +parse_chunk(chunk_data, max_chunk_size, num_columns, start_row, max_rows)
    }
    
    class GPUMemoryManager {
        +initialize_device_buffers(columns, row_count)
        +transfer_to_device(buffer, dtype)
        +cleanup_buffers(buffers)
    }
    
    class GPUDecoder {
        +decode_chunk(buffers, chunk_array, field_offsets, field_lengths, rows_in_chunk, columns)
    }
    
    class OutputHandler {
        +process_chunk_result(chunk_result)
        +print_summary(limit)
        +close()
    }
    
    class ParquetWriter {
        +write_chunk(chunk_data)
        +close()
    }
```

## 実装別処理フロー比較

```mermaid
flowchart TD
    subgraph GPU処理パイプライン統合版
        direction LR
        PG1[(PostgreSQL)] --> Binary1[バイナリ\nCOPY] --> GPUParser[GPUパース] --> OutputHandler --> Parquet1[(Parquet)]
    end
    
    subgraph シンプル版
        direction LR
        PG2[(PostgreSQL)] --> CSV[CSV出力] --> Pandas[Pandas\nDataFrame] --> cuDF[cuDF\nDataFrame] --> Parquet2[(Parquet)]
    end
    
    subgraph 大規模処理版
        direction LR
        PG3[(PostgreSQL)] --> Chunks[複数チャンク\nSQL] --> CuDFMulti[複数cuDF\nDataFrame] --> PyArrow[PyArrow\nテーブル] --> Parquet3[(Parquet)]
    end
    
    %% スタイル
    classDef database fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    
    class PG1,PG2,PG3,Parquet1,Parquet2,Parquet3 database;
    class GPUParser,OutputHandler,CSV,Pandas,cuDF,Chunks,CuDFMulti,PyArrow process;
```

## チャンク処理とParquet出力の詳細フロー

```mermaid
sequenceDiagram
    participant Postgres as PostgreSQL
    participant Processor as PgGpuProcessor
    participant Parser as BinaryParser
    participant GPU as GPUDecoder
    participant Output as OutputHandler
    participant Writer as ParquetWriter
    participant PA as PyArrow
    
    Processor->>Postgres: 接続とテーブル情報取得
    Postgres-->>Processor: テーブル情報
    
    Processor->>Postgres: バイナリデータ取得
    Postgres-->>Processor: バイナリデータ
    
    loop チャンク処理
        Processor->>Parser: チャンクパース
        Parser-->>Processor: 解析結果(フィールドオフセット等)
        
        Processor->>GPU: GPUデコード
        GPU-->>Processor: チャンク結果
        
        Processor->>Output: チャンク結果処理
        Output->>Writer: チャンクデータ書き込み
        Writer->>PA: PyArrow Tableに変換
        PA->>Writer: テーブル
        
        alt 最初のチャンク
            Writer->>Writer: 新規ファイル作成
        else 2回目以降
            Writer->>Writer: 追記処理
        end
    end
    
    Processor->>Output: 最終結果処理
    Output->>Writer: クローズ
    
    Processor->>Processor: 結果検証
    Processor-->>PA: Parquetファイル読み込み
    PA-->>Processor: テーブル
