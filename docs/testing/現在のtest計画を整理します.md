現在のtest計画を整理します。
<機能概要>
1. 現在の処理は大きく4つに分かれている。
   1. Postgresからpostgres_raw_binaryを/dev/shmにあるキューにためていく。
   2. キューから取り出し、kvikioを利用してGPUメモリに転送する。
   3. GPUメモリ上のpostgres_raw_binaryからrow offsetとfield indicesを作成する。そのindexを利用してpostgres_raw_binaryからcolumn_arrowsを作成する。column_arrows作成は固定長のarrow配列を作成する処理と、可変長を作成する処理がある。
   4. column_arrowをcudfを利用してcudf dataframeに変換し、cudfのto_parquetを利用してparquetファイルとして出力する。

2. データ型は以下の通り。
https://claude.ai/public/artifacts/340a85c2-37f1-4fe4-990f-e436943c4877
## PostgreSQL型 → Arrow型 → cuDF型のマッピング
| PostgreSQL型 | Arrow型 | cuDF型 | 備考 |
|-------------|---------|---------|------|
| **数値型** |
| SMALLINT | int16 | int16 | 16ビット整数 |
| INTEGER | int32 | int32 | 32ビット整数 |
| BIGINT | int64 | int64 | 64ビット整数 |
| REAL | float32 | float32 | 単精度浮動小数点 |
| DOUBLE PRECISION | float64 | float64 | 倍精度浮動小数点 |
| NUMERIC/DECIMAL | string | string | 文字列として読み取り（精度の損失を防ぐため） |
| **文字列型** |
| TEXT | string | object/string | UTF-8文字列 |
| VARCHAR(n) | string | object/string | 可変長文字列 |
| CHAR(n) | string | object/string | 固定長文字列 |
| **バイナリ型** |
| BYTEA | binary | ListDtype(uint8) | バイナリデータをuint8の配列として表現 |
| **日付・時刻型** |
| DATE | date32 | datetime64[ms] | 日付（cuDFでは日単位精度はサポートされず、ミリ秒精度で扱う） |
| TIME | time64[us] | (部分サポート) | 時刻（cuDFは主にdatetimeを使用） |
| TIMESTAMP | timestamp[us] | datetime64[us] | タイムスタンプ（マイクロ秒精度） |
| TIMESTAMP WITH TIME ZONE | timestamp[us, tz=UTC] | datetime64[us] | cuDFはタイムゾーンを内部的に扱わない |
| INTERVAL | duration | timedelta64 | 期間 |
| **論理型** |
| BOOLEAN | bool | bool8 | 真偽値（cuDFは1バイト/値、Arrowはビットマップ） |
| **配列型** |
| ARRAY | list | ListDtype | 配列（1次元配列のみ完全サポート） |
| **複合型** |
| COMPOSITE TYPE | struct | StructDtype | ユーザー定義複合型 |
| **ネットワーク型** |
| INET | string または binary | object/string | IPアドレス |
| MACADDR | string または binary | object/string | MACアドレス |
| **その他の型** |
| UUID | fixed_size_binary[16] | ListDtype(uint8) | UUID（16バイトの固定長バイナリとして） |
| JSON/JSONB | string | object/string | JSON（文字列として） |
| XML | string | object/string | XML（文字列として） |


2. 上記の全ての型について、機能1-4.毎のE2Eテストを実装する。
  テストデータ:
    - 通常値: 100行
    - NULL値: 各型で10%含む
    - 境界値: 最小値、最大値を1行ずつ
   検証: 値の完全一致
