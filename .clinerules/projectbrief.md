
# PostgreSQLデータのGPU直接デシリアライズとcuDF格納パイプラインの設計

## 背景と目的

PostgreSQLから大量のデータを取得してGPU上のデータフレーム（RAPIDS cuDF）に格納する際、**CPUでのシリアライズ/デシリアライズ処理がボトルネック**になっています。従来はPyArrowやADBCを使い、データベースからArrow形式にエクスポートしてからGPUに転送する方法がありますが、この過程で**CPUがデータ変換を行うため大きな遅延が発生**します。ユーザーの目的は、**CPUの介在を極力排除し、GPU上で直接PostgreSQLデータをデコードしてcuDFに取り込むパイプライン**を構築することです。

具体的な目標は次の通りです：

- **GPUによる直接データ展開**: PostgreSQLから取得したバイナリデータをGPUで並列にデコードし、cuDF DataFrameに格納する。これによりCPUによる逐次処理を回避し、ロード時間を短縮します。
- **CPU処理の最小化**: データ転送および変換パイプラインから可能な限りCPUの処理を取り除き、GPUへの直接パスを実現します。
- **既存技術の活用**: CUDAやRAPIDSの既存ライブラリ（例：cuIO、libcudf、kvikio、Thrust、CUDA Streams、GPUDirect Storageなど）を利用します。。
- **PostgreSQLからのデータ取得方式の見直し**: 標準的なクライアント経由ではなく、例えば`COPY ... TO STDOUT (FORMAT BINARY)`のバイナリプロトコル出力をそのままGPUで受け取る方法を採用します。
- 

以上の目標に沿って、以下ではシステムの構成を大枠から小枠へ順に説明し、各構成モジュールの役割や連携、使用可能な技術について詳述します。必要に応じてサンプルコードのスニペットやプロトタイプ構成例も示します。

## システム全体のアーキテクチャ（大枠）

まず、高レベルなデータフローの全体像を示します。構築するパイプラインは**以下のステップ**からなります：

1. **PostgreSQLからのバイナリデータ取得**: クエリ結果をPostgreSQLのCOPYバイナリフォーマットでエクスポートします（`COPY table TO STDOUT (FORMAT BINARY)`を利用）。これにより、結果行が**バイナリ形式**（各行についてカラム数や各フィールドの長さ・データがエンコードされたバイト列）で出力されます (@url https://www.postgresql.org/docs/current/sql-copy.html#:~:text=The%20,are%20in%20network%20byte%20order)。この出力データをGPUが直接受け取れるよう、後述する手段でメモリ転送を行います。
2. **データ転送とバッファ管理**: 取得したバイナリデータを**GPUメモリ上のバッファに格納**します。read_flrを参考にしてください。(@folder /mnt/1/a-hirota/cubatch/cpp/src/io/flr/)具体的にはCPUメモリからGPUメモリへのコピーは非同期転送（CUDAストリームを使用した`cudaMemcpyAsync`等）で行い、可能であれば**ページロック（ピン留め）メモリ**やGPUDirect技術を活用してCPUの関与や不要なデータコピーを削減します。大容量データに備え、**ダブルバッファリング**などで転送と後続の処理をオーバラップさせ、GPUが処理中に次のデータを同時に転送できるようにします。
3. **GPU上でのデータデコード（並列パース）**: GPU上に送り込まれたPostgreSQLバイナリフォーマットのバイト列を、CUDAカーネルにより**並列にパース**します。こちらもread_flrを参考にしてください。(@folder /mnt/1/a-hirota/cubatch/cpp/src/io/flr/) 各GPUスレッドまたはワープがデータの一部（例えば特定の行もしくはカラム）を担当し、バイナリデータから実際の値（INT型や文字列など）をデコードしてGPUメモリ上の出力バッファに書き込みます。**PostgreSQLのバイナリ形式**では各タプル（行）が「フィールド数（16bit）＋各フィールドの長さ（32bit）とデータ」が連続して格納され、NULLの場合は長さが-1で示されます (@url https://www.postgresql.org/docs/current/sql-copy.html#:~:text=Each%20tuple%20begins%20with%20a,follow%20in%20the%20NULL%20case)。GPUカーネルはこの構造を解釈しつつ、**数値はエンディアン変換**（PostgreSQL出力はネットワークバイトオーダー＝ビッグエンディアンのため (@url https://www.postgresql.org/docs/current/sql-copy.html#:~:text=The%20,are%20in%20network%20byte%20order)、GPU（リトルエンディアン）で扱うにはバイトスワップが必要）、文字列はバイト配列として抽出し、後で文字列カラム用の構造に格納します。並列処理により数百万行規模のデータでも高速にデコード可能です。
4. **cuDF DataFrameへの格納**: GPUカーネルで得られた各カラムのデータを用いて、RAPIDS cuDFのデータフレームオブジェクトを構築します。こちらもread_flrを参考にしてください。(@folder  /mnt/1/a-hirota/cubatch/cpp/src/io/flr/)。cuDF（内部のlibcudfライブラリ）はApache Arrow互換のカラムメモリレイアウトを採用しており、デバイスメモリ上に配置されたバッファから**直接データフレームを組み立てることが可能**です ([The Arrow C Device data interface — Apache Arrow v19.0.1](https://arrow.apache.org/docs/format/CDeviceDataInterface.html#:~:text=The%20current%20C%20Data%20Interface%2C,it%20between%20runtimes%20and%20libraries))。各カラムごとに、データ本体のデバイスメモリポインタと、Null値用のビットマスク（有無に応じて）や文字列の場合はオフセット配列を用意し、それらを組み合わせてcuDFの`column`オブジェクトを生成します。その後、それらのcolumnを集めて`cudf::table`（またはPythonのcudf.DataFrame）を作成します。これにより**データ取得からGPUデータフレーム生成までが一貫してGPUメモリ上で完結**し、中間でCPU上のpandasデータフレームやArrowテーブルを経由する必要がなくなります ([python - Is there anyway to read data from Microsoft sql directly into cudf (GPU's RAM)? - Stack Overflow](https://stackoverflow.com/questions/77960181/is-there-anyway-to-read-data-from-microsoft-sql-directly-into-cudf-gpus-ram#:~:text=Thanks%20for%20the%20question,is%20a%20reasonable%20choice))。

この全体フローでは、**CPUは主に制御と補助的な役割のみ**を担い、大量のデータの解析はGPU上で実行されます。GPUの並列処理能力を活かすことで、従来CPUでボトルネックとなっていたデータ変換を大幅に高速化できます。また**GPUDirect Storage/RDMA**等の技術によりCPUを介さないデータ移動を実現すれば、ストレージやネットワークからGPUへのI/O経路も最適化され、**データロード全体のスループット向上**が期待できます ([Boosting Data Ingest Throughput with GPUDirect Storage and RAPIDS cuDF | NVIDIA Technical Blog](https://developer.nvidia.com/blog/boosting-data-ingest-throughput-with-gpudirect-storage-and-rapids-cudf/#:~:text=CUDA,in%20data%20science)) ([NVIDIA GPUDirect Storage: 4 Key Features, Ecosystem & Use Cases](https://cloudian.com/guides/data-security/nvidia-gpudirect-storage-4-key-features-ecosystem-use-cases/#:~:text=GPUDirect%20Storage%20focuses%20on%20optimizing,AI%2C%20ML%2C%20and%20HPC%20workloads))。

以下、上記の各ステップをモジュール単位に分け、構成要素（中枠）とその詳細な処理内容（小枠）を説明します。

## モジュール構成と役割（中枠）

システムをいくつかの主要モジュールに分割し、それぞれの機能と相互作用について整理します。

- **データ取得モジュール（PostgreSQLクライアント）**
    
    PostgreSQLからデータをエクスポートし、GPU側に送るまでの処理を担当します。`COPY ... TO STDOUT (FORMAT BINARY)`などのインターフェースを用いて、結果セットをバイナリ形式で取得します。取得したバイナリデータは一時バッファ（CPU側メモリまたは直接GPUメモリ）に蓄え、GPUへの転送要求を行います。このモジュールではネットワークおよびPostgreSQLプロトコルの処理を行い、高スループットでデータを読み出せるよう**ストリーミング処理**や**パイプライン実行**（後段GPU処理との並行実行）を実装します。
    
- **メモリ転送・管理モジュール**
    
    データ取得モジュールから受け取ったバイト列をGPUメモリに移動させる部分です。CPU-GPU間のデータ転送を最適化するため、CUDAのピン留めメモリ（ページロックメモリ）を利用して**ゼロコピー転送**や**DMA転送**を可能にします。小さなチャンクに分割して順次コピーするのではなく、大きめのチャンクをまとめて非同期コピーしつつ、ダブルバッファリングで転送待ち時間を隠蔽します。また、システム構成によっては**GPUDirect**技術の利用も検討します。例えば、GPUDirect Storage対応のNVMeであればPostgreSQLが一旦ファイルに書き出したデータをGPUが直接読み込むことも可能です ([Boosting Data Ingest Throughput with GPUDirect Storage and RAPIDS cuDF | NVIDIA Technical Blog](https://developer.nvidia.com/blog/boosting-data-ingest-throughput-with-gpudirect-storage-and-rapids-cudf/#:~:text=CUDA,in%20data%20science))。あるいはInfinibandネットワーク＋RDMA対応NICを用いれば、**GPUDirect RDMA**によりネットワークから直接GPUメモリへデータを取り込むことも概念上は可能です ([NVIDIA GPUDirect Storage: 4 Key Features, Ecosystem & Use Cases](https://cloudian.com/guides/data-security/nvidia-gpudirect-storage-4-key-features-ecosystem-use-cases/#:~:text=GPUDirect%20Storage%20focuses%20on%20optimizing,AI%2C%20ML%2C%20and%20HPC%20workloads))（注：PostgreSQL標準ではサポートしないため特別な実装が必要）。
    
- **GPUデシリアライズ（パーサ）モジュール**
    
    GPU上で動作するCUDAカーネル群です。PostgreSQLバイナリフォーマットで格納された生データを解釈し、カラムごとの出力バッファに値を格納します。パーサは複数のステージに分かれる可能性があります。例えば**第一段階**では各行を担当するスレッドがバイナリデータをスキャンし、各フィールドの位置・長さを特定します（メタデータ抽出）。**第二段階**で、フィールドごとの値を適切な型に変換しながら各カラム用の出力メモリに書き出します。数値型であればビット単位でエンディアン変換しつつ32bitや64bitの値を格納し、文字列型であれば文字データをまとめて文字バッファにコピーし、対応するオフセットを記録します。これらの処理はスレッドブロックごとに並列実行され、行単位・カラム単位で効率的に行われます。**Thrust**などCUDAの並列アルゴリズムライブラリを用いて、例えば可変長データのオフセット計算にプレフィックススキャンを使う、NULLフラグのビットマスク生成にビット演算を使う、といった実装も可能です。デシリアライズ結果として、各カラムについて「データ配列（GPUメモリ上）」「NULLマスク（必要なら）」「文字列の場合はoffset配列と文字データ配列」といった低レベルメモリ構造が得られます。
    
- **cuDF組み立てモジュール**
    
    GPUでパースされたデータからcuDFのDataFrameを構成する部分です。C++レベルではlibcudfのAPIを用いて、前段で用意した各カラムのデバイスメモリから`cudf::column`オブジェクトを生成し、それらをまとめて`cudf::table`を構築します。Pythonレベルであれば、CuPyやNumbaで確保したGPUメモリをcudf.DataFrameに渡すことも可能です（cudfは内部でArrowメモリフォーマットに沿ったGPUバッファを扱えるため、**同じプロセス内**であればポインタの受け渡しで**ゼロコピー共有**ができます ([The Arrow C Device data interface — Apache Arrow v19.0.1](https://arrow.apache.org/docs/format/CDeviceDataInterface.html#:~:text=%2A%20Expose%20an%20ABI,the%20existing%20C%20data%20interface))）。このモジュールは主にメタデータの組み立てが中心であり、実データのコピーは発生しません。結果として得られたcuDF DataFrameは、GPU上でそのまま後続の処理（例えばGPU上での分析や機械学習パイプライン）に供せます。
    

以上が主要なモジュールです。それぞれのモジュール間は**ストリームパイプライン**のように連結し、非同期に動作します。例えば、先行するデータチャンクのGPUパース処理中に、次のデータチャンクをPostgreSQLから取得してGPUメモリに転送するといった並行動作が可能です。これによりGPUコアの演算とPCIeデータ転送、PostgreSQLからの読み出しを重畳させ、ハードウェア資源をフルに活用します。

次節からは、各モジュール内の詳細な処理内容や実現方法（小枠部分）を掘り下げます。

## PostgreSQLデータ取得と転送の詳細（小枠）

**● COPYバイナリ形式でのデータ取得**: PostgreSQLからデータを取得する際は、通常のSQL問い合わせ結果をテキストで受け取るのではなく、**COPYコマンドのバイナリモード**を使用します。例えばSQL文として`COPY (SELECT * FROM my_table) TO STDOUT (FORMAT BINARY)`を発行すると、クライアントはPostgreSQL独自のバイナリフォーマットでエンコードされた結果をストリームとして受け取れます ([PostgreSQL: Documentation: 17: COPY](https://www.postgresql.org/docs/current/sql-copy.html#:~:text=The%20,are%20in%20network%20byte%20order))。このバイナリフォーマットは、先述の通り**ファイルヘッダ・レコード集合・トレーラ**から構成されます ([PostgreSQL: Documentation: 17: COPY](https://www.postgresql.org/docs/current/sql-copy.html#:~:text=The%20,are%20in%20network%20byte%20order))。クライアント側ではlibpqの関数（例：`PQgetCopyData`）を用いてストリームからデータチャンクを逐次読み出します。可能であれば**非同期モード**で読み出しを行い、I/O待ちでブロックしないようにします。

**● CPUメモリからGPUメモリへの直接受け渡し**: 取得した生のバイナリチャンクはまずCPU上のバッファに格納されます。この際のボトルネックを減らすために、**ページロック済みメモリ**を利用します。CUDAの`cudaHostAlloc()`などで確保したホストメモリはページフォールトを伴わずにデバイスとのDMA転送が可能で、GPU側から「ピア・トゥ・ピア」で直接アクセスすることもできます ([NVIDIA GPUDirect Storage: 4 Key Features, Ecosystem & Use Cases](https://cloudian.com/guides/data-security/nvidia-gpudirect-storage-4-key-features-ecosystem-use-cases/#:~:text=GPUDirect%20Storage%20functions%20by%20establishing,and%20more%20efficient%20data%20handling))。従って、libpqがデータを書き込むバッファ自体をページロックメモリとして確保することで、コピーの手間を削減できます（libpqは通常自前でメモリ確保しますが、大きなチャンクを読み込んだ後ユーザープログラム側に制御が戻った時点で、そのメモリ内容をすみやかにGPUに渡す戦略です。必要ならバッファを`cudaHostRegister`でページロックすることも検討します）。読み出しと同時に`cudaMemcpyAsync`でGPUメモリへデータを送り、送信完了を待たずに次のチャンク取得を進めます。結果として、CPUはPostgreSQLとの通信とGPU転送の指示をするだけで、**データ自体の移動はDMAエンジン任せ**となります ([NVIDIA GPUDirect Storage: 4 Key Features, Ecosystem & Use Cases](https://cloudian.com/guides/data-security/nvidia-gpudirect-storage-4-key-features-ecosystem-use-cases/#:~:text=GPUDirect%20Storage%20functions%20by%20establishing,and%20more%20efficient%20data%20handling))。

**● GPUDirectによる最適化**: 場合によってはさらに直接的なデータ受け渡しも検討できます。同一マシン上でPostgreSQLサーバとGPUを搭載している場合、`COPY ... TO PROGRAM`機能を使ってサーバサイドで直接GPUメモリに書き出すプログラムを動かすことも可能です ([PostgreSQL: Documentation: 17: COPY](https://www.postgresql.org/docs/current/sql-copy.html#:~:text=Files%20named%20in%20a%20,since%20it%20allows%20reading))。この外部プログラム内でCUDAのAPIを使い、PostgreSQLサーバプロセスから渡された標準出力（パイプ経由のバイナリデータ）をそのままGPUに書き込めば、クライアントを経由しない分ネットワークスタックやソケットコピーのコストを削減できます。さらにネットワーク越しにデータ取得する場合でも、InfiniBand + RDMA対応NICが使える環境では、特殊な実装により**ネットワークからGPUメモリへのRDMA読み取り**が可能です ([NVIDIA GPUDirect Storage: 4 Key Features, Ecosystem & Use Cases](https://cloudian.com/guides/data-security/nvidia-gpudirect-storage-4-key-features-ecosystem-use-cases/#:~:text=GPUDirect%20Storage%20focuses%20on%20optimizing,AI%2C%20ML%2C%20and%20HPC%20workloads))。これにより**CPUを一切介在させずにデータ搬送**ができます（GPUDirect RDMAはMPIやNVMe-oFで実績があります ([NVIDIA GPUDirect Storage: 4 Key Features, Ecosystem & Use Cases](https://cloudian.com/guides/data-security/nvidia-gpudirect-storage-4-key-features-ecosystem-use-cases/#:~:text=GPUDirect%20Storage%20focuses%20on%20optimizing,AI%2C%20ML%2C%20and%20HPC%20workloads))）。ただしPostgreSQLのプロトコル自体がTCPベースでありRDMAを直接利用しないため、RDMA経由でデータを取得するにはカスタムのプロキシやプロトコル変換レイヤーが必要でしょう。現実的には、まずページロックメモリ＋非同期DMA転送による実装で十分高速化が見込めます。GPUDirect Storageについては、PostgreSQLが直接ファイルにコピー出力しGPUがそれを読む方式も考えられますが、リアルタイムクエリ処理よりバッチ的な用途向きです ([python - Is there anyway to read data from Microsoft sql directly into cudf (GPU's RAM)? - Stack Overflow](https://stackoverflow.com/questions/77960181/is-there-anyway-to-read-data-from-microsoft-sql-directly-into-cudf-gpus-ram#:~:text=a%20reasonable%20choice))。

**● 転送サイズとバッファリング**: データ取得と転送のモジュールでは、効率を高めるため**適切なチャンクサイズ**を選定します。小さすぎるチャンクだと転送オーバーヘッドが増え、大きすぎるとメモリ消費やレイテンシが増加します。例えば数MB～数十MB単位でチャンク分割し、GPU上にリングバッファを用意して順次埋めていく方法が考えられます。CUDAストリームを2つ以上使い、**一方のストリームでカーネル処理中に他方でデータコピー**を行うことで、PCIe帯域とGPU演算の両方を最大限活用します。

以上により、「PostgreSQLからGPUメモリ上のバッファまで」データを運ぶ部分のCPUボトルネックを大幅に緩和できます。次に、このGPU上の生データをどのようにデコードし、cuDFのカラム形式に変換するかを説明します。

## GPU上でのデータデコード処理（小枠）

**● PostgreSQLバイナリフォーマットの並列パース**: GPUに転送されたバイナリデータは、PostgreSQLのコピー形式に準拠しています。この形式では、ヘッダの後に複数の**タプル（行）**が連続し、最後にトレーラ（0xFFFF）で終端します ([PostgreSQL: Documentation: 17: COPY](https://www.postgresql.org/docs/current/sql-copy.html#:~:text=))。各タプルは**全フィールド数（16-bit整数）**に続いて各フィールドの「長さ（32-bit整数）とデータ本体」が並んでいます ([PostgreSQL: Documentation: 17: COPY](https://www.postgresql.org/docs/current/sql-copy.html#:~:text=Each%20tuple%20begins%20with%20a,follow%20in%20the%20NULL%20case))。NULL値のフィールドは長さが-1（0xFFFFFFFF）で記録され、データ本体のバイト列は存在しません ([PostgreSQL: Documentation: 17: COPY](https://www.postgresql.org/docs/current/sql-copy.html#:~:text=Each%20tuple%20begins%20with%20a,follow%20in%20the%20NULL%20case))。またファイル全体を通じて**整数値はネットワークバイト順（ビッグエンディアン）**でエンコードされています ([PostgreSQL: Documentation: 17: COPY](https://www.postgresql.org/docs/current/sql-copy.html#:~:text=The%20,are%20in%20network%20byte%20order))。

GPUで実装するパーサカーネルは、この構造を利用して並列にデータを読み解きます。典型的な実装アプローチとしては、**1スレッド＝1行（タプル）**を担当させ、そのスレッドが自分の行内のフィールドを順次読み出して各カラム出力に書き込む、という方法があります。各スレッドは以下の処理を行います（疑似コード風に示します）:

```
thread_id = 行インデックス
pos = 自スレッド担当行の開始オフセット（事前に計算済み）
num_fields = *(uint16_t*)(raw_buffer + pos) （まず16ビットのフィールド数を取得）
pos += 2
for col in 0 to num_fields-1:
    field_length = *(int32_t*)(raw_buffer + pos) （32ビット長を取得）
    pos += 4
    if (field_length == -1) {
        // NULLフィールド
        mark_null(thread_id, col)  // 該当カラムのNULLマスクにセット
    } else {
        field_data_ptr = raw_buffer + pos
        if (カラムcolのデータ型が固定長数値) {
            // エンディアン変換しつつ出力バッファに書き込む
            output_val = byteswap_load(field_data_ptr, field_length)
            device_column[col][thread_id] = output_val
        } else {
            // 可変長（文字列等）の場合はデータを一時バッファにコピーまたはオフセット記録
            output_offset = atomicAdd(total_bytes[col], field_length)
            memcpy(device_varlen_buffer[col] + output_offset, field_data_ptr, field_length)
            device_offsets[col][thread_id] = output_offset
            device_offsets[col][thread_id+1] = output_offset + field_length  // 次の位置（後でprefix sumでも可）
        }
    }
    pos += max(field_length, 0)  // 長さが-1(null)なら増分0、それ以外はそのバイト数進める

```

上記は概念例ですが、実際には**事前に各行の開始位置`pos`を求める工程**が必要です。行ごとの開始位置は、全体バッファをある程度均等に分割し各スレッドブロックに割り当て、各ブロック内で逐次スキャンして「自分の担当開始オフセット」までスキップする、という方法で計算できます。または、最初から**行区切りのインデックス配列**を作成する2段階パーサ（まず全行のオフセット一覧をGPUで計算→次に各行処理）にする手もあります。これはCSVなどをGPUパースする際によく用いられるテクニックです。

**● データ型ごとの処理**: フィールドデータの書き込み処理は**データ型**によって異なります。固定長の数値型（整数、浮動小数点など）の場合、`field_length`は型のバイトサイズになります。このサイズが期待通りか（例えばINTなら4バイト）確認しつつ、GPUで**エンディアンスワップ**して整数値/浮動小数値に変換します。エンディアン変換はCUDA組込みのバイト操作命令やビットシフトで実装できますが、簡単のため各4バイトごとに`__byte_perm`命令（32ビットレジスタのバイト順入れ替え）を用いる方法もあります。あるいは、CuPyを用いて後段で`cupy.ndarray.byteswap()`を呼び出す方法でも実装可能です。浮動小数点も同様にビットパターンを入れ替えるだけで問題ありません。

一方、可変長データ型（テキストvarcharやbytea等）の場合、`field_length`はそのデータのバイト数を示します。GPUで文字列バイトを扱うには、**文字データバッファ**と**オフセット配列**の2つを構築する必要があります。各行スレッドは自分の文字列フィールドを一旦どこかに書き出す必要がありますが、直接グローバルの出力バッファに書くと競合する可能性があるため、上記疑似コードでは`atomicAdd`を使ってグローバルバッファ上の書き込み開始位置を確保しています。より効率的には、各ブロック内で局所的にバイト数を集計してからブロックごとにオフセットを割り当て、最後にブロック内スレッドが各自コピーする、という方法も取れます。**Thrust**や**CUB**ライブラリのプリフィックスサム（inclusive scan/exclusive scan）を利用すれば、全体のオフセット計算も高速に実装できます。例えば各行の文字列長を配列に記録しておき、Thrustでプレフィックスサムを取ると累積オフセットが得られるので、それを各行の書き込み位置にできます。

**● ストリーム並列とカーネル分割**: データ量が非常に大きい場合、1回のカーネン起動で全行を処理するとスレッド数が多くなり過ぎたり、実行時間が長くなりタイムアウトの懸念があります。そこで、例えば**1チャンク＝数十万行**程度に区切り、チャンクごとに上記のGPUパースカーネルを実行します。各チャンク処理はCUDAストリームに割り当てて独立に実行できるため、前述のメモリ転送モジュールとパイプラインを形成します。また、カーネルの構成も最適化次第です。1つの大きなカーネルで全処理（オフセット計算＋値コピー）を行うよりも、**段階に分けたカーネル**を投入したほうがスケジューリングしやすい場合もあります。例えば:

- 第1カーネル: 各行のフィールド長だけを読み取り、各カラムの必要バイト数やNULL数を算出する。
- 必要に応じてCPU側でデバイスメモリを確保（各カラムの出力バッファを確保、文字列バッファは総バイト数に基づきalloc）。
- 第2カーネル: 各行を処理して数値カラムは直接書き込み、文字列カラムは適切な位置に書き込む（オフセットは第1カーネル結果のprefix sumで求めたものを使用）。

libcudfの実装でも、複雑なパーサは複数段に分かれていることがあります。例えばJSONやCSVのGPUパーサは、まず区切り位置を検出し、それからフィールド値を変換するステージに分かれています。このように段階化することで、処理ごとのメモリアクセスパターンを単純化でき、また一部の結果（例えば各カラムのサイズ）を使って後続処理用のメモリを動的に確保するといったことも可能になります。

**● エラーハンドリングと同期**: GPU上でパースする際、データ形式に不備があった場合の対処も考慮します。本来PostgreSQLから出力されるコピー形式は厳密に定義されているため、形式エラーは起こりにくいですが、例えば予期しないNULLや長さ不一致があれば、その行の処理をスキップしたりエラーフラグを立てておき、後でCPU側に通知します。カーネルの終了後にはCUDAのストリームを同期し、該当チャンクの処理完了を確認してから次の段階（DataFrame組み立て）に進みます。

このGPUデコードモジュールによって、**データ型ごとの変換処理がすべてGPU内で完結**し、中間結果として「GPUメモリ上の各カラム配列＋メタデータ（長さ配列やnullマスク等）」が手に入ります。ここから先は、cuDFのデータフレームオブジェクトにこれらを結合する仕上げの段階です。

## cuDFデータフレームへの格納と統合

GPUでデコードされたカラムデータを、cuDF（GPU DataFrame）として扱うには、適切なデータ構造に組み上げる必要があります。幸い、**cuDFのカラムはApache Arrowのメモリレイアウトと互換**があり、我々が用意したデバイスメモリバッファをそのまま利用できます ([The Arrow C Device data interface — Apache Arrow v19.0.1](https://arrow.apache.org/docs/format/CDeviceDataInterface.html#:~:text=The%20current%20C%20Data%20Interface%2C,it%20between%20runtimes%20and%20libraries))。具体的な組み立て手順は以下の通りです。

1. **カラムバッファからcudf::columnオブジェクトへ**: C++でlibcudfを使う場合、例えばint型カラムであれば`cudf::make_fixed_width_column(data_type(INT32), row_count, mask_state::UNALLOCATED)`でカラムの器を作り、そこに対して`column->set_data(device_ptr)`のようにデータバッファを移管します（実際のAPI呼び出しは省略していますが、cudfはデバイスメモリ上の生データを受け取るコンストラクタやファクトリ関数を備えています）。可変長の文字列カラムの場合は少し複雑で、libcudfの`cudf::strings::create_offsets`関数などでオフセット配列と文字バッファから`cudf::column`（strings専用の子クラス）を作成します。またNULLマスクについては、取得した各カラムのnullフラグ配列をビット圧縮し、cudfの有効ビットマスク（validity bitmask）として`column->set_null_mask()`で設定します。これらの操作によって**各カラムがGPUメモリ上のデータで構築**され、コピー無しでcudfカラムオブジェクトが得られます。
2. **columnsからtable（DataFrame）へ**: 構築した全ての`cudf::column`を配列やベクタに収集し、`cudf::table_view`を経由して`cudf::table`（もしくはPythonであれば`cudf.DataFrame`）を生成します。Pythonの場合、高レベルなAPIとして`cudf.DataFrame()`に対してカラム名をキー、上記Cupy配列（デバイスメモリ上の配列）を値とする辞書を渡す方法もあります。この場合も内部的にはcupy配列のデータポインタを取り出し、対応するcudf::columnを作る処理をしています。注意点として、cupy経由でNullマスクを指定するAPIは限定的なので、Nullを含むカラムはC++側で作ってからDataFrameに取り込む方が確実です。
3. **メモリ所有権と後処理**: 一連の処理で確保してきたGPUメモリバッファ（データバッファやオフセットバッファ等）は、最終的にcudf::columnに移管された後は**cudf側で管理**されます。例えばcudf::columnはデストラクタで自身のデータバッファを解放します。そのため、パーサで使った一時バッファ類は二重に解放しないよう注意が必要です（生データの大元バッファはパース完了後に不要になるので解放し、各カラム用に改めて確保したバッファはcudfに渡す、という整理をします）。また、組み立て完了後にパイプライン全体として次のチャンクの処理に進む際、必要なくなったバッファを適宜解放・再利用してメモリフットプリントを抑えます。

以上の手順で、**cuDFのデータフレームが完成**します。こうして得られたDataFrameは、pandas同様の操作をGPU上で高速に行える他、Daskなどを用いて分散処理することもできます。特筆すべきは、**データ取得からDataFrame構築まで一度もCPU上にデータを展開していない**点です。従来手法（一度pandas DataFrameを経由、あるいはArrow経由）ではCPU RAM上に全データを載せる必要がありメモリ帯域を消費していましたが、本手法ではそれを回避しています。このようにデータを極力GPU上に留めて処理することは、Apache Arrowプロジェクトでも**“データをホストに戻さずデバイス間で零拷送する”**という方向で標準化が進んでおり ([The Arrow C Device data interface — Apache Arrow v19.0.1](https://arrow.apache.org/docs/format/CDeviceDataInterface.html#:~:text=Goals))、本パイプラインの考え方はその最先端を行くものと言えます。

## 利用可能なライブラリ／フレームワークと拡張の可能性

本構成を実現するにあたり、いくつかの既存ライブラリや技術を活用できます。また不足する機能はカスタム実装や拡張によって補います。それぞれについて整理します。

- **RAPIDS cuIO / libcudf(read_flr/write_flr)**: RAPIDS cuDFには、CSVやParquetなどさまざまなデータ形式のGPU読み取り（パーサ）実装が含まれる**cuIO**モジュールがあります。残念ながらPostgreSQL独自のCOPYバイナリ形式に対する既成のリーダーは存在しません。しかし、cuIOの設計にならって**独自の入力フォーマットパーサ**を実装することは可能です。libcudfは高性能なメモリアロケータや列データ構造を提供しており、文字列カラム構築用のユーティリティやビットマスク操作関数なども利用できます。将来的には、PostgreSQLサーバサイドでApache Arrow形式を直接出力し、それをArrow GPUデバイスインターフェース経由でcudfに渡すといった統合も考えられます（ArrowのCデバイスインターフェースはGPUメモリ上のArrowバッファを他プロセスと共有する仕組みで現在実験的機能です ([The Arrow C Device data interface — Apache Arrow v19.0.1](https://arrow.apache.org/docs/format/CDeviceDataInterface.html#:~:text=%2A%20Expose%20an%20ABI,the%20existing%20C%20data%20interface))）。しかし現時点では、自前でパーサを実装しlibcudfのAPIでカラムを組み立てるアプローチが確実です。
- **Thrust / CUB**: GPU上の並列アルゴリズムについて、NVIDIAが提供するThrustやCUBライブラリを活用できます。特に**prefix sum（スキャン）や並列ソート・ユニーク**などはCUBに高度に最適化された実装があります。可変長データ処理では、各行の文字列長配列に対し`thrust::inclusive_scan`を使ってオフセット配列を生成し、それをもとに`cudaMemcpyAsync`で一括コピーするなどの手法が考えられます。また、Thrustの`transform`を用いれば、ビットマスク生成やエンディアン変換もシンプルに書けます。たとえば、
    
    ```cpp
    thrust::transform(d_ptr, d_ptr + n, output_ptr, [] __device__ (uint32_t x) {
        return __byte_perm(x, 0, 0x0123);  // 4バイトエンディアン変換の例
    });
    
    ```
    
    のようにCUDA組込み関数と組み合わせることで、forループを明示的に書かずに並列処理できます。もっとも、こうした抽象度の高い書き方は場合によってオーバーヘッドもあるため、性能追求段階では生のCUDAカーネルを書く形に落とし込むことになります。
    
- **CUDA Streamsとコンカレンシー**: 前述のように、CUDAストリームを複数駆使してデータ転送とカーネル実行をパイプライン化することが重要です。CUDAは同一デバイス上で複数のストリームを並行実行でき、**データ転送（CUDAメモリコピー）はデフォルトで非同期**なので、適切なストリームに割り当てればコピーとカーネルが重畳実行されます。たとえば`stream1`でデータコピー、`stream2`で前チャンクのパースカーネル、とし、各処理の後で必要な同期（イベントによる待機）を入れる設計です。さらに、GPU内でのコンカレンシーとしては、カーネルを小さいグリッドに分割してストリーム間で実行する「マルチストリーム並列」も検討できますが、まずはチャンク粒度の並行で十分でしょう。
- **GPUDirect Storage / RDMA**: データ転送最適化の高度な技術としてNVIDIA GPUDirectがあります。GPUDirect Storage (GDS)は**GPUとストレージ間の直接DMA**を可能にし、CPUのバウンスバッファを排除することで最大3～4倍のスループット向上を達成したケースも報告されています ([Boosting Data Ingest Throughput with GPUDirect Storage and RAPIDS cuDF | NVIDIA Technical Blog](https://developer.nvidia.com/blog/boosting-data-ingest-throughput-with-gpudirect-storage-and-rapids-cudf/#:~:text=CUDA,in%20data%20science))。これは主にファイルIO向けの技術ですが、PostgreSQLからのデータエクスポートにおいても、いったんサーバ側でファイル出力させそれをGPUで直接読む形で応用可能です（ただしオンラインクエリ処理には不向き）。GPUDirect RDMAは上述のとおりネットワーク越しにGPUメモリを直接やりとりする技術で、**CPU関与を完全になくす究極の形**です ([NVIDIA GPUDirect Storage: 4 Key Features, Ecosystem & Use Cases](https://cloudian.com/guides/data-security/nvidia-gpudirect-storage-4-key-features-ecosystem-use-cases/#:~:text=GPUDirect%20Storage%20focuses%20on%20optimizing,AI%2C%20ML%2C%20and%20HPC%20workloads))。現状のPostgreSQLには組み込まれていませんが、学術研究や一部のGPUデータベース（例：PG-Strom）では類似のコンセプトが実装されています ([GPUDirect SQL - PG-Strom Manual](https://heterodb.github.io/pg-strom/ssd2gpu/#:~:text=GPU%20Direct%20SQL%20Execution%20changes,I%2FO%20processing%20in%20the%20results))。PG-StromではNVMe SSD上のデータを直接GPUに読み込んでSQLフィルタ処理をするGPUDirect SQL機能があり、**GPUをストレージとメモリの間の前処理プロセッサとして活用する**ことでCPU負荷を劇的に減らしています ([GPUDirect SQL - PG-Strom Manual](https://heterodb.github.io/pg-strom/ssd2gpu/#:~:text=GPUDirect%20SQL%20Execution%20directly%20connects,wired%20speed%20of%20the%20hardware)) ([GPUDirect SQL - PG-Strom Manual](https://heterodb.github.io/pg-strom/ssd2gpu/#:~:text=GPU%20Direct%20SQL%20Execution%20changes,I%2FO%20processing%20in%20the%20results))。これら先行事例からも、GPU直接データロードの有効性が伺えます。

以上のライブラリ/技術を組み合わせ、足りない部分はカスタムコードで補うことで、PostgreSQLからcuDFへの直接パイプラインは実現可能です。実装に際しては、まず**小さなデータセットで正しく動作することを確認**し、徐々にチューニングや最適化（バッファサイズ調整、スレッド割り当て最適化など）を行うと良いでしょう。

## プロトタイプ構成例と疑似コード

最後に、以上の構成を踏まえた**プロトタイプ実装の流れ**を疑似コード形式で示します。C++ベースでlibpq（PostgreSQLクライアント）とCUDAを用いた場合の一例です。

```cpp
// 前提: 接続確立済みのPGconn* pgconn がある
PGresult* res = PQexec(pgconn, "COPY my_table TO STDOUT (FORMAT BINARY)");
if (PQresultStatus(res) != PGRES_COPY_OUT) {
    // エラー処理
}

const size_t CHUNK_SIZE = 8 * 1024 * 1024;  // 8MBチャンク例
uint8_t* host_buf;
cudaHostAlloc((void**)&host_buf, CHUNK_SIZE, cudaHostAllocDefault); // ページロック確保
uint8_t* dev_buf;
cudaMalloc(&dev_buf, CHUNK_SIZE);

bool done = false;
int read_bytes;
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
int chunk_index = 0;
std::vector<cudf::column> columns;  // 最終的なカラム配列
// ※実際はカラム別に別バッファを持つので、ここでは簡略化している

while (!done) {
    // PostgreSQLからバイナリデータを取得（非同期ループ）
    char* buffer = nullptr;
    read_bytes = PQgetCopyData(pgconn, &buffer, 0);  // ブロッキング呼び出し
    if (read_bytes == -1) {
        done = true;  // 終端
        break;
    } else if (read_bytes < 0) {
        // エラー処理
        break;
    }
    // bufferにread_bytesバイトのデータ
    // memcpyでhost_bufにコピー (実装ではポインタ受け渡し最適化検討)
    memcpy(host_buf, buffer, read_bytes);
    PQfreemem(buffer);

    // GPUへ非同期コピー
    cudaMemcpyAsync(dev_buf, host_buf, read_bytes, cudaMemcpyHostToDevice, stream1);

    // GPU上でデコードカーネル起動（stream2で実行、前回のchunkと並行可能）
    launch_decode_kernel(columns, dev_buf, read_bytes, stream2);

    // この例ではシンプルに交互ストリームで実行
    cudaStreamSynchronize(stream1); // 転送完了待ち
    cudaStreamSynchronize(stream2); // カーネル完了待ち

    chunk_index++;
}

// 全chunk処理完了、columnsにデータ格納済みとする
std::unique_ptr<cudf::table> gpu_table = cudf::table::create(std::move(columns));
// 必要に応じてcudf::table_view経由でDataFrameに渡す

```

上記は非常に簡略化した疑似コードですが、ポイントとしては：

- `PQgetCopyData`でデータ取得後、すぐにGPU転送とカーネル処理を非同期で発行し、次のループへ進む構造になっている点（実際にはループ内で2つのストリームを交互に使うなどの工夫が必要）。
- `launch_decode_kernel`は、取得したバイト列を解析して`columns`に追加していく処理を指します。実装上は、最初のチャンク受信時に列数やデータ型情報を把握し、各列用のデバイスバッファを確保しておき、以降のチャンクでは既存バッファの所定位置に書き足す形になるでしょう。疑似コードでは簡単のため`columns`ベクタに直接追加していますが、実際は各列ごとにバッファを持ち再割当てやサイズ拡張を管理します。
- ループ終了後、`cudf::table`を組み立てています。この時点で全データがGPU上に揃っている前提です。大きなデータの場合、途中でchunkごとにpartialなDataFrameを構成し、Daskなどで分散させることも考えられます。

Pythonで実装する場合も同様の流れになります。例えばpsycopgなどでCOPYを実行し、データを読み出す部分を`cursor.read()`でバイト列取得→cupyに渡してカーネル実行、といった形です。PythonではGILの制約があるため、バックエンドをC/C++拡張にするかマルチスレッドで受信とGPU処理を分けることになるでしょう。

## まとめ

本調査では、**PostgreSQLからGPUへの直接データ転送とデコードによるcuDF格納パイプライン**の構成について、大枠から詳細まで検討しました。CPUによるボトルネック処理（シリアライズ/デシリアライズ）を排除し、**GPUの並列処理と高速メモリ帯域を最大限に活用することで、データロードの高速化が期待できる**ことが分かりました。

特に重要な点として:

- PostgreSQLのCOPYバイナリ形式を活用し、データをテキスト変換せずバイナリのまま取得することで、後段のGPU処理に適した形でデータを受け渡せる。
- GPUへのデータ搬送ではページロックメモリやGPUDirectを用いてCPUの関与とメモリコピーを減らし、DMAによる直接転送でスループットを向上させる ([Boosting Data Ingest Throughput with GPUDirect Storage and RAPIDS cuDF | NVIDIA Technical Blog](https://developer.nvidia.com/blog/boosting-data-ingest-throughput-with-gpudirect-storage-and-rapids-cudf/#:~:text=CUDA,in%20data%20science)) ([NVIDIA GPUDirect Storage: 4 Key Features, Ecosystem & Use Cases](https://cloudian.com/guides/data-security/nvidia-gpudirect-storage-4-key-features-ecosystem-use-cases/#:~:text=GPUDirect%20Storage%20focuses%20on%20optimizing,AI%2C%20ML%2C%20and%20HPC%20workloads))。
- CUDAカーネルでPostgreSQLバイナリフォーマットを並列解釈し、カラムナーなメモリ構造（列ごとのバッファ＋オフセット等）を生成することで、そのままcuDFのDataFrameに組み込める。
- この一連の処理は、既存のGPUデータフレームライブラリ（RAPIDS）やCUDA機能（Streams、Thrustなど）を組み合わせて実現可能であり、一部機能はカスタム実装が必要になるものの技術的な見通しは明るい。

実装上は、メモリ管理や同期、エラー処理など考慮すべき細部も多く存在します。しかし、適切に調整すれば**「データベースから結果を取り出す→データフレームとして分析に供する」という一連の流れをフルGPUでこなす**ことができ、ビッグデータ分析におけるデータロード時間を大幅に短縮できるでしょう ([python - Is there anyway to read data from Microsoft sql directly into cudf (GPU's RAM)? - Stack Overflow](https://stackoverflow.com/questions/77960181/is-there-anyway-to-read-data-from-microsoft-sql-directly-into-cudf-gpus-ram#:~:text=Thanks%20for%20the%20question,is%20a%20reasonable%20choice))。今後、このような手法が一般化すれば、データエンジニアリングのワークフローはさらに高速化し、インサイト獲得までの時間が短縮されることが期待されます。
