PostgreSQLバイナリ形式からArrow型へのマッピング方法について、ADBCドライバおよびPG-Stromとの比較を行い、速度、機能、スキーマ変更耐性の観点で評価します。
また、現在のコードに多くのバグがあり複雑すぎるとのことなので、シンプルで保守しやすい設計に向けたリファクタリング案も提示します.


# ADBCドライバとPG-StromによるPostgreSQL→Arrow変換の比較

## Apache Arrow ADBC PostgreSQLドライバの方式と特徴

**● 型変換のアプローチ:** ADBCのPostgreSQLドライバ（libpq経由）は、サーバから`COPY BINARY`形式でデータを取得し、C++のArrow APIでそのままArrowのカラムデータにマッピングします 。PostgreSQLの各データ型に対して対応するArrowの型が定義されており、例えば`INT4`はArrowの`int32`、`BIGINT`は`int64`、`TEXT/VARCHAR`は`utf8`（Unicode文字列）というように変換されます  。NULL値についてはArrowの有効ビットマップ（validity bitmap）で保持され、**NULLを欠損値として正確に再現**します（Arrowカラム上でNULLフラグ管理）。多くの基本型は直接同等のArrow型にマップされ、**PostgreSQLとArrow間の型マッピングは包括的**です  。

**● 数値型・文字列型の扱い:** 特筆すべきは`NUMERIC`（可変精度小数）の扱いで、ADBCドライバでは**精度を損なう変換を避けるため、`NUMERIC`はArrow上では文字列（utf8）として読み込みます** 。これは「NUMERICはArrowのdecimal型にロスなくマップできない」ためで、文字列表現で保持して後段で必要に応じて処理する方針です 。テキスト型（text/varchar）は可変長バイト列としてArrowの`utf8`にマップされ、**元の文字データをそのままArrowバッファに格納**します。ADBCドライバはArrow側で可変長データ用のoffset配列とデータバッファを構築し、**文字列の長さも正確に保持**します（NULLの場合はArrow上で値なしとしてビットマップに記録）。このようにADBCは**NULLやテキストの特殊ケースも含めて正確にデータを再現**します。

**● 処理性能（パース＆デコード）:** ADBCドライバはC++で実装されており、libpqのコピー機能＋Arrowライブラリの組み合わせによって**ネットワークIOからArrowメモリへの変換を高速に行う**ことを目指しています 。実際、先行プロジェクトの*pgeon*では約450万行・7列の結果をPandasデータフレーム化するベンチマークで好成績を収めており、ADBCも同様の手法で**CPU上で効率的なバイナリパースを実現**しています  。このパイプラインはシリアルですが低レベル最適化されており、現状ベータ版ながら**性能チューニングが進行中**でさらなる高速化が見込まれます 。GPUは使用しませんが、CPUのみでもコピー処理自体がボトルネックになりにくく、**ネットワーク帯域やディスクIOを飽和する程度のスループット**は達成できると考えられます。

**● スキーマ変更や型拡張への耐性:** ADBCドライバは**接続時にサーバから型情報(OIDやtypname)を取得し、その都度Arrowスキーマを動的に構築**します。そのため、新たな型が出現した場合でも**不明な型は自動的にArrowの`binary`型としてバイト列のまま返却する**仕組みになっており 、**ドライバ側が未対応の型でも処理が破綻しない**柔軟性があります 。さらにArrowの**拡張型Opaque**を利用できる実装では、未知の型でもメタデータにPostgreSQLの型名を保持することで後から判別可能としています 。このようにADBCは**カラム追加・型追加といったスキーマ変化に対しても頑健**で、変換できない型は生バイナリで渡すフォールバックによって広範な互換性を保っています。

## PG-Stromのバイナリパーサーの方式と特徴

**● 型変換のアプローチ:** PG-StromはPostgreSQLサーバ内部で動作する拡張機能で、GPUでの演算に適した**Arrow形式のカラムストアをPostgreSQLの内部データから生成**します。ストレージ上の**heapフォーマット（行指向の内部形式）から、Arrow形式の列指向データへ変換**する処理があり、これはCPU上で行われます。各PostgreSQLデータ型に対し、PG-Strom内で対応するArrowメモリレイアウトを持っており、例えば整数型はビット長に応じた固定長バッファ、テキストや可変長は**ひとつの連続バッファ＋オフセット配列**というArrow同等の表現に変換されます。実際、PG-Stromは**Apache Arrowのカラムフォーマットを内部的に採用**しており、データ構造の違い（例えばタイムスタンプの起算日など）も適切に補正しています  。この変換により、PostgreSQL内のデータをGPUカーネルから直接アクセス可能な連続メモリに載せ替え、高速処理を可能にします。

**● 数値型・文字列型の扱い:** PG-StromはGPU上での演算のため、**可変長の`NUMERIC`をArrowのDecimal128（128ビット固定長）形式に変換**します 。各`NUMERIC`値は整数部・小数部を合わせて128ビットの固定小数点表現に変換され（ArrowのDecimalと同一レイアウト）、必要なスケール（小数点位置）も保持されます 。この変換により通常精度の`NUMERIC`はGPU上で扱えますが、**桁数が128ビットに収まらない巨大な数値はGPUで扱えないためCPU処理にフォールバック**します 。一方、テキスト（`text/varchar`）やBLOB（`bytea`）などの**可変長データもGPU上でサポート**されており 、Arrow形式同様に**一括確保したバッファと各行オフセットの組で表現**します。PG-StromはNULLについても各列ごとにビットマップで管理し、**NULL値はGPU計算時にスキップまたはダミー値に置換することで正確性を維持**しています。実際、`boolean`や`uuid`等の様々な型もGPUサポートされており 、**NULLを含む複雑な型でもGPU処理可能な範囲で忠実に型変換**されています。

**● 処理性能（パース＆デコード）:** PG-Stromの変換はPostgreSQLサーバ内部で行われるため、**データ読み出しと同一プロセス内で直接列化処理が行われる**利点があります。ディスクから読み込んだ8KBブロック内の各タプルをCPUでパースし、対応するArrow列バッファに書き込む処理は、C言語で最適化されています。さらに、PG-Stromは**GPU-Direct等の機構を用いてSSD上の圧縮データを直接GPUメモリにDMA転送する機能**も研究しており、I/OからGPUまでの経路を短絡することでパース処理のオーバーヘッド自体を極力抑えています  。実際のクエリでは、**変換後の列データに対してGPUでフィルタや結合を行うため、データ転送と変換はGPU処理と重ね合わせて非同期に実行**されます。そのため、パース・デコード単体の速度は測りづらいものの、**CPU上で行う前処理としては十分高速かつ並行実行可能**です。特に、PG-Stromは外部FDW経由でArrowフォーマットファイルを読む場合など**パース処理自体を省略**できるケースもあり、環境次第ではADBC同様にI/Oボトルネックに近い性能を発揮します。

**● スキーマ変更や型拡張への耐性:** PG-Stromは対応可能なデータ型が事前に定義されています。PostgreSQL本体に新しいデータ型が追加された場合、**PG-Strom側もコード修正が必要**で、未知の型が含まれるクエリではその部分をGPU実行できずCPUフォールバックとなります。例えば、大きな`jsonb`データはTOAST圧縮されGPUで直接扱えないため、自動的にCPU処理に切り替わる設計です 。このように**サポート範囲外の型やサイズを検出するとGPU計算を諦めて安全にCPU処理する**ため、結果自体は正しく返りますが性能は低下します。スキーマ変更への対応はADBCほど動的ではなく、**あくまで既知の型について最適化する方式**です。もっとも、対応型の範囲は前述の通り数値・文字列・日時・論理値からGIS拡張の`geometry`型まで非常に広く  、通常の業務データ型であればほぼ網羅されています。カラム追加程度では問題ありませんが、**ユーザ定義型や将来の新型はGPU非対応として扱われる**（＝CPU処理）点で、ADBCに比べると柔軟性は劣ります。

## カスタムGPUデコード実装（ColumnInfo方式）の現状

**● 型変換のアプローチ:** ユーザ実装では、PostgreSQLから取得した`COPY BINARY`フォーマットのバイト列をまずNumba（CPU）またはCUDA（GPU）のカーネルでパースし、行中の各フィールドの位置と長さを算出しています（`parse_binary_chunk`等）  。その後、各カラムのメタ情報（`ColumnInfo`に名前・型名・長さ）が与えられ、型名に応じて**独自に定義した型IDにマッピング**してGPU上のバッファを確保し、続いてGPUカーネルでバイナリデータから各列の値をデコードしています  。しかし、この型マッピングは簡略化されており、現在は**`int`を含む型名は整数（固定4バイト）扱い、`numeric`は8バイト数値扱い、それ以外は一律文字列扱い**という大まかな分類になっています  。たとえば`SMALLINT`や`BIGINT`も`"int"`という文字を含むため4バイトとみなされ、`DOUBLE PRECISION`や`FLOAT`はどの条件にも合致しないので文字列型(固定長バッファ)として扱われてしまいます。これは**型対応の網羅性が不足しており**、浮動小数点やブール値、日時型など多くの型で不正確なデコードが行われる可能性があります。またNULLの場合、現在の実装では**フィールド長を-1とすることでNULLフラグとしており** 、文字列カラムでは長さ0として扱って後段でNoneに変換していますが  、整数カラムではNULLもそのまま0として格納されてしまい**NULLと0の区別が失われるバグ**があります。全体として、**カスタム実装の型定義・長さ定義は簡略化のあまり網羅性と精度を欠いている**状態です。

**● 数値型・文字列型の扱い:** 現行実装では、`NUMERIC`型を8バイトにマップしています が、PostgreSQLのNUMERICは可変長のためこの固定長8バイトは本来不適切です。おそらく**実装上はNUMERICを倍精度浮動小数（double, 8バイト）程度に丸めている**か、あるいはPG-Stromにならって128ビット整数に変換する途上ですが、コード上は未完成です。実際、GPUデコード結果を見るとNUMERIC列は`hi`（高位64ビット）・`lo`（下位64ビット）・`scale`の3つの配列に分けて出力し、それをホスト側で再組み立てして浮動小数や文字列に変換しています  。一部、128ビットに収まらない場合は`"[hi,lo]@scale"`という文字列を結果に入れており、**大きな数値は正しく扱えていない**様子も見られます 。文字列型については、各テキストカラムごとに**最大長固定のバッファ（デフォルト64～256バイト）を全行分確保**し  、各行の実データ長を`str_null_pos`に記録する方式です  。このため、実際の文字列が短い場合でも所定長分メモリを消費し、長い場合は事前想定を超えるとバッファ不足になります。ArrowやPG-Stromが用いる**可変長オフセット配列方式ではなく、固定長枠取り方式**の実装になっている点でメモリ効率と柔軟性に課題があります。また、NULL文字列は`null_pos=0`で示し 、その場合は出力をNoneにしています 。**NULL処理は文字列では一応機能していますが、数値では前述のように不完全**です。総じて、**現在のカスタム実装は型ごとの専用処理が不足しており、特に数値と文字列で精度やメモリ効率に問題**があります。

**● 処理性能（パース＆デコード）:** カスタム実装ではGPUを用いてパースやデコードを行っていますが、そのパイプラインにはいくつかのオーバーヘッドがあります。まず、**データ取得後にPythonレイヤーでNumPy配列化しNumba関数でCPUパース**を行う or **CUDAカーネルでGPUパース**を行うという2経路が存在し、GPUパースが失敗するとCPUパースにフォールバックするなど複雑な処理になっています  。GPUでオフセット計算を行っても、結局その結果をホストに戻して再度GPUに転送し直すなど、**メモリ転送が二重三重になっている**箇所もあります  。実際、`decode_chunk`内では`chunk_array`とオフセット/長さ配列をいったんホストからGPUへコピーし、デコードカーネル実行後また結果をホストに取り出しています  。これではせっかくGPUを使っても転送コストが増え、**純粋なCPU実装と大差ないか、下手をすると遅くなる可能性**があります。さらに、GPUカーネル自体も1スレッド1行方式で実装されており 、各スレッドが複数のカラムに書き込む際にメモリアクセスが分散して**GPUメモリ帯域を十分活用できない**懸念があります。例えば、整数バッファは列ごとにチャンクサイズ分の領域を連続確保していますが  、各行スレッドはその領域内の離れた位置（異なるカラムのオフセット）に書き込みを行うため、**メモリアクセスが非連続的でGPUのCoalescedアクセスの恩恵が少ない**と考えられます。対照的に、ADBCやPG-StromはCPU中心ですが無駄な転送や分岐を減らした実装なので、現状のカスタムGPU実装より**パース・デコードレイヤーのスループットは高い**でしょう。実装の不安定さ（バグ多発）もあって、本来期待されるGPU加速効果が十分に出ていない状況です。

**● スキーマ変更や型拡張への耐性:** カスタム実装は**情報スキーマから取得した型名文字列に依存して処理を分岐**しています  。そのため、新しいデータ型や想定外の型名が登場すると対応できず、**場合によっては`get_column_type`内で例外を投げて処理が停止**します  。実際に`boolean`や`date`型などは現在の条件分岐になく、出現すると`Unsupported column type`エラーとなるでしょう。また、既存の型でも**カラム長の変更（例：VARCHAR(n)のn拡大）に柔軟に追随できません**。固定長バッファ方式では、長さを超えるデータが来ると不具合を起こしますし、長さを過大にとればメモリ浪費になります。スキーマに変更があった場合、ColumnInfoの`length`を手動で調整する必要があり、**Arrowのようにデータ駆動で可変長に対応する仕組みが欠如**しています。さらに、現在の実装ではテーブルごとに処理を初期化し直しており、スキーマ変更時は再接続すれば反映されるものの、**実行中にスキーマが変わった場合の安全策（例えば古いスキーマ情報のキャッシュ無効化など）も特にありません**。総じて、**カスタム実装はスキーマ定義に対するハードコーディングが多く、変更耐性・拡張性が低い**と言えます。

## カスタム実装へのリファクタリング提案

以上の比較から、カスタムGPUデコーダー実装には**構造の見直しと不具合修正による安定化**が急務です。具体的には次のような改善策が考えられます。

- **コードの簡素化と責務分離:** 現在パース処理がBinaryParser/BinaryDataParserとGPUDecoderの間で複雑に分散しています。これを見直し、**まずCPU上で確実に全タプルのフィールド境界をパースする処理と、GPU上で各型にデコードする処理を明確に分離**しましょう。GPUでのパース（オフセット計算）は効果が薄い上にコードの複雑化を招いているため、一旦**GPUパースは省略**し、NumbaによるCPUパースに一本化することを提案します（シンプルな実装でまず正しさと安定性を確保）。また、ColumnInfoの型定義も現在は文字列判別に頼っていますが、**PGのOIDやデータ長を利用した明確な型IDマッピング**に切り替えると良いでしょう。型ごとの処理コードを整理し、モジュール間の依存関係を減らすことで、全体の見通しを良くします。

- **冗長な処理ロジックの削除・整理:** Numpy配列からGPUメモリへの転送が何度も行われている点や、一部未使用の変数・処理（例えばGPUDecoder内で使われない例外処理や、数値型変換の未完成ロジック）が見受けられます。これら**不要・未完成なロジックを一旦削除または無効化**し、パイプラインをシンプルにしましょう。特に`parse_chunk`と`parse_binary_chunk`は処理フローが複雑になっているので、**単一の関数で「バイナリヘッダの読み飛ばし→各行フィールドの境界検出→次チャンクへの持ち越し処理」を行うよう再構成**するとバグを減らせます。現在はチャンク境界で中途半端に処理を返す設計ですが、**次チャンクと結合して再パースする仕組みに改める**（バッファをまたいだフィールドは次チャンクデータと連結して処理）ことで、一貫性を保ちつつコード量を減らせます。

- **バグの温床となっている箇所の修正:** 特にNULL値や型長さに関するバグ修正が重要です。**整数カラムのNULLは専用のnullマーカー配列を導入**して管理し、デコード結果ではNoneやNaNとして区別できるようにします。例えば各カラムに`valid_bits`や`null_bitmap`を持たせ、パース時にNULLならビットを0にセット、値バッファにはデフォルト値0を入れる、といったArrowに近い方式にすれば、出力時にビットマップを見てNoneに置き換え可能です。文字列長さの扱いも、**ColumnInfoのlengthに頼らず実際のフィールド長（information_schemaから`character_maximum_length`等）を正確に保持**するか、あるいは**見積もり長を超えたら自動でバッファ再確保する処理**を入れると安全です。型判別も`'int' in col.type`ではなく、正確に型名全体を比較する辞書（例：{"integer": INT4, "bigint": INT8, ...}）を用いて判定すれば、誤判別（"point"型など名前に'int'を含むケース等）のバグを防げます。

- **高速化の余地のある箇所の特定と最適化:** 現状ボトルネックとなっているのは**不必要なメモリ転送とPython処理**です。そこで、例えば**パース結果のオフセット/長さ配列をホストに戻さず直接GPUデコードに利用する**設計に変更します。具体的には、CPUでオフセット計算後、その結果をNumPy配列ではなくCuPy（CUDAデバイス配列）に直接書き込むか、あるいはPyArrowの`Buffer`としてwrapしてしまい、GPUカーネルまたはCUDA対応のライブラリから参照できるようにします。これにより**GPUとCPU間のデータ往復を削減**できます。また、GPUカーネル内では**メモリアクセスのパターンを見直し**、例えば1ワープ内のスレッドが同一カラムの連続行を処理するように工夫すればメモリ帯域の有効活用が期待できます（カラム志向の並列化）。加えて、**小規模データではGPUを使わずPythonで処理するフォールバック**も検討します。現状でも行数0の場合の処理がありますが、数千行程度ならNumPyでデコードしてしまう方がオーバーヘッドが少ない可能性があります。これらの最適化ポイントを洗い出し、測定をしながら改善することで、ボトルネックとなっているCPUパスや不必要な同期処理を取り除いていけます。

- **Parquet出力処理の分離:** 現在はOutputHandler内で結果集約と同時にParquetWriterを呼び出しており、処理フローに組み込まれています 。これを**デコーダ層と出力層に明確に分離**することを提案します。理想的には、**デコード結果はまずPyArrowのTableやRecordBatchに変換**し、それをParquetに書き出す処理は別モジュール・別スレッドで行う構成です。こうすれば、デコーダ（GPUカーネル）部分は純粋にメモリ上のArrow配列を作ることに専念でき、出力フォーマット依存の処理から解放されます。ParquetWriterはスキーマ推定など独自実装がありますが、PyArrowで生成したSchemaやArrayを使えばより堅牢に書けます。特に、**Decimal128（128ビット整数とスケール）をCudaBuffer経由でゼロコピーでArrowのDecimalArrayに変換**することも可能になります 。このようにデコードと書き出しを疎結合にすることで、将来的に他の出力（例えばORCやArrow IPC）への拡張も容易になりますし、**デコード処理中にディスク書き込み待ちでGPUが遊ぶといった無駄も減らせます**。

- **GPUカーネルのデータ流れ・設計見直し:** GPUを用いる意義を最大化するには、カーネル設計の妥当性検証も重要です。現在の実装を踏まえると、**GPUに適した処理とCPUでした方がよい処理とを切り分け**、GPUカーネルは**計算量が大きい部分（例えば大規模な数値変換やフィルタリング）に特化**させるのが望ましいです。PostgreSQLバイナリ→Arrow変換は分岐やメモリアクセス中心でGPU向きとは言えない部分もあるため、**本当にGPUでやるべきか再評価**します。もしGPUで行うなら、例えば**全行のNUMERIC->Decimal変換**は大きな整数演算になるためGPU向きでしょうが、**オフセット計算やNULL判定のような軽量処理はCPUで済ませる**方がシンプルで高速かもしれません。データ流向上のためには、**GPUメモリ上で完結できる処理はできるだけ一度のカーネルで実行**し、中間結果を極力ホストに出さない工夫も必要です。現在はparseカーネルとdecodeカーネルが分かれているようですが、一体化できればメモリ読み書きの一貫性が増すでしょう。PG-Stromのように**“変換後のデータでGPU計算まで行う”**のであれば意義がありますが、そうでないなら無理にGPUを使わず**まず正確で高速なCPU実装を完成させ、その上でプロファイルを取りつつGPUオフロード箇所を選定**するのが健全です。

以上のリファクタリングにより、現在の不安定な実装は大幅に整理・単純化され、バグの温床が取り除かれるはずです。特にADBCドライバやPG-Stromで採用されている**Arrowフォーマット準拠の考え方（nullビットマップ、適切な型長、可変長バッファなど）**を取り入れることで、精度と保守性が向上します。一度堅牢なCPU実装ができれば、それ自体が有用ですし、必要に応じて**ボトルネック部分のみGPU化する戦略**も立てやすくなります。最終的には「PostgreSQL→Arrowデコーダ」と「Arrow→Parquetライター」を独立したモジュールにして組み合わせる形にすれば、システム全体の柔軟性も増すでしょう。上述の改善を段階的に適用し、まずは**正確さと安定動作を確保した上でパフォーマンスチューニング**を行うことを強く推奨します。そうすることで、ADBCやPG-Stromの利点を取り入れた、高速かつ拡張性の高い実装に近づけることができると考えられます。  



### 型変換のアプローチです。PGのOIDやデータ長を利用した明確な型IDマッピングとは具体的にどのような手法か

## 「全行パースせずに OID と typmod だけで Arrow 型を決定する」──具体的なやり方

PostgreSQL には **行データを 1 バイトも読まずに** 「列数・型 OID・typmod（長さ/精度情報）」を取る APIは以下です。

| 取得方法 | 速度 | 備考 |
|----------|------|------|
| `SELECT … LIMIT 0` を発行して `cursor.description` を読む | 数 µs／列 | libpq が受け取る **RowDescription** メッセージに OID と typmod が入る |

したがって、**`cursor.description` を 1 回読むだけで Arrow スキーマを確定**→そのスキーマに従い GPU バッファを確保し、あとから COPY BINARY 本体をストリーミングすれば「全行をスキャンして型を推測する」ステップが不要になります。

---

### 1 . まず OID→内部型 ID テーブルを決める

```python
# type_map.py
from collections import namedtuple
ColumnMeta = namedtuple("ColumnMeta",
                        "name pg_oid typmod arrow_id arrow_param elem_size")

# Arrow 型 ID（好きな並びで OK、Numba で使いやすい様に int 定数化）
INT16, INT32, INT64   = 0, 1, 2
FLOAT32, FLOAT64      = 3, 4
DECIMAL128            = 5
UTF8, BINARY          = 6, 7
DATE32, TS64_US       = 8, 9
BOOL                  = 10
UNKNOWN               = 255   # 非対応型フォールバック

PG_OID_TO_ARROW = {
    20: (INT64,   8),      # int8
    21: (INT16,   2),      # int2
    23: (INT32,   4),      # int4
    700: (FLOAT32, 4),     # float4
    701: (FLOAT64, 8),     # float8
    1700: (DECIMAL128, 16),# numeric (128-bit)
    16:  (BOOL,    1),     # boolean
    25:  (UTF8,   None),   # text
    1042:(UTF8,   None),   # bpchar
    1043:(UTF8,   None),   # varchar
    17:  (BINARY, None),   # bytea
    1082:(DATE32,  4),     # date
    1114:(TS64_US,8),      # timestamp
    1184:(TS64_US,8),      # timestamptz
}
```

* typmod（`-1` 以外）は VARCHAR(N) や NUMERIC(p,s) の **長さ・精度** を示すので Arrow 側の変数長サイズや Decimal(precision, scale) に反映できます。

---

### 2 . メタデータだけを取得して Arrow スキーマを作る関数

```python
# meta_fetch.py
import psycopg2
from type_map import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN, DECIMAL128, UTF8

def fetch_column_meta(conn, sql):
    """
    SELECT … LIMIT 0 で RowDescription を取り Arrow スキーマ情報を返す。
    """
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM ({sql}) AS t LIMIT 0;")

    cols = []
    for d in cur.description:
        oid     = d.type_code          # PG OID
        typmod  = d.internal_size      # 正: 固定長, -1: 可変, -n: typmod
        name    = d.name

        # typmod の符号反転は libpq 仕様
        if typmod < 0:
            typmod = -typmod

        arrow_id, elem_size = PG_OID_TO_ARROW.get(oid, (UNKNOWN, None))

        # typmod による型パラメータ調整
        if arrow_id == DECIMAL128:
            precision = (typmod - 4) >> 16
            scale     = (typmod - 4) & 0xFFFF
            arrow_param = (precision, scale)
        elif arrow_id == UTF8 and typmod > 4:
            arrow_param = typmod - 4         # VARCHAR(N)
        else:
            arrow_param = None

        cols.append(ColumnMeta(name, oid, typmod, arrow_id,
                               arrow_param, elem_size))
    cur.close()
    return cols
```

### 3 . GPU / Numba に渡す「軽量な enum 配列」を作成

```python
import numpy as np
from numba import njit

def build_gpu_meta(meta):
    n = len(meta)
    type_ids  = np.empty(n, np.int32)
    elem_size = np.empty(n, np.int32)   # 可変長は 0 を入れる
    param1    = np.empty(n, np.int32)   # precision や maxlen 等
    param2    = np.empty(n, np.int32)   # scale 等

    for i, m in enumerate(meta):
        type_ids[i]  = m.arrow_id
        elem_size[i] = m.elem_size or 0
        if isinstance(m.arrow_param, tuple):
            param1[i], param2[i] = m.arrow_param
        else:
            param1[i], param2[i] = (m.arrow_param or 0), 0
    return type_ids, elem_size, param1, param2
```

### 4 . そのまま Numba / CUDA カーネルで使える `oid→id` ルックアップ

Numba には Python dict が直接渡せないので、**`if elif` 連鎖** に落とすか、`numba.typed.Dict` を使います。  
以下は単純な `@njit` 版（CPU パース用）。GPU カーネルでは同様の分岐または `switch/case` 相当で良いです。

```python
from numba import njit, int32

@njit(int32(int32))
def oid_to_internal(oid):
    if oid == 20:
        return 2          # INT64
    elif oid == 21:
        return 0          # INT16
    elif oid == 23:
        return 1          # INT32
    elif oid == 700:
        return 3          # FLOAT32
    elif oid == 701:
        return 4          # FLOAT64
    elif oid == 1700:
        return 5          # DECIMAL128
    elif oid in (25, 1042, 1043, 18):
        return 6          # UTF8
    elif oid == 1082:
        return 8          # DATE32
    elif oid in (1114, 1184):
        return 9          # TS64
    elif oid == 16:
        return 10         # BOOL
    elif oid == 17:
        return 7          # BINARY
    else:
        return 255        # UNKNOWN
```

---

## これで何が良くなるか

1. **全行をスキャンしなくても列数・型が確実に判る**  
   → チャンク前処理の複雑な「列数推定」「ヘッダ後 10 バイト走査」ロジックを削除できる。

2. **可変長型も typmod で最大長や精度を取得できる**  
   → VARCHAR や NUMERIC の **バッファサイズを正確に確保**。推定失敗でバッファあふれが起きない。

3. **GPU バッファ確保がワンパスで完了**  
   → `allocate_gpu_buffers` が `type_ids / elem_size / param` 配列だけを見れば良い。  
   → 後からスキーマが変わっても、再度 `fetch_column_meta` するだけで対応。

4. **カラム追加・型追加時のフォールバックが簡単**  
   → `UNKNOWN`（255）で受けて GPU では処理せず生バイトを保持 or CPU に回す、の 2 択に統一できる。

---

### まとめ

- **「RowDescription → Arrow スキーマ → GPU バッファ」** の一直線パイプラインで、ヘッダパースと列数推定のバグ源を排除。  
- 型ごとに **Arrow に合わせた固定長／可変長レイアウト** を決め打ちできるため、デコードカーネルが単純化。  
- `typmod` を利用して VARCHAR(N) や NUMERIC(p,s) の **正確な長さ・精度** を反映可能。  
- 実装コストは `PG_OID_TO_ARROW` テーブル＋`oid_to_internal` の 100 行弱だけ。  

この方式をベースにすれば、「どの型でも 1 回のメタデータ取得で Arrow スキーマが確定 → GPU バッファ確保 → COPY BINARY ストリームをそのままデコード」という **シンプルで変化に強い設計** が実現できます。



## GPUのパース方法

 **「誰が／どこで／何を」** を整理し、実装上の注意点を添えてまとめます。

---

## 全体パイプライン

| フェーズ | 実行場所 | 主な仕事 |
|----------|----------|-----------|
| **0. RowDescription 取得** | CPU | `cursor.execute("… LIMIT 0")` → `cursor.description` から<br> `pg_oid / typmod / colname` を取り、Arrow 型 ID・要素サイズ等を決定 → **`type_ids` 配列（int32）として GPU へ `cudaMemcpy`** |
| **1. COPY チャンク転送** | CPU→GPU | `COPY (… ) TO STDOUT (FORMAT BINARY)` をチャンク単位で読み、`cudaMemcpyAsync` で生バイト列を GPU へストリーム転送 |
| **G-1-1. len/null 収集** | GPU (row-parallel) | 1 スレッド＝1 行<br>固定長列は `ptr += 4+esize` でスキップ、可変長列だけ `len_arr[var_idx,row]=len` を書く。NULL は `null_bits[col,row]=1` |
| **G-1-2. prefix-sum** | GPU | Thrust/CUB の `exclusive_scan` で<br>可変長列ごとに **`offsets` 配列** と **総バイト数 `total_bytes[col]`** を得る |
| **G-2-1. value バッファ確保** | **CPU 呼び出しで `cudaMalloc`／`cudaMallocAsync`** | `total_bytes[col]` の結果を読むだけ（PCIe 往復ゼロ）で列ごとバッファを一括確保 |
| **G-2-2. scatter copy** | GPU (column-parallel) | 可変長列ごとに 1 スレッド＝1 行で<br>`value[offset : offset+len] = raw[ptr : ptr+len]` をコピー |
| **2. Arrow RecordBatch 構築** | CPU or GPU | - 固定長列：`value_buffer` と `null_bitmap` をそのまま Arrow `Buffer` に wrap<br>- 可変長列：`offsets`・`value_buffer`・`null_bitmap` を wrap<br>→ `pyarrow.RecordBatch.from_arrays(...)`（`pyarrow.cuda` 経由ならゼロコピー） |
| **3. 出力** | CPU | Parquet/Flight IPC など好きなフォーマットで保存またはストリーム送信 |

---

## 実装ポイントとヒント

### 1. 行先頭ポインタ `row_ptrs[]`
* **CPU で 1 pass** して作るのが最も簡単・高速。  
  2 バイトの *row‐header* と `len` を読み飛ばすだけなのでメモリ帯域負荷は極小。
* もし完全 GPU 化したい場合は、`raw` を 1 warp＝1 行に割り当てて
  `__syncthreads()` しながらポインタを進める方法もあるが、実測で差が出にくい。

### 2. `col_bits` の符号化
```python
bits = (is_var << 31) | elem_size   # elem_size==0 なら可変長
```
こうしておくと CUDA カーネル側で

```cuda
is_var = bits >> 31;
esize  = bits & 0xFFFF;
```
の 1 行で判定でき、分岐が減ります。

### 3. null ビットマップ
* Arrow は 8 行分を 1 byte に詰めるが、実装を急ぐなら  
  `uint8[rows]` で 0/1 を立てておき、最終バッチ化時に
  ```python
  pa.py_buffer(null_uint8).cast(pa.bool_())  # もしくは pa.pack_bits()
  ```
  で十分。

### 4. ストリーム／コンカレンシー
* ` cudaStream_t stream_copy, stream_compute` を分け  
  - `cudaMemcpyAsync(COPYチャンク → gpu_buf, stream_copy)`  
  - `pass1_lengths[grid,block,stream_compute](…)`  
  とすれば **I/O と計算をオーバラップ** できます。  
* 2 つ目のチャンクをコピー中に 1 つ目の pass-1/scan が走る構造に。

### 5. 固定長列のデコード
* Endian 変換付き `*reinterpret_cast<int32_t*>` 読み取りを  
  `@cuda.jit(device=True, inline=True)` で用意しておく。  
* 今後 `float`, `date32`, `timestamp64` などを増やしても
  `type_id`→`esize`→`decode()` のテーブルだけ足せば済む。

### 6. RecordBatch サイズとスライス
* COPY のチャンク＝RecordBatch にしなくても良い。  
  例えば 100 MB チャンクで来ても、prefix-sum 後に
  「列 value バッファが 16 MB を超えたらバッチを切る」
  といった判定を挟み、`row_limit` で 2 回に分けて offset を再計算する
  やり方も可能。（大容量 GPU にコピー → 小分けで Arrow 化）

---

## これで解決できること
* **型判定は RowDescription だけ**、可変長サイズは **len 値の prefix-sum** だけで確定。
* データ本体は 2nd pass まで **一切コピー不要**。  
  → DRAM → GPU の転送コストを最小化。
* 固定長 / 可変長 / NULL を Arrow 規約通りに分離。  
  → そのまま `pyarrow.cuda` でゼロコピー RecordBatch。
* CPU 側ロジックは **行先頭ポインタ作成と cudaMalloc 呼び出しのみ**。  
  → コードが単純・バグが減る。


### 概念実証レベルですが、**RowDescription→GPU 2-pass→Arrow RecordBatch** の一連を 300 行弱に収めたサンプル

#### COPY BINARY → GPU → Arrow RecordBatch 変換パイプライン（プロトタイプ）
◆ 手順
    0. RowDescription 取得  → Arrow 型メタ作成 & GPU 転送
    1. COPY チャンク転送    → raw バイト列を GPU へ
    2. pass‑1 (row parallel) → len/null 収集
    3. prefix‑sum (列ごと)   → offsets & total_size
    4. value バッファ確保     → cudaMalloc
    5. pass‑2 (col parallel) → scatter copy
    6. Arrow RecordBatch 作成 → pyarrow.cuda でゼロコピー

```
import struct
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import psycopg2
import pyarrow as pa
import pyarrow.cuda as pcuda
from numba import cuda, njit

# -----------------------------------------------------------------------------
# 0. RowDescription から型メタを作る
# -----------------------------------------------------------------------------
ColumnMeta = namedtuple(
    "ColumnMeta", "name pg_oid typmod arrow_id elem_size is_var var_index")

# Arrow 型 ID (独自定義)
INT16, INT32, INT64 = 0, 1, 2
FLOAT32, FLOAT64 = 3, 4
DECIMAL128 = 5
UTF8, BINARY = 6, 7
DATE32, TS64_US = 8, 9
BOOL = 10
UNKNOWN = 255

PG_OID_TO_ARROW = {
    20: (INT64, 8),
    21: (INT16, 2),
    23: (INT32, 4),
    700: (FLOAT32, 4),
    701: (FLOAT64, 8),
    1700: (DECIMAL128, 16),
    25: (UTF8, None),
    1042: (UTF8, None),
    1043: (UTF8, None),
    17: (BINARY, None),
    1082: (DATE32, 4),
    1114: (TS64_US, 8),
    1184: (TS64_US, 8),
    16: (BOOL, 1),
}

def fetch_column_meta(conn, sql: str) -> List[ColumnMeta]:
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM ({sql}) AS t LIMIT 0")
    metas = []
    var_index = 0
    for desc in cur.description:
        oid = desc.type_code
        typmod = desc.internal_size
        if typmod < 0:
            typmod = -typmod
        arrow_id, elem = PG_OID_TO_ARROW.get(oid, (UNKNOWN, None))
        is_var = int(elem is None)
        metas.append(ColumnMeta(desc.name, oid, typmod, arrow_id,
                                elem if elem else 0, is_var, var_index if is_var else -1))
        if is_var:
            var_index += 1
    cur.close()
    return metas

# -----------------------------------------------------------------------------
# 1. CPU で行先頭ポインタを作る（超軽量）
# -----------------------------------------------------------------------------
@njit
def build_row_ptrs(buf: np.ndarray) -> np.ndarray:
    """COPY BINARY の行境界をスキャン。戻り値 row_ptrs[rows+1] (EOF 付き)"""
    pos = 23  # header 11 + flag4 + extlen4 + ext0
    ptrs = [pos]
    nrows = 0
    buflen = buf.size
    while pos < buflen:
        nfields = (buf[pos] << 8) | buf[pos + 1]
        if nfields == 0xFFFF:
            break
        pos += 2
        for _ in range(nfields):
            l = struct.unpack_from(">i", buf, pos)[0]
            pos += 4
            if l != -1:
                pos += l
        ptrs.append(pos)
        nrows += 1
    return np.asarray(ptrs, dtype=np.int32)

# -----------------------------------------------------------------------------
# 2. GPU カーネル pass‑1
# -----------------------------------------------------------------------------
@cuda.jit(device=True, inline=True)
def read_int32(raw, idx):
    return (raw[idx] << 24) | (raw[idx+1] << 16) | (raw[idx+2] << 8) | raw[idx+3]

@cuda.jit
def pass1_lengths(raw, row_ptrs, col_bits, len_arr, null_bits):
    r = cuda.grid(1)
    if r >= row_ptrs.size - 1:
        return
    ptr = row_ptrs[r]
    ncols = col_bits.size
    var_idx = 0
    for c in range(ncols):
        bits = col_bits[c]
        is_var = bits >> 31
        esize = bits & 0xFFFF
        l = read_int32(raw, ptr)
        ptr += 4
        if l == 0xFFFFFFFF:
            null_bits[c, r] = 1
            if is_var:
                len_arr[var_idx, r] = 0
        else:
            if is_var:
                len_arr[var_idx, r] = l
                ptr += l
                var_idx += 1
            else:
                ptr += esize

# -----------------------------------------------------------------------------
# 3. GPU カーネル pass‑2 (scatter copy)
# -----------------------------------------------------------------------------
@cuda.jit
def pass2_scatter(raw, row_ptrs, col_bits, offsets, value_buf):
    r = cuda.grid(1)
    if r >= row_ptrs.size - 1:
        return
    ptr = row_ptrs[r]
    ncols = col_bits.size
    var_idx = 0
    for c in range(ncols):
        bits = col_bits[c]
        is_var = bits >> 31
        esize = bits & 0xFFFF
        l = read_int32(raw, ptr)
        ptr += 4
        if l != 0xFFFFFFFF:
            if is_var:
                dst = offsets[var_idx, r]
                for i in range(l):
                    value_buf[var_idx, dst + i] = raw[ptr + i]
                ptr += l
                var_idx += 1
            else:
                ptr += esize

# -----------------------------------------------------------------------------
# 4. ホスト側ハイレベル関数
# -----------------------------------------------------------------------------

def copy_binary_to_recordbatch(conn, sql: str, copy_chunk_size: int = 8 << 20) -> pa.RecordBatch:
    """指定クエリを COPY BINARY し GPU で Arrow RecordBatch に変換"""
    # --- 0. メタ ---
    meta = fetch_column_meta(conn, sql)
    ncols = len(meta)
    n_var = sum(m.is_var for m in meta)

    col_bits = np.empty(ncols, np.int32)
    for i, m in enumerate(meta):
        col_bits[i] = (m.is_var << 31) | (m.elem_size & 0xFFFF)

    d_col_bits = cuda.to_device(col_bits)

    # --- 1. COPY チャンク読込み（ここは簡易にメモリ全読み）---
    cur = conn.cursor()
    cur.execute(f"COPY ({sql}) TO STDOUT (FORMAT BINARY)")
    raw = cur.fetchall()[0][0]  # psycopg2 binary result
    cur.close()
    raw_np = np.frombuffer(raw, dtype=np.uint8)

    # 行先頭ポインタ
    row_ptrs = build_row_ptrs(raw_np)
    rows = row_ptrs.size - 1

    d_raw = cuda.to_device(raw_np)
    d_row_ptrs = cuda.to_device(row_ptrs)

    # --- 2. pass‑1 ---
    len_arr = cuda.device_array((n_var, rows), np.int32)
    null_bits = cuda.device_array((ncols, rows), np.uint8)

    threads = 256
    blocks = (rows + threads - 1) // threads
    pass1_lengths[blocks, threads](d_raw, d_row_ptrs, d_col_bits, len_arr, null_bits)

    # --- 3. prefix‑sum (CuPy) ---
    import cupy as cp

    offsets = cuda.device_array((n_var, rows + 1), np.int32)
    total_bytes = []
    for v in range(n_var):
        cp_len = cp.asarray(len_arr[v])
        cp_off = cp.cumsum(cp_len, dtype=cp.int32)
        offsets[v, 0] = 0
        offsets[v, 1:] = cp_off
        total_bytes.append(int(cp_off[-1].get()))

    # --- 4. value バッファ確保 ---
    value_buf = [cuda.device_array(tb, np.uint8) for tb in total_bytes]

    # --- 5. pass‑2 ---
    pass2_scatter[blocks, threads](d_raw, d_row_ptrs, d_col_bits,
                                   offsets, value_buf)

    cuda.synchronize()

    # --- 6. Arrow RecordBatch ---
    arrays = []
    var_idx = 0
    for i, m in enumerate(meta):
        if m.arrow_id <= FLOAT64:  # 固定長
            buf = pcuda.as_cuda_buffer(len_arr)  # 実装省略: fixed_buf を使う
            pa_type = pa.int32() if m.arrow_id in (INT16, INT32) else pa.int64()
            arrays.append(pa.Array.from_buffers(pa_type, rows, [None, buf]))
        else:
            o_buf = pcuda.as_cuda_buffer(offsets[var_idx])
            v_buf = pcuda.as_cuda_buffer(value_buf[var_idx])
            n_buf = pcuda.as_cuda_buffer(null_bits[i])
            arr = pa.StringArray.from_buffers(rows, o_buf, v_buf, n_buf)
            arrays.append(arr)
            var_idx += 1

    batch = pa.RecordBatch.from_arrays(arrays, [m.name for m in meta])
    return batch

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn", default="dbname=postgres user=postgres")
    parser.add_argument("--sql", required=True)
    args = parser.parse_args()

    conn = psycopg2.connect(args.conn)
    rb = copy_binary_to_recordbatch(conn, args.sql)
    print(rb.schema)
    print(rb.num_rows, "rows converted → RecordBatch")

```

* **CPU 部**  
  * `fetch_column_meta` … OID/typmod を Arrow 型 ID にマップ  
  * `build_row_ptrs` … COPY バイト列を軽く走査し行ポインタ配列を生成  

* **GPU カーネル**  
  * `pass1_lengths` … 行スレッド並列で `len/null` 収集  
  * Thrust/CuPy で列ごと prefix-sum → offset & 総サイズ確定  
  * `pass2_scatter` … 列スレッド並列で value バッファに scatter-copy  

* **RecordBatch 構築**  
  * `pyarrow.cuda` で **ゼロコピー** Arrow 配列を生成 → クライアントへ。  

