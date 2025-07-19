1. decimalの精度、スケールが固定
2. NULL mapの改修。現状ソースを解析＆どうあるべきか。

3. testを作成したい。どういうテストがあるべきか考えてくれる？
   現在の処理は大きく4つに分かれている。
   1. Postgresからpostgres_raw_binaryを/dev/shmにあるキューにためていく。
   2. キューから取り出し、kvikioを利用してGPUメモリに転送する。
   3. GPUメモリ上のpostgres_raw_binaryからrow offsetとfield indicesを作成する。そのindexを利用してpostgres_raw_binaryからcolumn_arrowsを作成する。column_arrows作成は固定長のarrow配列を作成する処理と、可変長を作成する処理がある。
   4. column_arrowをcudfを利用してcudf dataframeに変換し、cudfのto_parquetを利用してparquetファイルとして出力する。

  それぞれのtestを実装したい。どのようなテストケースがあるだろうか。上記の機能について機能毎のE2Eテストができればよいと思 う。テスト対象は全Postgresの属性と、属性に応じた境界テスト、そして性能テストだろう。具体的なテストケースを記載できますか？

- 毎回改修時に壊れるのでどこが壊れたのか見えるようにしたい。
- 現在--testがあるが、どちらかということこれはtestよりも

4. tidy(出力結果は変えずに内部のフォーマットのみを変更)

- 関数名は機能,input, outputにしたい。またinput, outputはpostgres, postgres_raw_binary, column_arrows, cudf_format, parquet_format
- 関数名・ファイル名・フォルダ名に決していれたはいけない。optimized, ultra,fast, v2, これらはただの形容詞で何を実施するか何も伝えていないからです。benchmarkこれもダメです。現在、ベンチマークではなく、本番稼働です！
- フォルダ名の変更は大きな変更となりますが妥協してはいけません。
