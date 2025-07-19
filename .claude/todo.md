decimalの精度、スケールが固定
NULL mapの改修。現状ソースを解析＆どうあるべきか。
testを作成したい。どういうテストがあるべきか考えてくれる？

- 毎回改修時に壊れるのでどこが壊れたのか見えるようにしたい。
- 現在--testがあるが、どちらかということこれはtestよりも
tidy(出力結果は変えずに内部のフォーマットのみを変更)
- 関数名は機能,input, outputにしたい。またinput, outputはpostgres, postgres_raw_binary, column_arrows, cudf_format, parquet_format
- 関数名・ファイル名・フォルダ名に決していれたはいけない。optimized, ultra,fast, v2, これらはただの形容詞で何を実施するか何も伝えていないからです。benchmarkこれもダメです。現在、ベンチマークではなく、本番稼働です！
- フォルダ名の変更は大きな変更となりますが妥協してはいけません。
