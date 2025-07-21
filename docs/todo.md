- [] 関数名を意味があるものにする。GPUパーサーのparse_binary_chunk_gpu_ultra_fast_v2_lite
what_input_outputでultraとかfastとかv2とかliteは不要。またinput, outputもpostgresなのか、rust_binaryなのか、arrow_like_bufferなのかcudfなのかparquetなのか。名称を統一したい。
- 他におかしい名称はbenchmark(もうベンチマークではない！本番適用！)rustがフォルダ名がおかしい。フォルダ名を変えることは大改修になると思うがぜひ変えたい。        # Rustバイナリ実行
        cmd = ['./rust_bench_optimized/target/release/pg_fast_copy_single_chunk']

- tidy
- この際、アウトプットは変更しないこと。PGMの内部構造をきれいにするだけ。
- 存在しているsrcの処理内容を整理する。
- 関数名に不必要な形容詞がないか確認する。あれば修正。
- 全体を通して機能が似ている処理がないか確認する。
- 似ている処理は統合できないか確認する。
-
