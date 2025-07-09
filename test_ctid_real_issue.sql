-- ctid範囲分割で本当に行が欠落するケースを探す

-- 重要な発見：SQLでctid範囲分割すると欠落しない（12,030,000行で完璧）
-- しかし、Rust+GPUでは15行欠落する
-- 原因は、バイナリデータの並列書き込みにある可能性

-- 1. PostgreSQL COPY BINARYフォーマットの構造
-- ヘッダー: PGCOPY\n\377\r\n\0 (11バイト)
-- フラグ: 4バイト
-- ヘッダー拡張: 4バイト
-- 各行: 行長(4バイト) + フィールド数(2バイト) + 各フィールド(長さ4バイト + データ)

-- 2. 行境界の問題を検証
-- 64MBバッファで書き込む場合、行の途中で切れる可能性
DO $$
DECLARE
    avg_row_size integer := 141; -- customerテーブルの平均行サイズ
    buffer_size integer := 64 * 1024 * 1024; -- 64MB
    rows_per_buffer integer;
    last_row_offset integer;
BEGIN
    rows_per_buffer := buffer_size / avg_row_size;
    last_row_offset := buffer_size % avg_row_size;
    
    RAISE NOTICE '64MBバッファ分析:';
    RAISE NOTICE '  - 平均行サイズ: % バイト', avg_row_size;
    RAISE NOTICE '  - バッファサイズ: % バイト', buffer_size;
    RAISE NOTICE '  - バッファあたり行数: %', rows_per_buffer;
    RAISE NOTICE '  - 最後の行のオフセット: % バイト', last_row_offset;
    RAISE NOTICE '  - 行境界をまたぐ確率: %％', (last_row_offset::float / avg_row_size * 100)::int;
END $$;

-- 3. 実際の行サイズ分布を確認
WITH row_sizes AS (
    SELECT 
        4 + 2 + -- 行長 + フィールド数
        8 * 4 + -- 各フィールドの長さ情報（8フィールド）
        4 + -- c_custkey (int4)
        4 + octet_length(c_name) + -- c_name (可変長)
        4 + octet_length(c_address) + -- c_address (可変長)
        4 + octet_length(c_city) + -- c_city (可変長)
        4 + octet_length(c_nation) + -- c_nation (可変長)
        4 + octet_length(c_region) + -- c_region (可変長)
        4 + octet_length(c_phone) + -- c_phone (可変長)
        4 + octet_length(c_mktsegment) as total_size -- c_mktsegment (可変長)
    FROM customer
    LIMIT 10000
)
SELECT 
    MIN(total_size) as min_size,
    MAX(total_size) as max_size,
    AVG(total_size)::int as avg_size,
    STDDEV(total_size)::int as stddev_size
FROM row_sizes;

-- 4. 16ワーカーの書き込みパターンをシミュレート
WITH worker_simulation AS (
    SELECT 
        worker_id,
        worker_id * 106000000 as start_offset, -- 約106MBずつ
        CASE 
            WHEN worker_id = 15 THEN 1696192244 -- 実際のファイルサイズ
            ELSE (worker_id + 1) * 106000000
        END as end_offset
    FROM generate_series(0, 15) as worker_id
)
SELECT 
    worker_id,
    start_offset,
    end_offset,
    end_offset - start_offset as size,
    CASE 
        WHEN (end_offset - start_offset) % 141 != 0 THEN '行境界をまたぐ可能性'
        ELSE 'クリーンな境界'
    END as boundary_status
FROM worker_simulation
ORDER BY worker_id;

-- 5. 結論：問題は並列書き込みの競合状態にある
-- 各ワーカーが独立してfetch_addでオフセットを取得
-- タイミングによって以下が発生：
--   1. ワーカーAが64MB書き込み中
--   2. ワーカーBも64MB分のオフセットを取得
--   3. ワーカーAの最後の行が不完全な状態でワーカーBの領域が始まる
--   4. GPUパーサーが不完全な行を検出できず、スキップ

SELECT 
    'ctid範囲分割の問題' as issue,
    'SQLでは問題なし（ページ単位で完全な行が保証される）' as sql_result,
    'バイナリ並列書き込みで行境界の破損が発生' as binary_issue,
    '解決策: 行単位の書き込み、またはロック機構' as solution;