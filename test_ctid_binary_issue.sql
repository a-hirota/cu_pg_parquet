-- COPYバイナリ形式での問題を調査

-- 1. ワーカー0の最後の行を詳しく確認
SELECT 
    ctid,
    c_custkey,
    length(c_name::text) as name_len,
    octet_length(c_name::bytea) as name_bytes
FROM customer
WHERE ctid >= '(13022,55)'::tid AND ctid < '(13023,3)'::tid
ORDER BY ctid;

-- 2. バイナリCOPYでのデータサイズを推定
-- ワーカー0の最後のページ付近
\copy (SELECT * FROM customer WHERE ctid >= '(13020,1)'::tid AND ctid < '(13023,1)'::tid) TO '/tmp/worker0_end.bin' WITH (FORMAT BINARY);
\copy (SELECT * FROM customer WHERE ctid >= '(13023,1)'::tid AND ctid < '(13026,1)'::tid) TO '/tmp/worker1_start.bin' WITH (FORMAT BINARY);

-- 3. 各ワーカーの実際のバイト数を確認
DO $$
DECLARE
    i integer;
    start_page integer;
    end_page integer;
    row_count bigint;
    total_pages integer := 208371;
    pages_per_worker integer := 13023;
BEGIN
    RAISE NOTICE 'Worker boundaries and row counts:';
    
    FOR i IN 0..15 LOOP
        start_page := i * pages_per_worker;
        IF i = 15 THEN
            end_page := total_pages;
        ELSE
            end_page := (i + 1) * pages_per_worker;
        END IF;
        
        -- 境界での詳細
        IF i < 3 THEN
            -- 境界ページの最後の数行
            EXECUTE format('SELECT ctid, c_custkey FROM customer WHERE ctid >= ''(%s,55)''::tid AND ctid < ''(%s,1)''::tid ORDER BY ctid',
                end_page - 1, end_page)
            USING start_page, end_page;
        END IF;
    END LOOP;
END $$;

-- 4. PostgreSQLのCOPY BINARYヘッダーを確認
-- ヘッダーは固定で、その後にタプル数が来る
SELECT 
    'COPY BINARY header size' as info,
    11 + 4 + 4 * 8 as header_bytes, -- signature(11) + flags(4) + field_count(4) * num_fields
    'Each row has 4-byte length prefix' as note;

-- 5. 問題の核心：行が複数ワーカーにまたがる可能性
-- ワーカー0が書き込んだ最後のバイト位置と、ワーカー1が書き込み始める位置
WITH worker_info AS (
    SELECT 
        0 as worker_id,
        751939 as row_count,
        141 as avg_row_size, -- customerテーブルの平均行サイズ
        751939 * 141 as estimated_bytes
)
SELECT 
    worker_id,
    row_count,
    estimated_bytes,
    estimated_bytes % 65536 as offset_in_64k_block,
    CASE 
        WHEN estimated_bytes % 141 != 0 THEN 'Row boundary likely crossed'
        ELSE 'Clean boundary unlikely'
    END as boundary_status
FROM worker_info;