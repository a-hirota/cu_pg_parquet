-- ctid範囲分割で行が欠落する理由を実証するSQL

-- 1. 全体の行数とページ数
SELECT 
    COUNT(*) as total_rows,
    pg_relation_size('customer'::regclass) / 8192 as total_pages
FROM customer;

-- 2. チャンク分割の比較（16ワーカー、1チャンク）
-- 現在のRust実装と同じ方法
DO $$
DECLARE
    total_pages integer;
    pages_per_worker integer;
    worker_rows bigint;
    total_worker_rows bigint := 0;
    i integer;
BEGIN
    -- 総ページ数を取得
    SELECT pg_relation_size('customer'::regclass) / 8192 INTO total_pages;
    pages_per_worker := total_pages / 16;
    
    RAISE NOTICE 'Total pages: %, Pages per worker: %', total_pages, pages_per_worker;
    
    -- 各ワーカーの行数をカウント（Rustと同じロジック）
    FOR i IN 0..15 LOOP
        IF i = 15 THEN
            -- 最後のワーカーは残り全て
            EXECUTE format('SELECT COUNT(*) FROM customer WHERE ctid >= ''(%s,1)''::tid AND ctid < ''(%s,1)''::tid',
                i * pages_per_worker, total_pages)
            INTO worker_rows;
        ELSE
            EXECUTE format('SELECT COUNT(*) FROM customer WHERE ctid >= ''(%s,1)''::tid AND ctid < ''(%s,1)''::tid',
                i * pages_per_worker, (i + 1) * pages_per_worker)
            INTO worker_rows;
        END IF;
        
        total_worker_rows := total_worker_rows + worker_rows;
        RAISE NOTICE 'Worker %: pages % to %, rows: %', 
            i, 
            i * pages_per_worker, 
            CASE WHEN i = 15 THEN total_pages ELSE (i + 1) * pages_per_worker END,
            worker_rows;
    END LOOP;
    
    RAISE NOTICE 'Total rows from workers: %', total_worker_rows;
    RAISE NOTICE 'Expected rows: 12030000';
    RAISE NOTICE 'Missing rows: %', 12030000 - total_worker_rows;
END $$;

-- 3. なぜ欠落するか：ページ境界の詳細調査
WITH page_boundaries AS (
    SELECT 
        page_num,
        MIN(tuple_id) as min_tuple,
        MAX(tuple_id) as max_tuple,
        COUNT(*) as rows_in_page
    FROM (
        SELECT 
            (ctid::text::point)[0]::int as page_num,
            (ctid::text::point)[1]::int as tuple_id
        FROM customer
        WHERE (ctid::text::point)[0]::int IN (13022, 13023, 26045, 26046)
    ) t
    GROUP BY page_num
)
SELECT * FROM page_boundaries ORDER BY page_num;

-- 4. 問題の実証：ctid < '(page,1)' は page-1 の最後のタプルまで
-- ページ13023の境界を調査
SELECT 
    'Rows with ctid < (13023,1)' as description,
    COUNT(*) as count,
    MAX(ctid) as max_ctid
FROM customer
WHERE ctid < '(13023,1)'::tid

UNION ALL

SELECT 
    'Rows in pages 0-13022' as description,
    COUNT(*) as count,
    MAX(ctid) as max_ctid
FROM customer
WHERE (ctid::text::point)[0]::int <= 13022;

-- 5. 各ページの最後のタプルIDを確認（ワーカー境界付近）
WITH worker_boundaries AS (
    SELECT 13022 as page UNION ALL
    SELECT 26045 UNION ALL
    SELECT 39068 UNION ALL
    SELECT 52091
)
SELECT 
    w.page as boundary_page,
    (SELECT MAX((ctid::text::point)[1]::int) 
     FROM customer 
     WHERE (ctid::text::point)[0]::int = w.page) as last_tuple_in_page,
    (SELECT COUNT(*) 
     FROM customer 
     WHERE (ctid::text::point)[0]::int = w.page) as rows_in_page
FROM worker_boundaries w
ORDER BY w.page;