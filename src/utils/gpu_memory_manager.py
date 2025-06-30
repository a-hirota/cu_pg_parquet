"""
GPUメモリ管理の改善ユーティリティ
フラグメンテーションを最小化し、大規模テーブル処理時のOOMを防ぐ
"""
import rmm
import cupy as cp
import gc
import warnings
from typing import Optional, Dict, Any


class GPUMemoryManager:
    """改善されたGPUメモリ管理クラス"""
    
    def __init__(self, strategy: str = 'arena'):
        """
        Args:
            strategy: 'arena', 'async', 'pool', 'binning'のいずれか
        """
        self.strategy = strategy
        self.original_mr = None
        self.memory_resource = None
        
    def setup_memory_resource(self) -> None:
        """メモリリソースの初期化"""
        try:
            # 現在のメモリリソースを保存
            self.original_mr = rmm.mr.get_current_device_resource()
            
            # GPUメモリ情報を取得
            gpu_memory = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
            
            if self.strategy == 'arena':
                # Arena Memory Resource（フラグメンテーション回避に最適）
                upstream = rmm.mr.CudaMemoryResource()
                self.memory_resource = rmm.mr.ArenaMemoryResource(
                    upstream_mr=upstream,
                    arena_size=None,  # デフォルトで利用可能メモリの半分
                    dump_log_on_failure=True  # デバッグ用
                )
                print(f"✅ Arena Memory Resource を設定（フラグメンテーション対策）")
                
            elif self.strategy == 'async':
                # CUDA Async Memory Resource（メモリをシステムに返却可能）
                # CUDA 11.2以上が必要
                self.memory_resource = rmm.mr.CudaAsyncMemoryResource()
                print(f"✅ CUDA Async Memory Resource を設定（動的メモリ管理）")
                
            elif self.strategy == 'pool':
                # Pool Memory Resource（従来方式の改善版）
                # より保守的な設定（70%初期、85%最大）
                self.memory_resource = rmm.mr.PoolMemoryResource(
                    upstream_mr=rmm.mr.CudaMemoryResource(),
                    initial_pool_size=int(gpu_memory * 0.7),
                    maximum_pool_size=int(gpu_memory * 0.85)
                )
                print(f"✅ Pool Memory Resource を設定（初期70%, 最大85%）")
                
            elif self.strategy == 'binning':
                # Binning Memory Resource（様々なサイズのアロケーションに対応）
                pool_mr = rmm.mr.PoolMemoryResource(
                    upstream_mr=rmm.mr.CudaMemoryResource(),
                    initial_pool_size=int(gpu_memory * 0.6),
                    maximum_pool_size=int(gpu_memory * 0.8)
                )
                self.memory_resource = rmm.mr.BinningMemoryResource(
                    upstream_mr=pool_mr,
                    min_size_exponent=10,  # 1KB
                    max_size_exponent=26   # 64MB
                )
                print(f"✅ Binning Memory Resource を設定（サイズ別最適化）")
                
            # メモリリソースを設定
            rmm.mr.set_current_device_resource(self.memory_resource)
            
        except Exception as e:
            warnings.warn(f"メモリリソース設定エラー: {e}, デフォルトに戻します")
            # エラー時は従来の方式にフォールバック
            self._setup_fallback_pool()
    
    def _setup_fallback_pool(self) -> None:
        """フォールバック用の標準プール設定"""
        try:
            gpu_memory = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=int(gpu_memory * 0.5),
                maximum_pool_size=int(gpu_memory * 0.7)
            )
            print("⚠️ フォールバック: 標準プール（50%-70%）を使用")
        except Exception as e:
            print(f"❌ RMM初期化エラー: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """現在のGPUメモリ使用状況を取得"""
        try:
            # CuPyメモリプール情報
            mempool = cp.get_default_memory_pool()
            used_bytes = mempool.used_bytes()
            total_bytes = mempool.total_bytes()
            
            # CUDA Runtime情報
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            
            # RMM情報（利用可能な場合）
            rmm_info = {}
            if rmm.is_initialized():
                try:
                    info = rmm.get_info()
                    rmm_info = {
                        'rmm_total_bytes': info.total_bytes / 1024**3,
                        'rmm_free_bytes': info.free_bytes / 1024**3 if hasattr(info, 'free_bytes') else 0
                    }
                except:
                    pass
            
            return {
                'cupy_used_gb': used_bytes / 1024**3,
                'cupy_total_gb': total_bytes / 1024**3,
                'cuda_free_gb': free_mem / 1024**3,
                'cuda_total_gb': total_mem / 1024**3,
                'cuda_used_gb': (total_mem - free_mem) / 1024**3,
                'utilization': (total_mem - free_mem) / total_mem * 100,
                **rmm_info
            }
        except Exception as e:
            return {'error': str(e)}
    
    def aggressive_cleanup(self) -> None:
        """積極的なメモリクリーンアップ"""
        try:
            # CuPyメモリプールをクリア
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            # 使用中のメモリを記録
            before_used = mempool.used_bytes()
            
            # メモリブロックを解放
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            # Python GCを強制実行
            gc.collect()
            
            # CUDAデバイスを同期
            cp.cuda.Stream.null.synchronize()
            
            # 解放されたメモリ量を計算
            after_used = mempool.used_bytes()
            freed_gb = (before_used - after_used) / 1024**3
            
            if freed_gb > 0:
                print(f"🧹 メモリクリーンアップ: {freed_gb:.2f} GB 解放")
                
        except Exception as e:
            warnings.warn(f"メモリクリーンアップエラー: {e}")
    
    def log_memory_status(self, label: str = "") -> None:
        """メモリ状況をログ出力"""
        info = self.get_memory_info()
        if 'error' not in info:
            print(f"\n{'='*60}")
            print(f"📊 GPUメモリ状況 {label}")
            print(f"{'='*60}")
            print(f"CUDA使用量: {info['cuda_used_gb']:.2f}/{info['cuda_total_gb']:.2f} GB ({info['utilization']:.1f}%)")
            print(f"CuPy Pool: {info['cupy_used_gb']:.2f}/{info['cupy_total_gb']:.2f} GB")
            if 'rmm_total_bytes' in info:
                print(f"RMM使用量: {info['rmm_total_bytes']:.2f} GB")
            print(f"{'='*60}\n")
    
    def reset(self) -> None:
        """メモリリソースをリセット"""
        if self.original_mr is not None:
            try:
                rmm.mr.set_current_device_resource(self.original_mr)
                print("メモリリソースをリセットしました")
            except:
                pass


# 使いやすいグローバル関数
def setup_gpu_memory(strategy: str = 'arena') -> GPUMemoryManager:
    """GPUメモリ管理のセットアップ"""
    manager = GPUMemoryManager(strategy)
    manager.setup_memory_resource()
    return manager


def cleanup_gpu_memory() -> None:
    """GPUメモリの積極的クリーンアップ"""
    manager = GPUMemoryManager()
    manager.aggressive_cleanup()