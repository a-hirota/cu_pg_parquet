"""
kvikioを使用したPostgreSQLヒープファイル直接GPU読み込み機能

このモジュールは、kvikioライブラリを使用してPostgreSQLのヒープファイルを
直接GPUメモリに読み込む機能を提供します。
"""

import os
import sys
from typing import Optional
import numpy as np
import cupy as cp
from numba import cuda
import rmm
import kvikio


class HeapFileReaderError(Exception):
    """ヒープファイル読み込み関連のエラー"""
    pass


class KvikioNotInitializedError(HeapFileReaderError):
    """kvikioが初期化されていない場合のエラー"""
    pass


def _ensure_cuda_context():
    """CUDAコンテキストが適切に初期化されていることを確認"""
    try:
        # CUDAデバイスが利用可能かチェック
        if not cuda.is_available():
            raise HeapFileReaderError("CUDAデバイスが利用できません")
        
        # 現在のCUDAコンテキストを取得/作成
        try:
            ctx = cuda.current_context()
        except Exception:
            # コンテキストが存在しない場合は新しく作成
            cuda.select_device(0)
            ctx = cuda.current_context()
        
        # コンテキストマネージャープロトコルが使えない場合はNoneを返す
        if not hasattr(ctx, '__enter__'):
            return None
        
        return ctx
    except Exception as e:
        raise HeapFileReaderError(f"CUDAコンテキストの初期化に失敗しました: {str(e)}")


def _ensure_kvikio_initialized():
    """kvikioが適切に初期化されていることを確認"""
    try:
        # kvikioの初期化状態をチェック
        # kvikioが利用可能かの基本的なチェック
        kvikio_available = hasattr(kvikio, 'CuFile')
        if not kvikio_available:
            raise KvikioNotInitializedError("kvikioが正しくインストールされていません")
        
        return True
    except Exception as e:
        raise KvikioNotInitializedError(f"kvikioの初期化チェックに失敗しました: {str(e)}")


def _ensure_rmm_initialized():
    """RMMメモリプールが初期化されていることを確認"""
    try:
        # RMMの初期化状態をチェック
        current_device = cp.cuda.Device()
        with current_device:
            # 小さなテストバッファを作成して動作確認
            test_buffer = rmm.DeviceBuffer(size=1024)
            del test_buffer  # すぐに解放
        return True
    except Exception as e:
        # RMMが初期化されていない場合は初期化を試行
        try:
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=2**30  # 1GB
            )
            return True
        except Exception as init_error:
            raise HeapFileReaderError(f"RMMの初期化に失敗しました: {str(init_error)}")


def _read_with_kvikio(heap_file_path: str, file_size: int, device_buffer, cuda_ctx):
    """kvikioを使用してファイルを読み込む共通処理"""
    try:
        # CuFileオブジェクトを作成してファイルを開く
        with kvikio.CuFile(heap_file_path, "r") as cufile:
            # ファイル全体をGPUメモリに直接読み込み
            bytes_read = cufile.read(device_buffer)
            
            if bytes_read != file_size:
                raise HeapFileReaderError(
                    f"読み込みサイズが一致しません。期待値: {file_size}, 実際: {bytes_read}"
                )
        
        # DeviceBufferからcuda.devicearrayに変換
        gpu_ptr = device_buffer.ptr
        
        # numpy配列としてラップしてからcuda.devicearrayに変換
        if cuda_ctx is not None:
            gpu_array = cuda.devicearray.DeviceNDArray(
                shape=(file_size,),
                strides=(1,),
                dtype=np.uint8,
                gpu_data=cuda.cudadrv.driver.MemoryPointer(
                    context=cuda_ctx,
                    pointer=gpu_ptr,
                    size=file_size
                )
            )
        else:
            # コンテキストが使用できない場合、デフォルトコンテキストを使用
            gpu_array = cuda.devicearray.DeviceNDArray(
                shape=(file_size,),
                strides=(1,),
                dtype=np.uint8,
                gpu_data=cuda.cudadrv.driver.MemoryPointer(
                    context=cuda.current_context(),
                    pointer=gpu_ptr,
                    size=file_size
                )
            )
        
        return gpu_array
        
    except Exception as kvikio_error:
        # kvikio固有のエラーをキャッチ
        raise HeapFileReaderError(f"kvikioでの読み込みに失敗しました: {str(kvikio_error)}")


def read_heap_file_direct(heap_file_path: str) -> cuda.devicearray:
    """
    kvikioを使用してPostgreSQLヒープファイルを直接GPUメモリに読み込む
    
    この関数は、指定されたパスからPostgreSQLのヒープファイルを読み込み、
    その内容をGPUメモリに直接配置します。ファイル全体を一度に読み込み、
    cuda.devicearrayとして返します。
    
    Args:
        heap_file_path (str): 読み込むPostgreSQLヒープファイルのパス
        
    Returns:
        cuda.devicearray: GPUメモリ上のファイルデータ（uint8配列）
        
    Raises:
        HeapFileReaderError: ファイルが存在しない、読み込みエラーなど
        KvikioNotInitializedError: kvikioが正しく初期化されていない場合
        
    Example:
        >>> gpu_data = read_heap_file_direct("/path/to/postgresql/heap/file")
        >>> print(f"読み込んだデータサイズ: {gpu_data.size} bytes")
    """
    
    # 入力検証
    if not isinstance(heap_file_path, str):
        raise HeapFileReaderError("heap_file_pathは文字列である必要があります")
    
    if not heap_file_path.strip():
        raise HeapFileReaderError("heap_file_pathは空文字列にできません")
    
    # ファイル存在確認
    if not os.path.exists(heap_file_path):
        raise HeapFileReaderError(f"指定されたファイルが存在しません: {heap_file_path}")
    
    if not os.path.isfile(heap_file_path):
        raise HeapFileReaderError(f"指定されたパスはファイルではありません: {heap_file_path}")
    
    # ファイルサイズ取得
    try:
        file_size = os.path.getsize(heap_file_path)
        if file_size == 0:
            raise HeapFileReaderError(f"ファイルサイズが0です: {heap_file_path}")
    except OSError as e:
        raise HeapFileReaderError(f"ファイルサイズの取得に失敗しました: {str(e)}")
    
    # 事前チェック
    _ensure_kvikio_initialized()
    cuda_ctx = _ensure_cuda_context()
    _ensure_rmm_initialized()
    
    try:
        # RMMを使用してGPUメモリバッファを割り当て
        # コンテキストマネージャーが使用可能な場合のみwithを使用
        if cuda_ctx is not None:
            with cuda_ctx:
                device_buffer = rmm.DeviceBuffer(size=file_size)
                gpu_array = _read_with_kvikio(heap_file_path, file_size, device_buffer, cuda_ctx)
                return gpu_array
        else:
            # コンテキストマネージャーが使用できない場合
            device_buffer = rmm.DeviceBuffer(size=file_size)
            gpu_array = _read_with_kvikio(heap_file_path, file_size, device_buffer, None)
            return gpu_array
                
    except rmm.RMMError as e:
        raise HeapFileReaderError(f"RMMメモリ割り当てに失敗しました: {str(e)}")
    except cuda.cudadrv.driver.CudaAPIError as e:
        raise HeapFileReaderError(f"CUDA操作に失敗しました: {str(e)}")
    except Exception as e:
        raise HeapFileReaderError(f"予期しないエラーが発生しました: {str(e)}")


def get_heap_file_info(heap_file_path: str) -> dict:
    """
    ヒープファイルの基本情報を取得する補助関数
    
    Args:
        heap_file_path (str): ヒープファイルのパス
        
    Returns:
        dict: ファイル情報（サイズ、存在確認など）
    """
    info = {
        'exists': False,
        'is_file': False,
        'size': 0,
        'readable': False,
        'path': heap_file_path
    }
    
    try:
        info['exists'] = os.path.exists(heap_file_path)
        if info['exists']:
            info['is_file'] = os.path.isfile(heap_file_path)
            if info['is_file']:
                info['size'] = os.path.getsize(heap_file_path)
                info['readable'] = os.access(heap_file_path, os.R_OK)
    except Exception:
        pass
    
    return info


__all__ = [
    'read_heap_file_direct',
    'get_heap_file_info',
    'HeapFileReaderError',
    'KvikioNotInitializedError'
]