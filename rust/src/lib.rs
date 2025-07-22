use pyo3::prelude::*;

pub mod postgres;
pub mod cuda;
pub mod arrow_builder;
pub mod ffi;

#[pymodule]
fn gpupgparser_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ffi::fetch_postgres_copy_binary_to_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::transfer_to_gpu_numba, m)?)?;
    Ok(())
}
