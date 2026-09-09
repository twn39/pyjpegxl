use jpegxl_rs::ThreadsRunner;
use jpegxl_sys::threads::thread_parallel_runner::{
    JxlThreadParallelRunnerCreate, JxlThreadParallelRunnerDefaultNumWorkerThreads,
    JxlThreadParallelRunnerDestroy,
};
use pyo3::prelude::*;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static GLOBAL_NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

#[pyfunction]
pub fn set_num_threads(n: usize) {
    GLOBAL_NUM_THREADS.store(n, Ordering::SeqCst);
    if let Ok(mut lock) = get_runner_cache().lock() {
        *lock = None;
    }
}

#[pyfunction]
pub fn get_num_threads() -> usize {
    GLOBAL_NUM_THREADS.load(Ordering::SeqCst)
}

pub fn get_effective_threads() -> usize {
    let threads = GLOBAL_NUM_THREADS.load(Ordering::Relaxed);
    if threads == 0 {
        unsafe { JxlThreadParallelRunnerDefaultNumWorkerThreads() }
    } else {
        threads
    }
}

pub fn get_threads_runner() -> Option<ThreadsRunner<'static>> {
    let threads = GLOBAL_NUM_THREADS.load(Ordering::Relaxed);
    if threads == 1 {
        None
    } else if threads > 1 {
        ThreadsRunner::new(None, Some(threads))
    } else {
        Some(ThreadsRunner::default())
    }
}

pub struct SharedCRunner {
    ptr: *mut c_void,
}

unsafe impl Send for SharedCRunner {}
unsafe impl Sync for SharedCRunner {}

impl Drop for SharedCRunner {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                JxlThreadParallelRunnerDestroy(self.ptr);
            }
        }
    }
}

impl SharedCRunner {
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

fn get_runner_cache() -> &'static Mutex<Option<Arc<SharedCRunner>>> {
    static RUNNER_CACHE: OnceLock<Mutex<Option<Arc<SharedCRunner>>>> = OnceLock::new();
    RUNNER_CACHE.get_or_init(|| Mutex::new(None))
}

pub fn get_shared_c_runner() -> Option<Arc<SharedCRunner>> {
    let threads = GLOBAL_NUM_THREADS.load(Ordering::Relaxed);
    if threads == 1 {
        return None;
    }

    let cache = get_runner_cache();
    let mut lock = cache.lock().ok()?;
    if let Some(ref runner) = *lock {
        return Some(runner.clone());
    }

    let effective = get_effective_threads();
    let ptr = unsafe { JxlThreadParallelRunnerCreate(ptr::null(), effective) };
    if ptr.is_null() {
        return None;
    }

    let shared = Arc::new(SharedCRunner { ptr });
    *lock = Some(shared.clone());
    Some(shared)
}
