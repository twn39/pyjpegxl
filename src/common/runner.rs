use jpegxl_rs::ThreadsRunner;
use jpegxl_sys::threads::thread_parallel_runner::{
    JxlThreadParallelRunnerCreate, JxlThreadParallelRunnerDefaultNumWorkerThreads,
    JxlThreadParallelRunnerDestroy,
};
use pyo3::prelude::*;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

static GLOBAL_NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

#[pyfunction]
pub fn set_num_threads(n: usize) {
    GLOBAL_NUM_THREADS.store(n, Ordering::SeqCst);
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

pub fn with_threads_runner<R>(f: impl FnOnce(Option<&ThreadsRunner<'static>>) -> R) -> R {
    let threads = GLOBAL_NUM_THREADS.load(Ordering::Relaxed);
    if threads == 1 {
        return f(None);
    }

    let effective = get_effective_threads();
    TLS_THREADS_RUNNER.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if let Some((cached_threads, ref runner)) = *borrow {
            if cached_threads == effective {
                return f(Some(runner));
            }
        }

        let new_runner = ThreadsRunner::new(None, Some(effective));
        if let Some(runner) = new_runner {
            *borrow = Some((effective, runner));
            let runner_ref = borrow.as_ref().map(|(_, r)| r);
            f(runner_ref)
        } else {
            *borrow = None;
            f(None)
        }
    })
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

use std::cell::RefCell;

thread_local! {
    static TLS_RUNNER: RefCell<Option<(usize, Arc<SharedCRunner>)>> = const { RefCell::new(None) };
    static TLS_THREADS_RUNNER: RefCell<Option<(usize, ThreadsRunner<'static>)>> = const { RefCell::new(None) };
}

pub fn get_shared_c_runner() -> Option<Arc<SharedCRunner>> {
    let threads = GLOBAL_NUM_THREADS.load(Ordering::Relaxed);
    if threads == 1 {
        return None;
    }

    let effective = get_effective_threads();
    TLS_RUNNER.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if let Some((cached_threads, ref runner)) = *borrow {
            if cached_threads == effective {
                return Some(runner.clone());
            }
        }

        let ptr = unsafe { JxlThreadParallelRunnerCreate(ptr::null(), effective) };
        if ptr.is_null() {
            return None;
        }

        let shared = Arc::new(SharedCRunner { ptr });
        *borrow = Some((effective, shared.clone()));
        Some(shared)
    })
}
