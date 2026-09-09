use jpegxl_sys::decode::JxlDecoderDestroy;
use jpegxl_sys::encoder::encode::JxlEncoderDestroy;
use jpegxl_sys::threads::thread_parallel_runner::JxlThreadParallelRunnerDestroy;

macro_rules! define_guard {
    ($name:ident, $destroy:path) => {
        pub struct $name<T>(pub *mut T);
        impl<T> Drop for $name<T> {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe { $destroy(self.0 as _) }
                }
            }
        }
    };
}

define_guard!(EncoderGuard, JxlEncoderDestroy);
define_guard!(DecoderGuard, JxlDecoderDestroy);
define_guard!(RunnerGuard, JxlThreadParallelRunnerDestroy);
