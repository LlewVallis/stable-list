macro_rules! __if {
    ((false); $($body:tt)*) => {};
    (($($cond:tt)*); $($body:tt)*) => {
        $($body)*
    };
}

macro_rules! impl_iter {
    (
        on = $name:ident { $($params:tt)* } where { $($bounds:tt)* };
        inner = $inner:ident;
        item = { $($item:tt)* };
        map = $map:tt;
        double_ended = $double_ended:tt;
        fused = $fused:tt;
        exact_size = $exact_size:tt;
        clone = $clone:tt;
        force_sync = $sync:tt;
        force_send = $send:tt;
    ) => {
        impl<$($params)*> ::core::iter::Iterator for $name<$($params)*> where $($bounds)* {
            type Item = $($item)*;

            fn next(&mut self) -> Option<Self::Item> {
                self.nth(0)
            }

            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                let inner = self.$inner.nth(n)?;
                ($map)(self, inner)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.$inner.size_hint()
            }
        }

        crate::util::__if! {
            ($double_ended);
            impl<$($params)*> ::core::iter::DoubleEndedIterator for $name<$($params)*> where $($bounds)* {
                fn next_back(&mut self) -> Option<Self::Item> {
                    self.nth_back(0)
                }

                fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                    let inner = self.$inner.nth_back(n)?;
                    ($map)(self, inner)
                }
            }
        }

        crate::util::__if! {
            ($fused);
            impl<$($params)*> ::core::iter::FusedIterator for $name<$($params)*> where $($bounds)* {}
        }

        crate::util::__if! {
            ($exact_size);
            impl<$($params)*> ::core::iter::ExactSizeIterator for $name<$($params)*> where $($bounds)* {}
        }

        crate::util::__if! {
            ($clone);
            impl<$($params)*> ::core::clone::Clone for $name<$($params)*> where $($bounds)* {
                fn clone(&self) -> Self {
                    ($clone)(self)
                }
            }
        }

        crate::util::__if! {
            ($clone);
            unsafe impl<$($params)*> ::core::marker::Sync for $name<$($params)*> where $($bounds)* {}
        }

        crate::util::__if! {
            ($clone);
            unsafe impl<$($params)*> ::core::marker::Send for $name<$($params)*> where $($bounds)* {}
        }
    };
}

pub(crate) use __if;
pub(crate) use impl_iter;

macro_rules! assume_assert {
    ($cond:expr) => {{
        let cond = $cond;
        #[cfg(debug_assertions)]
        assert!(cond);
        $crate::util::assume(cond);
    }};
}

#[inline(always)]
pub unsafe fn assume(condition: bool) {
    if !condition {
        core::hint::unreachable_unchecked();
    }
}

pub(crate) use assume_assert;

pub trait UnwrapExt<T> {
    unsafe fn unwrap_assume(self) -> T;
}

impl<T> UnwrapExt<T> for Option<T> {
    unsafe fn unwrap_assume(self) -> T {
        assume_assert!(self.is_some());
        self.unwrap_unchecked()
    }
}

impl<T, E> UnwrapExt<T> for Result<T, E> {
    unsafe fn unwrap_assume(self) -> T {
        assume_assert!(self.is_ok());
        self.unwrap_unchecked()
    }
}
