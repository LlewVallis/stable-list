use core::{mem, ptr};
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::ops::Range;
use core::ptr::NonNull;

use allocator_api2::alloc::{Allocator, Global};

use crate::{DefaultGrowthStrategy, GrowthStrategy, StableList};
use crate::util::{assume_assert, impl_iter};

pub(super) struct RawChunksIter<'a, T, S: GrowthStrategy<T>> {
    blocks_ptr: NonNull<*const [T]>,
    indices: Range<usize>,
    last_index: usize,
    last_len: usize,
    _marker: PhantomData<(&'a (), S)>,
}

impl<'a, T, S: GrowthStrategy<T>> RawChunksIter<'a, T, S> {
    pub(super) fn new<A: Allocator>(list: &'a StableList<T, S, A>) -> Self {
        unsafe {
            let (last_index, last_len) = if list.used_blocks > 0 {
                let elements_before = list.strategy.cumulative_capacity(list.used_blocks - 1);
                (list.used_blocks - 1, list.len - elements_before)
            } else {
                assume_assert!(list.len == 0);
                (0, 0)
            };

            let (blocks_ptr, _, _) = list.blocks_raw_parts;

            Self {
                blocks_ptr,
                indices: 0..list.used_blocks,
                last_index,
                last_len,
                _marker: PhantomData,
            }
        }
    }
}

impl_iter! {
    on = RawChunksIter { 'a, T, S } where { S: GrowthStrategy<T> };
    inner = indices;
    item = { NonNull<[T]> };
    map = { |this: &mut Self, index: usize| unsafe {
        let block_ptr = *this.blocks_ptr.as_ptr().add(index);
        let block_cap = (*block_ptr).len();

        let block_len = if index == this.last_index {
            this.last_len
        } else {
            block_cap
        };

        assume_assert!(block_len > 0);

        let ptr = ptr::slice_from_raw_parts_mut(block_ptr as *mut T, block_len);
        Some(NonNull::new_unchecked(ptr))
    }};
    double_ended = true;
    fused = true;
    exact_size = true;
    clone = { |this: &Self| Self {
        blocks_ptr: this.blocks_ptr,
        indices: this.indices.clone(),
        last_index: this.last_index,
        last_len: this.last_len,
        _marker: PhantomData,
    }};
    force_sync = true;
    force_send = true;
}

/// Returned by [`StableList::chunks_iter`].
pub struct ChunksIter<'a, T: 'a, S: GrowthStrategy<T> = DefaultGrowthStrategy<T>> {
    raw: RawChunksIter<'a, T, S>,
    _marker: PhantomData<&'a (T, S)>,
}

impl<'a, T: 'a, S: GrowthStrategy<T>> ChunksIter<'a, T, S> {
    pub(super) fn new<A: Allocator>(list: &'a StableList<T, S, A>) -> Self {
        Self {
            raw: RawChunksIter::new(list),
            _marker: PhantomData,
        }
    }
}

impl_iter! {
    on = ChunksIter { 'a, T, S } where { T: 'a, S: GrowthStrategy<T> };
    inner = raw;
    item = { &'a [T] };
    map = { |_this: &mut Self, ptr: NonNull<[T]>|
        unsafe {
            Some(ptr.as_ref())
        }
    };
    double_ended = true;
    fused = true;
    exact_size = true;
    clone = { |this: &Self| Self {
        raw: this.raw.clone(),
        _marker: PhantomData,
    }};
    force_sync = false;
    force_send = false;
}

/// Returned by [`StableList::chunks_iter_mut`].
pub struct ChunksIterMut<'a, T: 'a, S: GrowthStrategy<T> = DefaultGrowthStrategy<T>> {
    raw: RawChunksIter<'a, T, S>,
    _marker: PhantomData<&'a mut (T, S)>,
}

impl<'a, T: 'a, S: GrowthStrategy<T>> ChunksIterMut<'a, T, S> {
    pub(super) fn new<A: Allocator>(list: &'a mut StableList<T, S, A>) -> Self {
        Self {
            raw: RawChunksIter::new(list),
            _marker: PhantomData,
        }
    }
}

impl_iter! {
    on = ChunksIterMut { 'a, T, S } where { T: 'a, S: GrowthStrategy<T> };
    inner = raw;
    item = { &'a mut [T] };
    map = { |_this: &mut Self, mut ptr: NonNull<[T]>|
        unsafe {
            Some(ptr.as_mut())
        }
    };
    double_ended = true;
    fused = true;
    exact_size = true;
    clone = false;
    force_sync = false;
    force_send = false;
}

pub(super) struct RawIter<'a, T, S: GrowthStrategy<T> = DefaultGrowthStrategy<T>> {
    chunks: RawChunksIter<'a, T, S>,
    len: usize,
    front: Range<NonNull<T>>,
    back: Range<NonNull<T>>,
}

impl<'a, T, S: GrowthStrategy<T>> RawIter<'a, T, S> {
    pub(super) fn new<A: Allocator>(list: &'a StableList<T, S, A>) -> Self {
        let empty = NonNull::dangling()..NonNull::dangling();

        Self {
            chunks: RawChunksIter::new(list),
            len: list.len,
            front: empty.clone(),
            back: empty,
        }
    }
}

impl<'a, T, S: GrowthStrategy<T>> Iterator for RawIter<'a, T, S> {
    type Item = NonNull<T>;

    fn next(&mut self) -> Option<NonNull<T>> {
        #[allow(clippy::iter_nth_zero)]
        self.nth(0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    fn nth(&mut self, n: usize) -> Option<NonNull<T>> {
        let mut remaining = n + 1;

        if remaining > self.len {
            self.len = 0;
            return None;
        } else {
            self.len -= remaining;
        }

        if mem::size_of::<T>() == 0 {
            return Some(NonNull::dangling());
        }

        loop {
            unsafe {
                let mut range = &mut self.front;

                if range.is_empty() {
                    if let Some(next_chunk) = self.chunks.next() {
                        let base = next_chunk.as_ptr() as *mut T;
                        let len = next_chunk.len();

                        range.start = NonNull::new_unchecked(base);
                        range.end = NonNull::new_unchecked(base.add(len));
                    } else {
                        range = &mut self.back;
                    }
                }

                assume_assert!(!range.is_empty());

                let base = range.start.as_ptr();
                let len = range.end.as_ptr().offset_from(base) as usize;

                assume_assert!(len > 0);

                let offset = len.min(remaining);
                remaining -= offset;

                let new_base = base.add(offset);
                range.start = NonNull::new_unchecked(new_base);

                if remaining == 0 {
                    return Some(NonNull::new_unchecked(new_base.sub(1)));
                }
            }
        }
    }
}

impl<'a, T, S: GrowthStrategy<T>> DoubleEndedIterator for RawIter<'a, T, S> {
    fn next_back(&mut self) -> Option<NonNull<T>> {
        self.nth_back(0)
    }

    fn nth_back(&mut self, n: usize) -> Option<NonNull<T>> {
        let mut remaining = n + 1;

        if remaining > self.len {
            self.len = 0;
            return None;
        } else {
            self.len -= remaining;
        }

        if mem::size_of::<T>() == 0 {
            return Some(NonNull::dangling());
        }

        loop {
            unsafe {
                let mut range = &mut self.back;

                if range.is_empty() {
                    if let Some(next_chunk) = self.chunks.next_back() {
                        let base = next_chunk.as_ptr() as *mut T;
                        let len = next_chunk.len();

                        range.start = NonNull::new_unchecked(base);
                        range.end = NonNull::new_unchecked(base.add(len));
                    } else {
                        range = &mut self.front;
                    }
                }

                assume_assert!(!range.is_empty());

                let end = range.end.as_ptr();
                let len = end.offset_from(range.start.as_ptr()) as usize;

                assume_assert!(len > 0);

                let offset = len.min(remaining);
                remaining -= offset;

                assume_assert!(offset > 0);

                let new_end = NonNull::new_unchecked(end.sub(offset));
                range.end = new_end;

                if remaining == 0 {
                    return Some(new_end);
                }
            }
        }
    }
}

impl<'a, T, S: GrowthStrategy<T>> FusedIterator for RawIter<'a, T, S> {}

impl<'a, T, S: GrowthStrategy<T>> ExactSizeIterator for RawIter<'a, T, S> {}

impl<'a, T, S: GrowthStrategy<T>> Clone for RawIter<'a, T, S> {
    fn clone(&self) -> Self {
        Self {
            chunks: self.chunks.clone(),
            len: self.len,
            front: self.front.clone(),
            back: self.back.clone(),
        }
    }
}

unsafe impl<'a, T, S: GrowthStrategy<T>> Sync for RawIter<'a, T, S> {}

unsafe impl<'a, T, S: GrowthStrategy<T>> Send for RawIter<'a, T, S> {}

/// Returned by [`StableList::iter`].
pub struct Iter<'a, T: 'a, S: GrowthStrategy<T> = DefaultGrowthStrategy<T>> {
    raw: RawIter<'a, T, S>,
    _marker: PhantomData<&'a T>,
}

impl_iter! {
    on = Iter { 'a, T, S } where { T: 'a, S: GrowthStrategy<T> };
    inner = raw;
    item = { &'a T };
    map = { |_this: &mut Self, ptr: NonNull<T>|
        unsafe {
            Some(ptr.as_ref())
        }
    };
    double_ended = true;
    fused = true;
    exact_size = true;
    clone = { |this: &Self| Self {
        raw: this.raw.clone(),
        _marker: PhantomData,
    }};
    force_sync = false;
    force_send = false;
}

impl<'a, T, S: GrowthStrategy<T>> Iter<'a, T, S> {
    pub(super) fn new<A: Allocator>(list: &'a StableList<T, S, A>) -> Self {
        Self {
            raw: RawIter::new(list),
            _marker: PhantomData,
        }
    }
}

/// Returned by [`StableList::iter_mut`].
pub struct IterMut<'a, T: 'a, S: GrowthStrategy<T> = DefaultGrowthStrategy<T>> {
    raw: RawIter<'a, T, S>,
    _marker: PhantomData<&'a mut (T, S)>,
}

impl<'a, T, S: GrowthStrategy<T>> IterMut<'a, T, S> {
    pub(super) fn new<A: Allocator>(list: &'a mut StableList<T, S, A>) -> Self {
        Self {
            raw: RawIter::new(list),
            _marker: PhantomData,
        }
    }
}

impl_iter! {
    on = IterMut { 'a, T, S } where { T: 'a, S: GrowthStrategy<T> };
    inner = raw;
    item = { &'a mut T };
    map = { |_this: &mut Self, mut ptr: NonNull<T>|
        unsafe {
            Some(ptr.as_mut())
        }
    };
    double_ended = true;
    fused = true;
    exact_size = true;
    clone = false;
    force_sync = false;
    force_send = false;
}

/// Returned by [`StableList::into_iter`].
pub struct IntoIter<T, S: GrowthStrategy<T> = DefaultGrowthStrategy<T>, A: Allocator = Global> {
    alloc: A,
    raw: RawIter<'static, T, S>,
    blocks_len: usize,
    blocks_cap: usize,
    _marker: PhantomData<(T, S, A)>,
}

impl<T, S: GrowthStrategy<T>, A: Allocator> IntoIter<T, S, A> {
    pub(super) fn new(list: StableList<T, S, A>) -> Self {
        unsafe {
            let (_, blocks_len, blocks_cap) = list.blocks_raw_parts;
            let raw = RawIter::new(&list);
            let raw = mem::transmute::<RawIter<'_, T, S>, RawIter<'static, T, S>>(raw);

            let alloc = ptr::read(&list.alloc);
            mem::forget(list);

            Self {
                alloc,
                raw,
                blocks_len,
                blocks_cap,
                _marker: PhantomData,
            }
        }
    }
}

impl<T, S: GrowthStrategy<T>, A: Allocator> Drop for IntoIter<T, S, A> {
    fn drop(&mut self) {
        unsafe {
            for ptr in self.raw.by_ref() {
                ptr::drop_in_place(ptr.as_ptr());
            }

            let blocks_ptr = self.raw.chunks.blocks_ptr.as_ptr() as *mut *mut [T];
            StableList::<T, S, A>::free_blocks(&self.alloc, blocks_ptr, self.blocks_len, self.blocks_cap);
        }
    }
}

impl_iter! {
    on = IntoIter { T, S, A } where { S: GrowthStrategy<T>, A: Allocator };
    inner = raw;
    item = { T };
    map = { |_this: &mut Self, ptr: NonNull<T>|
        unsafe {
            Some(ptr::read(ptr.as_ptr()))
        }
    };
    double_ended = true;
    fused = true;
    exact_size = true;
    clone = false;
    force_sync = false;
    force_send = false;
}
