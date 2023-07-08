use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::ops::Range;
use core::ptr::NonNull;
use core::{mem, ptr};

use allocator_api2::alloc::{Allocator, Global};

use crate::util::{assume_assert, impl_iter};
use crate::{Block, DefaultGrowthStrategy, GrowthStrategy, StableList};

pub(super) struct RawChunksIter<'a, T, S: GrowthStrategy<T> = DefaultGrowthStrategy<T>> {
    strategy: S,
    blocks: &'a [*mut Block],
    indices: Range<usize>,
    last_len: usize,
    _marker: PhantomData<NonNull<T>>,
}

impl<'a, T, S: GrowthStrategy<T>> RawChunksIter<'a, T, S> {
    pub(super) fn new<A: Allocator>(list: &'a StableList<T, S, A>) -> Self {
        unsafe {
            let blocks = list.block_table.as_ref();
            let blocks = &blocks[..list.used_blocks];

            let last_block_len = if blocks.len() > 1 {
                let elements_before = list.strategy.cumulative_capacity(blocks.len() - 1);
                list.len - elements_before
            } else {
                list.len
            };

            Self {
                strategy: list.strategy.clone(),
                blocks,
                indices: 0..blocks.len(),
                last_len: last_block_len,
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
        let block = this.blocks[index];

        let block_len = if index == this.blocks.len() - 1 {
            this.last_len
        } else {
            this.strategy.block_capacity(index)
        };

        assume_assert!(block_len > 0);

        let ptr = ptr::slice_from_raw_parts_mut(block as *mut T, block_len);
        Some(NonNull::new_unchecked(ptr))
    }};
    double_ended = true;
    fused = true;
    exact_size = true;
    clone = { |this: &Self| Self {
        strategy: this.strategy.clone(),
        blocks: this.blocks,
        indices: this.indices.clone(),
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
    _marker: PhantomData<(T, S, A)>,
}

impl<T, S: GrowthStrategy<T>, A: Allocator> IntoIter<T, S, A> {
    pub(super) fn new(list: StableList<T, S, A>) -> Self {
        unsafe {
            let raw = RawIter::new(&list);
            let raw = mem::transmute::<_, RawIter<'static, T, S>>(raw);

            let alloc = ptr::read(&list.alloc);
            mem::forget(list);

            Self {
                alloc,
                raw,
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

            let strategy = &self.raw.chunks.strategy;
            let block_table = NonNull::from(self.raw.chunks.blocks);
            StableList::<T, S, A>::free_blocks(strategy, &self.alloc, block_table);
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
