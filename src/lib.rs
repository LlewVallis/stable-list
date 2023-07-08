#![doc = include_str!("doc.md")]
#![cfg_attr(not(any(test, doc)), no_std)]
#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![warn(missing_debug_implementations, missing_docs)]

extern crate alloc;

use alloc::alloc::handle_alloc_error;
use core::alloc::{Layout, LayoutError};
use core::cmp::Ordering;
use core::fmt::{Debug, Formatter};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use core::ptr::NonNull;
use core::{fmt, ptr};

use allocator_api2::alloc::{AllocError, Allocator, Global};

pub use growth_strategy::*;
pub use iter::{ChunksIter, ChunksIterMut, IntoIter, Iter, IterMut};

use crate::iter::RawIter;
use crate::util::{assume_assert, is_zst, UnwrapExt};

mod growth_strategy;
mod iter;
mod util;

const ZST_BLOCK_TABLE: &[*mut Block] = &[NonNull::dangling().as_ptr()];

/// The default strategy used by a `StableList`.
///
/// See [`DoublingGrowthStrategy`].
pub type DefaultGrowthStrategy<T> = DoublingGrowthStrategy<T>;

#[doc = include_str!("doc.md")]
pub struct StableList<T, S: GrowthStrategy<T> = DefaultGrowthStrategy<T>, A: Allocator = Global> {
    strategy: S,
    alloc: A,
    len: usize,
    capacity: usize,
    used_blocks: usize,
    block_table: NonNull<[*mut Block]>,
    next_free: NonNull<T>,
    _marker: PhantomData<T>,
}

struct Block;

impl<T> Default for StableList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StableList<T> {
    /// Create a new `StableList` with the default growth strategy and the global allocator.
    ///
    /// Use [`new_with`](StableList::new_with) for more control over growth and allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let new_list = StableList::<u32>::new();
    /// ```
    pub fn new() -> Self {
        Self::new_with(DoublingGrowthStrategy::default(), Global)
    }
}

impl<T, S: GrowthStrategy<T>, A: Allocator> StableList<T, S, A> {
    /// Create a new `StableList` with a custom growth strategy and allocator.
    ///
    /// Just use [`new`](StableList::new) if you want the defaults.
    ///
    /// # Examples
    ///
    /// ```
    /// # use allocator_api2::alloc::Global;
    /// # use collections::{FlatGrowthStrategy, StableList};
    /// let new_list = StableList::<u32, _, _>::new_with(FlatGrowthStrategy::default(), Global);
    /// ```
    pub fn new_with(strategy: S, alloc: A) -> Self {
        Self {
            alloc,
            strategy,
            len: 0,
            capacity: 0,
            used_blocks: 0,
            next_free: NonNull::dangling(),
            block_table: NonNull::from(&[]),
            _marker: PhantomData,
        }
    }

    /// Access the allocator used by the list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use allocator_api2::alloc::Global;
    /// # use collections::StableList;
    /// let list = StableList::<u32>::new();
    /// list.allocator(); // Equals `Global`
    /// ```
    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    /// Returns the number of elements in the list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    /// assert_eq!(list.len(), 0);
    ///
    /// list.push(1);
    /// assert_eq!(list.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the list contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    /// assert!(list.is_empty());
    ///
    /// list.push(1);
    /// assert!(!list.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a reference to the element located at the index, or `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    /// assert_eq!(list.get(0), None);
    ///
    /// list.push(1);
    /// assert_eq!(list.get(0), Some(&1));
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        unsafe { self.get_ptr(index).map(|ptr| &*ptr) }
    }

    /// Returns a mutable reference to the element located at the index, or `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    /// assert_eq!(list.get_mut(0), None);
    ///
    /// list.push(1);
    /// assert_eq!(list.get_mut(0), Some(&mut 1));
    /// ```
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        unsafe { self.get_ptr(index).map(|ptr| &mut *ptr) }
    }

    fn get_ptr(&self, index: usize) -> Option<*mut T> {
        if index >= self.len {
            return None;
        }

        if is_zst::<T>() {
            return Some(NonNull::dangling().as_ptr());
        }

        unsafe {
            let (block_index, sub_index) = self.strategy.translate_index(index);
            assume_assert!(block_index < self.block_table.len());
            let block = *self.block_table.as_ref().get_unchecked(block_index);
            Some((block as *mut T).add(sub_index))
        }
    }

    /// Adds an element to the end of the list.
    ///
    /// # Panics
    ///
    /// 1. If the list is at maximum capacity.
    ///    The maximum capacity is influenced by the element size, growth strategy and the platforms pointer width.
    ///    Barring extreme circumstances, the maximum capacity should be big enough.
    /// 2. If the allocator fails to allocate memory required for the new element.
    ///
    /// If you want to recover from the second case, use [`try_push`](StableList::try_push).
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    /// list.push(1);
    /// list.push(2);
    ///
    /// assert_eq!(list[0], 1);
    /// assert_eq!(list[1], 2);
    /// ```
    pub fn push(&mut self, value: T) {
        self.try_push_internal(value)
            .unwrap_or_else(|(_, layout)| handle_alloc_error(layout))
    }

    /// Adds an element to the end of the list, handling memory allocation failure gracefully.
    ///
    /// # Panics
    ///
    /// Although allocation failure won't panic, `try_push` still panics if the list is at maximum capacity.
    /// See [`push`](StableList::push#Panics).
    ///
    /// # Examples
    ///
    /// We generally don't expect allocation to fail:
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    /// assert!(list.try_push(1).is_ok());
    /// ```
    ///
    /// But it might:
    ///
    /// ```
    /// # use std::alloc::Layout;
    /// # use std::ptr::NonNull;
    /// # use allocator_api2::alloc::{Allocator, AllocError};
    /// # use collections::{DefaultGrowthStrategy, StableList};
    /// struct DummyAllocator;
    ///
    /// unsafe impl Allocator for DummyAllocator {
    ///     fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    ///         Err(AllocError)
    ///     }
    ///
    ///     unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {}
    /// }
    ///
    /// let mut list = StableList::new_with(DefaultGrowthStrategy::new(), DummyAllocator);
    /// assert!(list.try_push(1).is_err());
    /// ```
    pub fn try_push(&mut self, value: T) -> Result<(), AllocError> {
        self.try_push_internal(value).map_err(|(err, _)| err)
    }

    fn try_push_internal(&mut self, value: T) -> Result<(), (AllocError, Layout)> {
        unsafe {
            if self.len == self.capacity {
                self.try_add_block()?;
            }

            self.push_fast_path(value);
        }

        Ok(())
    }

    /// Like [`extend`](StableList::extend), but with graceful handling of allocation failure.
    ///
    /// This method is to [`extend`](StableList::extend) as [`try_push`](StableList::try_push) is to [`push`](StableList::push).
    /// If the method fails, the iterator and list are left in valid but unspecified states.
    ///
    /// # Panics
    ///
    /// See [`try_push`](StableList::try_push#Panics).
    ///
    /// # Examples
    ///
    /// We generally don't expect allocation to fail:
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    /// assert!(list.try_extend(0..100).is_ok());
    /// ```
    ///
    /// But it might:
    ///
    /// ```
    /// # use std::alloc::Layout;
    /// # use std::ptr::NonNull;
    /// # use allocator_api2::alloc::{Allocator, AllocError};
    /// # use collections::{DefaultGrowthStrategy, StableList};
    /// struct DummyAllocator;
    ///
    /// unsafe impl Allocator for DummyAllocator {
    ///     fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    ///         Err(AllocError)
    ///     }
    ///
    ///     unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {}
    /// }
    ///
    /// let mut list = StableList::new_with(DefaultGrowthStrategy::new(), DummyAllocator);
    /// assert!(list.try_extend(0..100).is_err());
    /// ```
    pub fn try_extend<I: IntoIterator<Item = T>>(&mut self, iter: I) -> Result<(), AllocError> {
        self.try_extend_internal(iter).map_err(|(err, _)| err)
    }

    fn try_extend_internal<I: IntoIterator<Item = T>>(
        &mut self,
        iter: I,
    ) -> Result<(), (AllocError, Layout)> {
        let mut iter = iter.into_iter();

        let Some(mut next) = iter.next() else { return Ok(()) };

        loop {
            while self.len < self.capacity {
                unsafe {
                    self.push_fast_path(next);
                }

                next = match iter.next() {
                    Some(next) => next,
                    None => return Ok(()),
                };
            }

            unsafe {
                self.try_add_block()?;
            }
        }
    }

    unsafe fn push_fast_path(&mut self, value: T) {
        assume_assert!(self.len < self.capacity);

        self.len += 1;
        ptr::write(self.next_free.as_ptr(), value);
        self.next_free = NonNull::new_unchecked(self.next_free.as_ptr().add(1));
    }

    /// Removes and returns the last element in the list, if it exists.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    ///
    /// list.push(1);
    /// list.push(2);
    ///
    /// assert_eq!(list.pop(), Some(2));
    /// assert_eq!(list.pop(), Some(1));
    /// assert_eq!(list.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        unsafe {
            let ptr = self.next_free.as_ptr().sub(1);
            let result = ptr::read(ptr);
            self.pop_without_read();
            Some(result)
        }
    }

    unsafe fn pop_without_read(&mut self) {
        assume_assert!(self.len > 0);
        self.len -= 1;

        unsafe {
            let ptr = self.next_free.as_ptr().sub(1);
            self.next_free = NonNull::new_unchecked(ptr);

            if self.strategy.is_threshold_point(self.len) {
                self.pop_block();
            }
        }
    }

    unsafe fn pop_block(&mut self) {
        assume_assert!(self.used_blocks > 0);

        self.used_blocks -= 1;

        if self.used_blocks == 0 {
            self.capacity = 0;
        } else {
            let block_index = self.used_blocks - 1;
            let block_ptr = *self.block_table.as_ref().get_unchecked(block_index);
            let block_capacity = self.strategy.block_capacity(block_index);

            self.capacity = self.strategy.cumulative_capacity(self.used_blocks);
            self.next_free = NonNull::new_unchecked((block_ptr as *mut T).add(block_capacity));
        }

        assume_assert!(self.len == self.capacity);
    }

    /// Inserts an element at a specific index, shifting elements after it to the right.
    ///
    /// Note that this method is a lot slower than [`push`](StableList::push), which `StableList` is optimized for.
    /// Also keep in mind that the memory address of any elements to the right of the index will change.
    ///
    /// # Panics
    ///
    /// See [`push`](StableList::push#Panics).
    /// Also, the index must be lesser or equal to the length of the list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    ///
    /// list.push(10);
    /// list.push(30);
    /// list.insert(1, 20);
    ///
    /// assert_eq!(list, StableList::from_iter([10, 20, 30]));
    /// ```
    pub fn insert(&mut self, index: usize, value: T) {
        self.try_insert_internal(index, value)
            .unwrap_or_else(|(_, layout)| handle_alloc_error(layout))
    }

    /// Like [`insert`](StableList::insert), but with graceful handling of allocation failure.
    ///
    /// # Panics
    ///
    /// See [`try_push`](StableList::try_push#Panics).
    /// Also, the index must be lesser or equal to the length of the list.
    ///
    /// # Examples
    ///
    /// We generally don't expect allocation to fail:
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    /// assert!(list.try_insert(0, 1).is_ok());
    /// ```
    ///
    /// But it might:
    ///
    /// ```
    /// # use std::alloc::Layout;
    /// # use std::ptr::NonNull;
    /// # use allocator_api2::alloc::{Allocator, AllocError};
    /// # use collections::{DefaultGrowthStrategy, StableList};
    /// struct DummyAllocator;
    ///
    /// unsafe impl Allocator for DummyAllocator {
    ///     fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    ///         Err(AllocError)
    ///     }
    ///
    ///     unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {}
    /// }
    ///
    /// let mut list = StableList::new_with(DefaultGrowthStrategy::new(), DummyAllocator);
    /// assert!(list.try_insert(0, 1).is_err());
    /// ```
    pub fn try_insert(&mut self, index: usize, value: T) -> Result<(), AllocError> {
        self.try_insert_internal(index, value)
            .map_err(|(err, _)| err)
    }

    fn try_insert_internal(&mut self, index: usize, value: T) -> Result<(), (AllocError, Layout)> {
        if index > self.len {
            panic!("index out of bounds (index={}, len={})", index, self.len);
        }

        unsafe {
            if self.len == self.capacity {
                self.try_add_block()?;
            }

            self.len += 1;
            self.next_free = NonNull::new_unchecked(self.next_free.as_ptr().add(1));

            let mut iter = RawIter::new(self).rev().take(self.len - index).peekable();
            while let Some(dest) = iter.next() {
                match iter.peek() {
                    Some(src) => {
                        ptr::copy_nonoverlapping(src.as_ptr(), dest.as_ptr(), 1);
                    }
                    None => {
                        ptr::write(dest.as_ptr(), value);
                        break;
                    }
                }
            }

            Ok(())
        }
    }

    /// Removes and returns the element at a specific index, shifting elements after it to the left.
    ///
    /// # Panics
    ///
    /// An element must exist at the index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::from_iter([1, 2, 3]);
    ///
    /// assert_eq!(list.remove(0), 1);
    /// assert_eq!(list.remove(1), 3);
    /// assert_eq!(list.remove(0), 2);
    /// ```
    pub fn remove(&mut self, index: usize) -> T {
        if index >= self.len {
            panic!("index out of bounds (index={}, len={})", index, self.len);
        }

        unsafe {
            let mut iter = RawIter::new(self).skip(index).peekable();
            let result = ptr::read(iter.peek().unwrap_assume().as_ptr());

            while let Some(dest) = iter.next() {
                match iter.peek() {
                    Some(src) => {
                        ptr::copy_nonoverlapping(src.as_ptr(), dest.as_ptr(), 1);
                    }
                    None => break,
                }
            }

            self.pop_without_read();
            result
        }
    }

    #[inline(never)]
    unsafe fn try_add_block(&mut self) -> Result<(), (AllocError, Layout)> {
        assume_assert!(self.len == self.capacity);

        if self.used_blocks < self.block_table.len() {
            let next_block = self.block_table.as_ref()[self.used_blocks];

            self.used_blocks += 1;
            self.capacity = self.strategy.block_capacity(self.used_blocks);
            self.next_free = NonNull::new_unchecked(next_block as *mut T);

            return Ok(());
        }

        if self.used_blocks == self.strategy.max_blocks() {
            panic!("cannot add more than {} elements", self.len);
        }

        if is_zst::<T>() {
            self.allocate_zst()
        } else {
            self.allocate_non_zst()
        }
    }

    unsafe fn allocate_zst(&mut self) -> Result<(), (AllocError, Layout)> {
        // Only one block should ever be constructed
        assume_assert!(self.capacity == 0);
        assume_assert!(self.used_blocks == 0);

        self.capacity = usize::MAX;
        self.block_table = NonNull::from(ZST_BLOCK_TABLE);
        self.used_blocks = 1;

        Ok(())
    }

    unsafe fn allocate_non_zst(&mut self) -> Result<(), (AllocError, Layout)> {
        let block_index = self.block_table.len();
        assume_assert!(self.used_blocks == block_index);

        let (layout, block_table_offset) =
            Self::layout_block(&self.strategy, block_index).unwrap_assume();

        let allocation = match self.alloc.allocate(layout) {
            Ok(allocation) => allocation.as_ptr() as *mut u8,
            Err(err) => return Err((err, layout)),
        };

        let block = allocation as *mut Block;

        unsafe {
            let block_table_base = allocation.add(block_table_offset) as *mut *mut Block;
            let block_table = ptr::slice_from_raw_parts_mut(block_table_base, block_index + 1);
            self.move_block_table(block_table, block);
        }

        self.next_free = NonNull::new_unchecked(allocation as *mut T);
        self.used_blocks = block_index + 1;
        self.capacity = self.strategy.cumulative_capacity(self.used_blocks);

        Ok(())
    }

    unsafe fn layout_block(
        strategy: &S,
        block_index: usize,
    ) -> Result<(Layout, usize), LayoutError> {
        Self::layout_block_with_capacity(strategy.block_capacity(block_index), block_index)
    }

    fn layout_block_with_capacity(
        capacity: usize,
        block_index: usize,
    ) -> Result<(Layout, usize), LayoutError> {
        assert!(!is_zst::<T>());

        let elements_layout = Layout::array::<T>(capacity)?;
        let block_table_layout = Layout::array::<*mut Block>(block_index + 1)?;
        let (layout, block_table_offset) = elements_layout.extend(block_table_layout)?;
        Ok((layout, block_table_offset))
    }

    unsafe fn move_block_table(&mut self, new_table: *mut [*mut Block], new_block: *mut Block) {
        let src = self.block_table.as_ref().as_ptr();
        let dest = new_table as *mut *mut Block;
        ptr::copy_nonoverlapping(src, dest, self.block_table.len());
        ptr::write(dest.add(self.block_table.len()), new_block);

        self.block_table = NonNull::new_unchecked(new_table);
    }

    /// Returns an iterator over contiguous ranges of elements from the list.
    ///
    /// Use [`iter`](StableList::iter) if you want an iterator over each element.
    ///
    /// This method guarantees that for any `list`:
    /// ```
    /// # use collections::StableList;
    /// # let list = StableList::from_iter([1, 2, 3]);
    /// # assert!(
    /// Iterator::eq(list.chunks().flatten(), list.iter())
    /// # );
    /// ```
    ///
    /// So, each element appears in exactly one chunk, and order is preserved in and between chunks.
    /// The chunks returned by this method are the same as the blocks produced by the growth strategy.
    ///
    /// # Examples
    ///
    /// Doubling growth:
    ///
    /// ```
    /// # use allocator_api2::alloc::Global;
    /// # use collections::{DoublingGrowthStrategy, StableList};
    /// let mut list = StableList::new_with(DoublingGrowthStrategy::<_, 1>::new(), Global);
    /// list.extend(0..16);
    ///
    /// let expected: [&[i32]; 5] = [
    ///     &[0], &[1], &[2, 3], &[4, 5, 6, 7],
    ///     &[8, 9, 10, 11, 12, 13, 14, 15]
    /// ];
    ///
    /// assert!(Iterator::eq(list.chunks(), expected));
    /// ```
    ///
    /// Flat growth:
    ///
    /// ```
    /// # use allocator_api2::alloc::Global;
    /// # use collections::{FlatGrowthStrategy, StableList};
    /// let mut list = StableList::new_with(FlatGrowthStrategy::<_, 4>::new(), Global);
    /// list.extend(0..16);
    ///
    /// let expected = [&[0, 1, 2, 3], &[4, 5, 6, 7], &[8, 9, 10, 11], &[12, 13, 14, 15]];
    ///
    /// assert!(Iterator::eq(list.chunks(), expected));
    /// ```
    pub fn chunks(&self) -> ChunksIter<T, S> {
        ChunksIter::new(self)
    }

    /// Like [`chunks`](StableList::chunks), but returning mutable slices.
    ///
    /// See [`chunks`](StableList::chunks) for more information.
    pub fn chunks_mut(&mut self) -> ChunksIterMut<T, S> {
        ChunksIterMut::new(self)
    }

    /// Returns an iterator over each item in the list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let list = StableList::from_iter([1, 2, 3]);
    /// let mut iter = list.iter();
    ///
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<T, S> {
        Iter::new(self)
    }

    /// Like [`iter`](StableList::iter), but returning mutable references.
    ///
    /// See [`iter`](StableList::iter) for more information.
    pub fn iter_mut(&mut self) -> IterMut<T, S> {
        IterMut::new(self)
    }

    unsafe fn free_blocks(strategy: &S, alloc: &A, block_table: NonNull<[*mut Block]>) {
        let blocks_base = block_table.as_ref() as *const _ as *mut *mut Block;
        let blocks_len = block_table.as_ref().len();

        for block_index in 0..blocks_len {
            if !is_zst::<T>() {
                let (layout, _) = Self::layout_block(strategy, block_index).unwrap_assume();
                let ptr = NonNull::new_unchecked(*blocks_base.add(block_index) as *mut u8);
                alloc.deallocate(ptr, layout);
            }
        }
    }
}

impl<T, S: GrowthStrategy<T>, A: Allocator> Drop for StableList<T, S, A> {
    fn drop(&mut self) {
        unsafe {
            for chunk in self.chunks_mut() {
                ptr::drop_in_place(chunk as *mut [T]);
            }

            Self::free_blocks(&self.strategy, &self.alloc, self.block_table);
        }
    }
}

impl<T, S: GrowthStrategy<T>, A: Allocator> Extend<T> for StableList<T, S, A> {
    /// Extends the list with zero or more elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut list = StableList::new();
    /// list.extend(0..5);
    /// assert_eq!(list, StableList::from_iter([0, 1, 2, 3, 4]));
    /// ```
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.try_extend_internal(iter)
            .unwrap_or_else(|(_, layout)| handle_alloc_error(layout))
    }
}

impl<T> FromIterator<T> for StableList<T> {
    /// Creates a new list with the elements from an iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// # use collections::StableList;
    /// let mut expected = StableList::new();
    /// let actual = StableList::from_iter([1, 2]);
    ///
    /// expected.push(1);
    /// expected.push(2);
    ///
    /// assert_eq!(expected, actual);
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut list = Self::default();
        list.extend(iter);
        list
    }
}

impl<T, S: GrowthStrategy<T>, A: Allocator> IntoIterator for StableList<T, S, A> {
    type Item = T;
    type IntoIter = IntoIter<T, S, A>;

    /// Converts the list into an iterator.
    ///
    /// The iterator returns each element by value.
    /// When dropped, the returned iterator drops the rest of the elements.
    fn into_iter(self) -> IntoIter<T, S, A> {
        IntoIter::new(self)
    }
}

impl<'a, T, S: GrowthStrategy<T>, A: Allocator> IntoIterator for &'a StableList<T, S, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, S>;

    fn into_iter(self) -> Iter<'a, T, S> {
        self.iter()
    }
}

impl<'a, T, S: GrowthStrategy<T>, A: Allocator> IntoIterator for &'a mut StableList<T, S, A> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, S>;

    fn into_iter(self) -> IterMut<'a, T, S> {
        self.iter_mut()
    }
}

impl<T, S: GrowthStrategy<T>, A: Allocator> Index<usize> for StableList<T, S, A> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        match self.get(index) {
            Some(result) => result,
            None => {
                panic!("index out of bounds (index={}, len={})", index, self.len);
            }
        }
    }
}

impl<T, S: GrowthStrategy<T>, A: Allocator> IndexMut<usize> for StableList<T, S, A> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        let len = self.len;

        match self.get_mut(index) {
            Some(result) => result,
            None => {
                panic!("index out of bounds (index={}, len={})", index, len);
            }
        }
    }
}

impl<T: Debug> Debug for StableList<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<T: Clone, S: GrowthStrategy<T> + Clone, A: Allocator + Clone> Clone for StableList<T, S, A> {
    fn clone(&self) -> Self {
        let mut result = Self::new_with(self.strategy.clone(), self.alloc.clone());
        result.extend(self.iter().cloned());
        result
    }
}

impl<T: PartialEq, S: GrowthStrategy<T>, A: Allocator> PartialEq for StableList<T, S, A> {
    fn eq(&self, other: &Self) -> bool {
        Iterator::eq(self.iter(), other.iter())
    }
}

impl<T: Eq, S: GrowthStrategy<T>, A: Allocator> Eq for StableList<T, S, A> {}

impl<T: Hash, S: GrowthStrategy<T>, A: Allocator> Hash for StableList<T, S, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len);

        for value in self.iter() {
            Hash::hash(value, state);
        }
    }
}

impl<T: PartialOrd, S: GrowthStrategy<T>, A: Allocator> PartialOrd for StableList<T, S, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Iterator::partial_cmp(self.iter(), other.iter())
    }
}

impl<T: Ord, S: GrowthStrategy<T>, A: Allocator> Ord for StableList<T, S, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Iterator::cmp(self.iter(), other.iter())
    }
}

unsafe impl<T: Send, S: GrowthStrategy<T> + Send, A: Allocator + Send> Send
    for StableList<T, S, A>
{
}

unsafe impl<T: Sync, S: GrowthStrategy<T> + Sync, A: Allocator + Sync> Sync
    for StableList<T, S, A>
{
}

#[cfg(test)]
mod test {
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use allocator_api2::alloc::Global;
    use core::fmt::Debug;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use core::{iter, slice};

    use crate::growth_strategy::FlatGrowthStrategy;
    use crate::{
        ChunksIter, DefaultGrowthStrategy, DoublingGrowthStrategy, GrowthStrategy, Iter, StableList,
    };

    struct Model<T, S: GrowthStrategy<T> = DefaultGrowthStrategy<T>> {
        list: StableList<T, S>,
        vec: Vec<T>,
    }

    impl<T> Default for Model<T> {
        fn default() -> Self {
            Self::new_with(DefaultGrowthStrategy::<T>::default())
        }
    }

    impl<T, S: GrowthStrategy<T>> Model<T, S> {
        pub fn new_with(strategy: S) -> Self {
            Self {
                list: StableList::new_with(strategy, Global),
                vec: Vec::default(),
            }
        }

        #[track_caller]
        pub fn push(&mut self, value: T)
        where
            T: Clone,
        {
            self.list.push(value.clone());
            self.vec.push(value);
        }

        #[track_caller]
        pub fn set(&mut self, index: usize, value: T)
        where
            T: Clone,
        {
            self.list[index] = value.clone();
            self.vec[index] = value;
        }

        #[track_caller]
        pub fn extend<I: IntoIterator<Item = T> + Clone>(&mut self, iter: I) {
            self.list.extend(iter.clone());
            self.vec.extend(iter)
        }

        #[track_caller]
        pub fn pop(&mut self)
        where
            T: Eq + Debug,
        {
            assert_eq!(self.list.pop(), self.vec.pop());
        }

        #[track_caller]
        pub fn insert(&mut self, index: usize, value: T)
        where
            T: Clone,
        {
            self.list.insert(index, value.clone());
            self.vec.insert(index, value);
        }

        #[track_caller]
        pub fn remove(&mut self, index: usize)
        where
            T: Eq + Debug,
        {
            assert_eq!(self.list.remove(index), self.vec.remove(index));
        }

        #[track_caller]
        pub fn check_len(&self) {
            assert_eq!(self.list.len(), self.vec.len());
        }

        #[track_caller]
        pub fn check_index_equality(&self, index: usize)
        where
            T: Eq + Debug,
        {
            assert_eq!(self.list.get(index), self.vec.get(index));
        }

        #[track_caller]
        pub fn check_all_indicies_equality(&self)
        where
            T: Eq + Debug,
        {
            for i in 0..=self.list.len() {
                self.check_index_equality(i);
            }
        }

        #[track_caller]
        pub fn check_iter_equality(&self)
        where
            T: Eq + Debug,
        {
            assert!(Iterator::eq(self.list.iter(), self.vec.iter()))
        }

        #[track_caller]
        pub fn all_checks(&self)
        where
            T: Eq + Debug,
        {
            self.check_len();
            self.check_all_indicies_equality();
            self.check_iter_equality();
        }

        #[track_caller]
        pub fn iter(&self) -> ModelIter<T, S> {
            let result = ModelIter {
                list: self.list.iter(),
                vec: self.vec.iter(),
            };

            result.check_len();

            result
        }
    }

    struct ModelIter<'a, T, S: GrowthStrategy<T> = DefaultGrowthStrategy<T>> {
        list: Iter<'a, T, S>,
        vec: slice::Iter<'a, T>,
    }

    impl<'a, T, S: GrowthStrategy<T>> ModelIter<'a, T, S> {
        #[track_caller]
        pub fn next(&mut self)
        where
            T: Debug + Eq,
        {
            assert_eq!(self.list.next(), self.vec.next());
            self.check_len();
        }

        #[track_caller]
        pub fn next_back(&mut self)
        where
            T: Debug + Eq,
        {
            assert_eq!(self.list.next_back(), self.vec.next_back());
            self.check_len();
        }

        #[track_caller]
        pub fn nth(&mut self, n: usize)
        where
            T: Debug + Eq,
        {
            assert_eq!(self.list.nth(n), self.vec.nth(n));
            self.check_len();
        }

        #[track_caller]
        pub fn nth_back(&mut self, n: usize)
        where
            T: Debug + Eq,
        {
            assert_eq!(self.list.nth_back(n), self.vec.nth_back(n));
            self.check_len();
        }

        #[track_caller]
        fn check_len(&self) {
            assert_eq!(self.list.len(), self.vec.len());
            assert_eq!(self.list.size_hint(), self.vec.size_hint());
        }
    }

    const N: &[usize] = &[0, 1, 2, 5, 10, 100, 1_000];

    #[test]
    fn extend() {
        for n in N {
            let mut model = Model::default();
            model.extend(0..*n);
            model.all_checks();
        }
    }

    #[test]
    fn push_many() {
        for n in N {
            let mut model = Model::default();

            for i in 0..*n {
                model.push(i);
            }

            model.all_checks();
        }
    }

    #[test]
    fn mutate() {
        for n in N {
            let mut model = Model::default();
            model.extend((0..*n).rev());

            for i in 0..*n {
                model.set(i, i);
            }

            model.all_checks();
        }
    }

    #[test]
    fn extend_zsts() {
        for n in N {
            let mut model = Model::default();
            model.extend(vec![(); *n]);
            model.all_checks();
        }
    }

    #[test]
    fn extend_overaligned() {
        #[repr(align(128))]
        #[derive(Debug, Copy, Clone, Eq, PartialEq)]
        struct Overaligned(u32);

        for n in N {
            let mut model = Model::default();
            model.extend(vec![Overaligned(0); *n]);
            model.all_checks();
        }
    }

    #[test]
    fn extend_large_type() {
        for n in N {
            let mut model = Model::default();
            model.extend(vec![[0; 256]; *n]);
            model.all_checks();
        }
    }

    #[test]
    fn drops() {
        for n in N {
            let strong = Arc::new(());
            let weak = Arc::downgrade(&strong);

            let list = StableList::from_iter(vec![strong; *n]);
            drop(list);

            assert_eq!(weak.strong_count(), 0);
        }
    }

    #[test]
    fn drops_zst() {
        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct Zst;

        impl Drop for Zst {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        for n in N {
            DROP_COUNT.store(0, Ordering::SeqCst);

            let mut list = StableList::new();
            (0..*n).for_each(|_| list.push(Zst));
            drop(list);

            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), *n);
        }
    }

    #[test]
    fn is_pointer_stable() {
        for n in N {
            let mut list = StableList::new();
            let mut pointers = Vec::new();

            for i in 0..*n {
                list.push(i);
                pointers.push(&list[i] as *const _);
            }

            for (i, elem) in list.iter().enumerate() {
                assert_eq!(elem as *const _, pointers[i]);
            }
        }
    }

    #[test]
    fn pop_and_readd() {
        for start_count in N {
            for pop_count in N {
                for readd_count in N {
                    let mut model = Model::default();

                    model.extend(0..*start_count);
                    (0..*pop_count).for_each(|_| model.pop());
                    model.extend(0..*readd_count);

                    model.all_checks();
                }
            }
        }
    }

    #[test]
    fn pop_and_readd_front() {
        const SMALL_N: &[usize] = &[0, 1, 2, 5, 10, 100];

        for start_count in SMALL_N {
            for pop_count in SMALL_N {
                for readd_count in SMALL_N {
                    if pop_count > start_count {
                        continue;
                    }

                    let mut model = Model::default();

                    (0..*start_count).for_each(|i| model.insert(0, i));
                    (0..*pop_count).for_each(|_| model.remove(0));
                    (0..*readd_count).for_each(|i| model.insert(0, i));

                    model.all_checks();
                }
            }
        }
    }

    #[test]
    fn insert_middle() {
        for n in N {
            if *n == 0 {
                continue;
            }

            let mut model = Model::default();
            model.extend(0..*n);
            model.insert(*n / 2, 0);
            model.all_checks();
        }
    }

    #[test]
    fn remove_middle() {
        for n in N {
            if *n == 0 {
                continue;
            }

            let mut model = Model::default();
            model.extend(0..*n);
            model.remove(*n / 2);
            model.all_checks();
        }
    }

    #[test]
    fn prepend_many() {
        for n in N {
            let mut model = Model::default();
            (0..*n).for_each(|i| model.insert(0, i));
            model.all_checks();
        }
    }

    #[test]
    fn prepend_many_zst() {
        for n in N {
            let mut model = Model::default();
            (0..*n).for_each(|_| model.insert(0, ()));
            model.all_checks();
        }
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_index_panics() {
        let list = StableList::from_iter([1, 2]);
        let _ = list[2];
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_remove_panics() {
        let mut list = StableList::from_iter([1, 2]);
        list.remove(2);
    }

    #[test]
    fn iter_forward() {
        for n in N {
            let mut model = Model::default();
            model.extend(0..*n);

            let mut iter = model.iter();
            (0..=*n).for_each(|_| iter.next());
        }
    }

    #[test]
    fn iter_backward() {
        for n in N {
            let mut model = Model::default();
            model.extend(0..*n);

            let mut iter = model.iter();
            (0..=*n).for_each(|_| iter.next_back());
        }
    }

    #[test]
    fn iter_alternating() {
        for n in N {
            let mut model = Model::default();
            model.extend(0..*n);

            let mut iter = model.iter();

            for _ in 0..*n {
                iter.next();
                iter.next_back();
            }
        }
    }

    #[test]
    fn iter_nth() {
        for n in N {
            let mut model = Model::default();
            model.extend(0..*n);

            for i in 0..*n {
                let mut iter = model.iter();
                iter.nth(i);
            }
        }
    }

    #[test]
    fn iter_nth_back() {
        for n in N {
            let mut model = Model::default();
            model.extend(0..*n);

            for i in 0..*n {
                let mut iter = model.iter();
                iter.nth_back(i);
            }
        }
    }

    #[test]
    fn iter_forward_zst() {
        for n in N {
            let mut model = Model::default();
            model.extend(vec![(); *n]);

            let mut iter = model.iter();
            (0..=*n).for_each(|_| iter.next());
        }
    }

    #[test]
    fn iter_backward_zst() {
        for n in N {
            let mut model = Model::default();
            model.extend(vec![(); *n]);

            let mut iter = model.iter();
            (0..=*n).for_each(|_| iter.next_back());
        }
    }

    #[test]
    fn no_empty_chunks() {
        for n in N {
            let list = StableList::from_iter(0..*n);

            for chunk in list.chunks() {
                assert!(!chunk.is_empty());
            }
        }
    }

    #[test]
    fn chunk_sizes_add_to_len() {
        for n in N {
            let list = StableList::from_iter(0..*n);
            let sum = list.chunks().map(|c| c.len()).sum::<usize>();
            assert_eq!(sum, list.len());
        }
    }

    #[test]
    fn no_empty_chunks_zst() {
        for n in N {
            let list = StableList::from_iter(vec![(); *n]);

            for chunk in list.chunks() {
                assert!(!chunk.is_empty());
            }
        }
    }

    #[test]
    fn chunk_sizes_add_to_len_zst() {
        for n in N {
            let list = StableList::from_iter(vec![(); *n]);
            let sum = list.chunks().map(|c| c.len()).sum::<usize>();
            assert_eq!(sum, list.len());
        }
    }

    #[test]
    fn into_iter_forwards() {
        for n in N {
            let list = StableList::from_iter(0..*n);
            assert!(Iterator::eq(list.into_iter(), 0..*n));
        }
    }

    #[test]
    fn into_iter_backwards() {
        for n in N {
            let list = StableList::from_iter(0..*n);
            assert!(Iterator::eq(list.into_iter().rev(), (0..*n).rev()));
        }
    }

    #[test]
    fn into_iter_drops() {
        for n in N {
            let strong = Arc::new(());
            let weak = Arc::downgrade(&strong);

            let list = StableList::from_iter(vec![strong; *n]);
            list.into_iter();

            assert_eq!(weak.strong_count(), 0);
        }
    }

    #[test]
    fn into_iter_doesnt_drop_iterated() {
        for a in N {
            for b in N {
                let strong = Arc::new(());
                let weak = Arc::downgrade(&strong);

                let list = StableList::from_iter(vec![strong; *a + *b]);
                let mut iter = list.into_iter();

                let _buffer = iter.by_ref().take(*a).collect::<Vec<_>>();
                drop(iter);

                assert_eq!(weak.strong_count(), *a);
            }
        }
    }

    #[test]
    fn custom_initial_capacity() {
        macro_rules! case {
            ($cap:literal, $values:expr) => {
                let mut model = Model::new_with(DoublingGrowthStrategy::<_, $cap>::new());
                model.extend($values);
                model.all_checks();
            };
        }

        case!(1, 0..100);
        case!(2, 0..100);
        case!(4, 0..100);
        case!(8, 0..100);
        case!(16, 0..100);
        case!(32, 0..100);

        case!(8, iter::repeat(()).take(100));
    }

    #[test]
    fn flat_growth_strategy() {
        macro_rules! case {
            ($cap:literal, $values:expr) => {
                let mut model = Model::new_with(FlatGrowthStrategy::<_, $cap>::new());
                model.extend($values);
                model.all_checks();
            };
        }

        case!(0, 0..100);
        case!(1, 0..100);
        case!(2, 0..100);
        case!(3, 0..100);
        case!(4, 0..100);
        case!(5, 0..100);

        case!(100, 0..100);
        case!(128, 0..100);

        case!(25, iter::repeat(()).take(100));
    }

    #[allow(clippy::extra_unused_lifetimes)]
    fn _variance<'a>(list: StableList<&'static u32>) {
        let _: StableList<&'a u32> = list;
    }

    fn _variance_iter<'a>(list: Iter<'a, &'static u32>) {
        let _: Iter<'a, &'a u32> = list;
    }

    fn _variance_chunks<'a>(list: ChunksIter<'a, &'static u32>) {
        let _: ChunksIter<'a, &'a u32> = list;
    }
}
