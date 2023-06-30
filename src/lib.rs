#![doc = include_str!("doc.md")]
#![cfg_attr(not(any(test, doc)), no_std)]
#![cfg_attr(feature = "nightly", feature(allocator_api))]

extern crate alloc;

use alloc::alloc::handle_alloc_error;
use core::alloc::{Layout, LayoutError};
use core::cmp::Ordering;
use core::fmt::{Debug, Formatter};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use core::ptr::NonNull;
use core::{fmt, mem, ptr};

use allocator_api2::alloc::{AllocError, Allocator, Global};

pub use iter::{ChunksIter, ChunksIterMut, IntoIter, Iter, IterMut};

use crate::iter::RawIter;
use crate::util::{assume_assert, UnwrapExt};

mod iter;
mod util;

const ZST_BLOCK_TABLE: &[*mut Block] = &[NonNull::dangling().as_ptr()];

#[doc = include_str!("doc.md")]
pub struct StableList<T, A: Allocator = Global> {
    alloc: A,
    len: usize,
    capacity: usize,
    used_blocks: usize,
    block_table: NonNull<[*mut Block]>,
    next_free: NonNull<T>,
    _marker: PhantomData<T>,
}

struct Block;

impl<T, A: Allocator + Default> Default for StableList<T, A> {
    fn default() -> Self {
        Self::new_in(A::default())
    }
}

impl<T> StableList<T, Global> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T, A: Allocator> StableList<T, A> {
    pub fn new_in(alloc: A) -> Self {
        Self {
            alloc,
            len: 0,
            capacity: 0,
            used_blocks: 0,
            next_free: NonNull::dangling(),
            block_table: NonNull::from(&[]),
            _marker: PhantomData,
        }
    }

    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        unsafe { self.get_ptr(index).map(|ptr| &*ptr) }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        unsafe { self.get_ptr(index).map(|ptr| &mut *ptr) }
    }

    fn get_ptr(&self, index: usize) -> Option<*mut T> {
        if index >= self.len {
            return None;
        }

        if Self::is_zst() {
            return Some(NonNull::dangling().as_ptr());
        }

        let bits = usize::BITS - index.leading_zeros();
        let block_index = bits.saturating_sub(Self::first_block_bits()) as usize;
        let mask = (Self::first_block_capacity() * (1usize << block_index >> 1)).wrapping_sub(1);
        let sub_index = index & mask;

        unsafe {
            assume_assert!(block_index < self.block_table.len());
            let block = *self.block_table.as_ref().get_unchecked(block_index);
            Some((block as *mut T).add(sub_index))
        }
    }

    pub fn push(&mut self, value: T) {
        self.try_push_internal(value)
            .unwrap_or_else(|(_, layout)| handle_alloc_error(layout))
    }

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

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        unsafe {
            let ptr = self.next_free.as_ptr().sub(1);
            let result = ptr::read(ptr);
            self.pop_uninit();
            Some(result)
        }
    }

    unsafe fn pop_uninit(&mut self) {
        assume_assert!(self.len > 0);
        self.len -= 1;

        unsafe {
            let ptr = self.next_free.as_ptr().sub(1);
            self.next_free = NonNull::new_unchecked(ptr);

            if Self::is_threshold(self.len) {
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
            let block_capacity = Self::block_capacity(block_index);

            self.capacity /= 2;
            self.next_free = NonNull::new_unchecked((block_ptr as *mut T).add(block_capacity));
        }

        assume_assert!(self.len == self.capacity);
    }

    pub fn insert(&mut self, index: usize, value: T) {
        self.try_insert_internal(index, value)
            .unwrap_or_else(|(_, layout)| handle_alloc_error(layout))
    }

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

            self.pop_uninit();
            result
        }
    }

    #[inline(never)]
    unsafe fn try_add_block(&mut self) -> Result<(), (AllocError, Layout)> {
        assume_assert!(self.len == self.capacity);

        if self.used_blocks < self.block_table.len() {
            let next_block = self.block_table.as_ref()[self.used_blocks];

            self.used_blocks += 1;
            self.capacity = Self::block_capacity(self.used_blocks);
            self.next_free = NonNull::new_unchecked(next_block as *mut T);

            return Ok(());
        }

        if self.len == Self::max_elements() {
            panic!("cannot add more than {} elements", Self::max_elements());
        }

        if Self::is_zst() {
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

        let (layout, block_table_offset) = Self::layout_block(block_index).unwrap_assume();

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

        self.capacity += Self::block_capacity(block_index);
        self.next_free = NonNull::new_unchecked(allocation as *mut T);
        self.used_blocks = block_index + 1;

        Ok(())
    }

    unsafe fn move_block_table(&mut self, new_table: *mut [*mut Block], new_block: *mut Block) {
        let src = self.block_table.as_ref().as_ptr();
        let dest = new_table as *mut *mut Block;
        ptr::copy_nonoverlapping(src, dest, self.block_table.len());
        ptr::write(dest.add(self.block_table.len()), new_block);

        self.block_table = NonNull::new_unchecked(new_table);
    }

    pub fn chunks(&self) -> ChunksIter<T> {
        ChunksIter::new(self)
    }

    pub fn chunks_mut(&mut self) -> ChunksIterMut<T> {
        ChunksIterMut::new(self)
    }

    pub fn iter(&self) -> Iter<T> {
        Iter::new(self)
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(self)
    }

    unsafe fn free_blocks(alloc: &A, block_table: NonNull<[*mut Block]>) {
        let blocks = block_table.as_ref();
        let blocks_len = blocks.len();

        for (block_index, block) in blocks.iter().copied().enumerate() {
            if !Self::is_zst() {
                let (layout, _) = Self::layout_block(block_index).unwrap_assume();
                let ptr = NonNull::new_unchecked(block).cast();
                alloc.deallocate(ptr, layout);
            }

            // After the last block is deallocated, we can no longer reference to the block
            // table. Return immediately to make sure that doesn't happen
            if block_index + 1 == blocks_len {
                return;
            }
        }
    }

    fn layout_block(block_index: usize) -> Result<(Layout, usize), LayoutError> {
        assert!(!Self::is_zst());

        let capacity = Self::block_capacity(block_index);
        let elements_layout = Layout::array::<T>(capacity)?;
        let block_table_layout = Layout::array::<*mut Block>(block_index + 1)?;
        let (layout, block_table_offset) = elements_layout.extend(block_table_layout)?;
        Ok((layout, block_table_offset))
    }

    fn is_threshold(n: usize) -> bool {
        let big_enough = n & ((1 << Self::first_block_bits()) - 1) == 0;
        let is_pow_2 = n & n.wrapping_sub(1) == 0;
        is_pow_2 && big_enough
    }

    fn block_capacity(n: usize) -> usize {
        if Self::is_zst() {
            return usize::MAX;
        }

        Self::first_block_capacity() * 2usize.pow(n.saturating_sub(1) as u32)
    }

    fn first_block_capacity() -> usize {
        if Self::is_zst() {
            return usize::MAX;
        }

        2usize.pow(Self::first_block_bits())
    }

    fn first_block_bits() -> u32 {
        match mem::size_of::<T>() {
            0 => usize::BITS,
            1 => 6,
            2 => 5,
            n if n <= 4 => 4,
            n if n <= 8 => 3,
            n if n <= 16 => 2,
            n if n <= 32 => 1,
            _ => 0,
        }
    }

    fn max_elements() -> usize {
        if Self::is_zst() {
            usize::MAX
        } else {
            let upper_bound = (isize::MAX as usize) / mem::size_of::<T>();
            (upper_bound + 1).next_power_of_two() / 2
        }
    }

    fn is_zst() -> bool {
        mem::size_of::<T>() == 0
    }
}

impl<T, A: Allocator> Drop for StableList<T, A> {
    fn drop(&mut self) {
        unsafe {
            let mut freed_elements = 0;

            for (block_index, block) in self.block_table.as_ref().iter().copied().enumerate() {
                let capacity = Self::block_capacity(block_index);
                let initialized = self.len.saturating_sub(freed_elements).min(capacity);
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(block as *mut T, initialized));

                freed_elements += capacity;
            }

            Self::free_blocks(&self.alloc, self.block_table);
        }
    }
}

impl<T, A: Allocator> Extend<T> for StableList<T, A> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.try_extend_internal(iter)
            .unwrap_or_else(|(_, layout)| handle_alloc_error(layout))
    }
}

impl<T> FromIterator<T> for StableList<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut list = Self::default();
        list.extend(iter);
        list
    }
}

impl<T, A: Allocator> IntoIterator for StableList<T, A> {
    type Item = T;
    type IntoIter = IntoIter<T, A>;

    fn into_iter(self) -> IntoIter<T, A> {
        IntoIter::new(self)
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a StableList<T, A> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T, A: Allocator> IntoIterator for &'a mut StableList<T, A> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T, A: Allocator> Index<usize> for StableList<T, A> {
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

impl<T, A: Allocator> IndexMut<usize> for StableList<T, A> {
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

impl<T: Clone, A: Allocator + Clone> Clone for StableList<T, A> {
    fn clone(&self) -> Self {
        let mut result = Self::new_in(self.alloc.clone());
        result.extend(self.iter().cloned());
        result
    }
}

impl<T: PartialEq, A: Allocator> PartialEq for StableList<T, A> {
    fn eq(&self, other: &Self) -> bool {
        Iterator::eq(self.iter(), other.iter())
    }
}

impl<T: Eq, A: Allocator> Eq for StableList<T, A> {}

impl<T: Hash, A: Allocator> Hash for StableList<T, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len);

        for value in self.iter() {
            Hash::hash(value, state);
        }
    }
}

impl<T: PartialOrd, A: Allocator> PartialOrd for StableList<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Iterator::partial_cmp(self.iter(), other.iter())
    }
}

impl<T: Ord, A: Allocator> Ord for StableList<T, A> {
    fn cmp(&self, other: &Self) -> Ordering {
        Iterator::cmp(self.iter(), other.iter())
    }
}

unsafe impl<T: Send> Send for StableList<T> {}

unsafe impl<T: Sync> Sync for StableList<T> {}

#[cfg(test)]
mod test {
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::fmt::Debug;
    use core::slice;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use crate::{ChunksIter, Iter, StableList};

    struct Model<T> {
        list: StableList<T>,
        vec: Vec<T>,
    }

    impl<T> Default for Model<T> {
        fn default() -> Self {
            Self {
                list: StableList::default(),
                vec: Vec::default(),
            }
        }
    }

    impl<T> Model<T> {
        pub fn push(&mut self, value: T)
        where
            T: Clone,
        {
            self.list.push(value.clone());
            self.vec.push(value);
        }

        pub fn set(&mut self, index: usize, value: T)
        where
            T: Clone,
        {
            self.list[index] = value.clone();
            self.vec[index] = value;
        }

        pub fn extend<I: IntoIterator<Item = T> + Clone>(&mut self, iter: I) {
            self.list.extend(iter.clone());
            self.vec.extend(iter)
        }

        pub fn pop(&mut self)
        where
            T: Eq + Debug,
        {
            assert_eq!(self.list.pop(), self.vec.pop());
        }

        pub fn insert(&mut self, index: usize, value: T)
        where
            T: Clone,
        {
            self.list.insert(index, value.clone());
            self.vec.insert(index, value);
        }

        pub fn remove(&mut self, index: usize)
        where
            T: Eq + Debug,
        {
            assert_eq!(self.list.remove(index), self.vec.remove(index));
        }

        pub fn check_len(&self) {
            assert_eq!(self.list.len(), self.vec.len());
        }

        pub fn check_index_equality(&self, index: usize)
        where
            T: Eq + Debug,
        {
            assert_eq!(self.list.get(index), self.vec.get(index));
        }

        pub fn check_all_indicies_equality(&self)
        where
            T: Eq + Debug,
        {
            for i in 0..=self.list.len() {
                self.check_index_equality(i);
            }
        }

        pub fn check_iter_equality(&self)
        where
            T: Eq + Debug,
        {
            assert!(Iterator::eq(self.list.iter(), self.vec.iter()))
        }

        pub fn all_checks(&self)
        where
            T: Eq + Debug,
        {
            self.check_len();
            self.check_all_indicies_equality();
            self.check_iter_equality();
        }

        pub fn iter(&self) -> ModelIter<T> {
            let result = ModelIter {
                list: self.list.iter(),
                vec: self.vec.iter(),
            };

            result.check_len();

            result
        }
    }

    struct ModelIter<'a, T> {
        list: Iter<'a, T>,
        vec: slice::Iter<'a, T>,
    }

    impl<'a, T> ModelIter<'a, T> {
        pub fn next(&mut self)
        where
            T: Debug + Eq,
        {
            assert_eq!(self.list.next(), self.vec.next());
            self.check_len();
        }

        pub fn next_back(&mut self)
        where
            T: Debug + Eq,
        {
            assert_eq!(self.list.next_back(), self.vec.next_back());
            self.check_len();
        }

        pub fn nth(&mut self, n: usize)
        where
            T: Debug + Eq,
        {
            assert_eq!(self.list.nth(n), self.vec.nth(n));
            self.check_len();
        }

        pub fn nth_back(&mut self, n: usize)
        where
            T: Debug + Eq,
        {
            assert_eq!(self.list.nth_back(n), self.vec.nth_back(n));
            self.check_len();
        }

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