use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;
use core::{fmt, mem};

use crate::growth_strategy::private::Sealed;
use crate::util::{assume_assert, is_zst};
use crate::StableList;

pub trait GrowthStrategy<T>: Clone + Sealed {
    #[doc(hidden)]
    fn max_blocks(&self) -> usize;

    #[doc(hidden)]
    unsafe fn block_capacity(&self, index: usize) -> usize;

    #[doc(hidden)]
    unsafe fn cumulative_capacity(&self, blocks: usize) -> usize;

    #[doc(hidden)]
    unsafe fn is_threshold_point(&self, len: usize) -> bool;

    #[doc(hidden)]
    unsafe fn translate_index(&self, index: usize) -> (usize, usize);
}

pub struct DoublingGrowthStrategy<T, const INITIAL_CAPACITY: usize = 0> {
    _marker: PhantomData<fn() -> T>,
}

impl<T, const INITIAL_CAPACITY: usize> Debug for DoublingGrowthStrategy<T, INITIAL_CAPACITY> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("DoublingGrowthStrategy").finish()
    }
}

impl<T> Default for DoublingGrowthStrategy<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const INITIAL_CAPACITY: usize> DoublingGrowthStrategy<T, INITIAL_CAPACITY> {
    pub fn new() -> Self {
        assert!(
            INITIAL_CAPACITY == 0 || INITIAL_CAPACITY.is_power_of_two(),
            "initial capacity must be a power of two"
        );

        Self {
            _marker: PhantomData,
        }
    }

    fn first_block_capacity(&self) -> usize {
        2usize.pow(Self::first_block_bits(self))
    }

    fn first_block_bits(&self) -> u32 {
        if is_zst::<T>() {
            return usize::BITS;
        }

        if INITIAL_CAPACITY == 0 {
            match mem::size_of::<T>() {
                0 => unreachable!(),
                // 64 elements
                n if n <= 1 => 6,
                // 32 elements
                n if n <= 3 => 5,
                // 16 elements
                n if n <= 15 => 4,
                // 8 elements
                n if n <= 63 => 3,
                // 4 elements
                n if n <= 121 => 2,
                // 2 elements
                n if n <= 255 => 1,
                // 1 element
                _ => 0,
            }
        } else {
            INITIAL_CAPACITY.ilog2()
        }
    }
}

impl<T, const INITIAL_CAPACITY: usize> GrowthStrategy<T>
    for DoublingGrowthStrategy<T, INITIAL_CAPACITY>
{
    fn max_blocks(&self) -> usize {
        if is_zst::<T>() {
            return 1;
        }

        let mut count = 0usize;

        loop {
            let capacity = self.first_block_capacity() * 2usize.pow(count.saturating_sub(1) as u32);

            if StableList::<T>::layout_block_with_capacity(capacity, count).is_ok() {
                count += 1;
            } else {
                return count;
            }
        }
    }

    unsafe fn block_capacity(&self, index: usize) -> usize {
        assume_assert!(index < self.max_blocks());

        if is_zst::<T>() {
            return usize::MAX;
        }

        self.first_block_capacity() * 2usize.pow(index.saturating_sub(1) as u32)
    }

    unsafe fn cumulative_capacity(&self, blocks: usize) -> usize {
        assume_assert!(blocks <= self.max_blocks());

        if is_zst::<T>() {
            return if blocks == 0 { 0 } else { usize::MAX };
        }

        self.first_block_capacity() * (1usize << blocks >> 1)
    }

    unsafe fn is_threshold_point(&self, len: usize) -> bool {
        if is_zst::<T>() {
            return len == 0;
        }

        let big_enough = len & ((1 << self.first_block_bits()) - 1) == 0;
        let is_pow_2 = len & len.wrapping_sub(1) == 0;
        is_pow_2 && big_enough
    }

    unsafe fn translate_index(&self, index: usize) -> (usize, usize) {
        let bits = usize::BITS - index.leading_zeros();
        let block_index = bits.saturating_sub(self.first_block_bits()) as usize;
        let mask = (self.first_block_capacity() * (1usize << block_index >> 1)).wrapping_sub(1);
        let sub_index = index & mask;
        (block_index, sub_index)
    }
}

impl<T, const INITIAL_CAPACITY: usize> Copy for DoublingGrowthStrategy<T, INITIAL_CAPACITY> {}

impl<T, const INITIAL_CAPACITY: usize> Clone for DoublingGrowthStrategy<T, INITIAL_CAPACITY> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const INITIAL_CAPACITY: usize> Sealed for DoublingGrowthStrategy<T, INITIAL_CAPACITY> {}

pub struct FlatGrowthStrategy<T, const BLOCK_CAPACITY: usize = 0> {
    _marker: PhantomData<fn() -> T>,
}

impl<T, const BLOCK_CAPACITY: usize> Debug for FlatGrowthStrategy<T, BLOCK_CAPACITY> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("FlatGrowthStrategy").finish()
    }
}

impl<T> Default for FlatGrowthStrategy<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const BLOCK_CAPACITY: usize> FlatGrowthStrategy<T, BLOCK_CAPACITY> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData
        }
    }

    fn block_capacity(&self) -> usize {
        if is_zst::<T>() {
            return usize::MAX;
        }

        if BLOCK_CAPACITY == 0 {
            (256 / mem::size_of::<T>()).next_power_of_two()
        } else {
            BLOCK_CAPACITY
        }
    }
}

impl<T, const BLOCK_CAPACITY: usize> GrowthStrategy<T> for FlatGrowthStrategy<T, BLOCK_CAPACITY> {
    fn max_blocks(&self) -> usize {
        if is_zst::<T>() {
            return 1;
        }

        let mut search_space = 0..(usize::MAX / self.block_capacity());

        while search_space.len() > 1 {
            let mid = search_space.start + (search_space.len() / 2);

            if StableList::<T>::layout_block_with_capacity(self.block_capacity(), mid).is_ok() {
                search_space = mid..search_space.end;
            } else {
                search_space = search_space.start..mid;
            }
        }

        search_space.start
    }

    unsafe fn block_capacity(&self, index: usize) -> usize {
        assume_assert!(index < self.max_blocks());
        self.block_capacity()
    }

    unsafe fn cumulative_capacity(&self, blocks: usize) -> usize {
        assume_assert!(blocks <= self.max_blocks());
        self.block_capacity() * blocks
    }

    unsafe fn is_threshold_point(&self, len: usize) -> bool {
        len % self.block_capacity() == 0
    }

    unsafe fn translate_index(&self, index: usize) -> (usize, usize) {
        assume_assert!(index < self.max_blocks());
        (index / self.block_capacity(), index % self.block_capacity())
    }
}

impl<T, const BLOCK_CAPACITY: usize> Copy for FlatGrowthStrategy<T, BLOCK_CAPACITY> {}

impl<T, const BLOCK_CAPACITY: usize> Clone for FlatGrowthStrategy<T, BLOCK_CAPACITY> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const BLOCK_CAPACITY: usize> Sealed for FlatGrowthStrategy<T, BLOCK_CAPACITY> {}

mod private {
    pub trait Sealed {}
}
