use core::{fmt, mem};
use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;

use crate::growth_strategy::private::Sealed;
use crate::StableList;
use crate::util::{assume_assert, is_zst};

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

impl<T, const INITIAL_CAPACITY: usize> Default for DoublingGrowthStrategy<T, INITIAL_CAPACITY> {
    fn default() -> Self {
        assert!(INITIAL_CAPACITY == 0 || INITIAL_CAPACITY.is_power_of_two(), "initial capacity must be a power of two");

        let result = Self {
            _marker: PhantomData,
        };

        assert_ne!(result.max_blocks(), 0, "");

        result
    }
}

impl<T, const INITIAL_CAPACITY: usize> DoublingGrowthStrategy<T, INITIAL_CAPACITY> {
    fn first_block_capacity(&self) -> usize {
        2usize.pow(Self::first_block_bits(self))
    }

    fn first_block_bits(&self) -> u32 {
        if INITIAL_CAPACITY == 0 {
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
        } else {
            INITIAL_CAPACITY.ilog2()
        }
    }
}

impl<T, const INITIAL_CAPACITY: usize> GrowthStrategy<T> for DoublingGrowthStrategy<T, INITIAL_CAPACITY> {
    fn max_blocks(&self) -> usize {
        if is_zst::<T>() {
            return 1;
        }

        let mut count = 0usize;

        loop {
            unsafe {
                let capacity = self.first_block_capacity() * 2usize.pow(count.saturating_sub(1) as u32);

                if StableList::<T>::layout_block_with_capacity(capacity, count).is_ok() {
                    count += 1;
                } else {
                    return count;
                }
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
            return if blocks == 0 {
                0
            } else {
                usize::MAX
            };
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

mod private {
    pub trait Sealed {}
}
