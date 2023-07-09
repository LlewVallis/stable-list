use core::fmt::{Debug, Formatter};
use core::marker::PhantomData;
use core::{fmt, mem};

use crate::growth_strategy::private::Sealed;
use crate::util::{assume_assert, is_zst};
use crate::StableList;

/// A strategy used by a [`StableList`] to dynamically expand.
///
/// The [default growth strategy](crate::DefaultGrowthStrategy) should work well, but the structs implementing this trait can be used to tune performance.
/// Custom implementations of this trait are not supported, and the methods of the trait should not be called directly.
/// Specifically, the methods of this trait are considered an implementation detail and may be changed at will.
///
/// The chunks returned by [`chunks`](StableList::chunks) and [`chunks_mut`](StableList::chunks_mut) follow the growth strategy.
/// Note that for ZSTs, all growth strategies have exactly one block with `usize::MAX` elements.
///
/// See [`DoublingGrowthStrategy`] and [`FlatGrowthStrategy`].
pub trait GrowthStrategy<T>: Clone + Sealed {
    #[doc(hidden)]
    fn max_blocks(&self) -> usize;

    #[doc(hidden)]
    unsafe fn block_capacity(&self, index: usize) -> usize;

    #[doc(hidden)]
    unsafe fn cumulative_capacity(&self, blocks: usize) -> usize;

    #[doc(hidden)]
    fn max_capacity(&self) -> usize {
        unsafe { self.cumulative_capacity(self.max_blocks()) }
    }

    #[doc(hidden)]
    unsafe fn is_threshold_point(&self, len: usize) -> bool;

    #[doc(hidden)]
    unsafe fn translate_index(&self, index: usize) -> (usize, usize);
}

/// Doubles the capacity of the list whenever it runs out of memory.
///
/// The const-generic parameter may be used to specify how many elements reside in the first block that is allocated.
/// If set to `0` (the default), a reasonable value is determined based on the size of the element.
/// If a non-default initial capacity is specified, it must be a power of two.
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
    const MAX_BLOCKS: usize = Self::compute_max_blocks();
    const FIRST_BLOCK_CAPACITY: usize = Self::compute_first_block_capacity();
    const FIRST_BLOCK_BITS: u32 = Self::compute_first_block_bits();

    /// Construct a new `DoublingGrowthStrategy`.
    ///
    /// # Panics
    ///
    /// If the initial capacity isn't either zero or a power of two.
    pub fn new() -> Self {
        assert!(
            INITIAL_CAPACITY == 0 || INITIAL_CAPACITY.is_power_of_two(),
            "initial capacity must be zero or a power of two"
        );

        Self {
            _marker: PhantomData,
        }
    }

    const fn compute_first_block_capacity() -> usize {
        if is_zst::<T>() {
            usize::MAX
        } else {
            2usize.pow(Self::FIRST_BLOCK_BITS)
        }
    }

    const fn compute_first_block_bits() -> u32 {
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

    const fn compute_max_blocks() -> usize {
        if is_zst::<T>() {
            return 1;
        }

        let mut count = 0usize;
        let mut next_capacity = Self::FIRST_BLOCK_CAPACITY;

        loop {
            if count == usize::BITS as usize {
                return count;
            }

            if StableList::<T>::layout_block(next_capacity).is_err() {
                return count;
            }

            count += 1;

            next_capacity = match next_capacity.checked_mul(2) {
                Some(n) => n,
                None => return count,
            }
        }
    }
}

impl<T, const INITIAL_CAPACITY: usize> GrowthStrategy<T>
    for DoublingGrowthStrategy<T, INITIAL_CAPACITY>
{
    fn max_blocks(&self) -> usize {
        Self::MAX_BLOCKS
    }

    unsafe fn block_capacity(&self, index: usize) -> usize {
        assume_assert!(index < self.max_blocks());

        if is_zst::<T>() {
            return usize::MAX;
        }

        Self::FIRST_BLOCK_CAPACITY * 2usize.pow(index.saturating_sub(1) as u32)
    }

    unsafe fn cumulative_capacity(&self, blocks: usize) -> usize {
        assume_assert!(blocks <= self.max_blocks());
        Self::FIRST_BLOCK_CAPACITY * (1usize << blocks >> 1)
    }

    unsafe fn is_threshold_point(&self, len: usize) -> bool {
        if is_zst::<T>() {
            return len == 0;
        }

        let big_enough = len & ((1 << Self::FIRST_BLOCK_BITS) - 1) == 0;
        let is_pow_2 = len & len.wrapping_sub(1) == 0;
        is_pow_2 && big_enough
    }

    unsafe fn translate_index(&self, index: usize) -> (usize, usize) {
        let bits = usize::BITS - index.leading_zeros();
        let block_index = bits.saturating_sub(Self::FIRST_BLOCK_BITS) as usize;
        let mask = (Self::FIRST_BLOCK_CAPACITY * (1usize << block_index >> 1)).wrapping_sub(1);
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

/// Adds a fixed amount of capacity whenever the list runs out of memory.
///
/// The const-generic parameter may be used to specify how many elements reside in each block that is allocated.
/// If set to `0` (the default), a reasonable value is determined based on the size of the element.
/// Initial capacities that are a power of two are recommended for performance reasons.
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
    const CAPACITY: usize = Self::compute_actual_block_capacity();
    const MAX_BLOCKS: usize = Self::compute_max_blocks();

    /// Constructs a new `FlatGrowthStrategy`.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    const fn compute_actual_block_capacity() -> usize {
        if is_zst::<T>() {
            return usize::MAX;
        }

        if BLOCK_CAPACITY == 0 {
            (256 / mem::size_of::<T>()).next_power_of_two()
        } else {
            BLOCK_CAPACITY
        }
    }

    const fn compute_max_blocks() -> usize {
        if is_zst::<T>() {
            return 1;
        }

        if StableList::<T>::layout_block(Self::CAPACITY).is_ok() {
            usize::MAX / Self::CAPACITY
        } else {
            0
        }
    }
}

impl<T, const BLOCK_CAPACITY: usize> GrowthStrategy<T> for FlatGrowthStrategy<T, BLOCK_CAPACITY> {
    fn max_blocks(&self) -> usize {
        Self::MAX_BLOCKS
    }

    unsafe fn block_capacity(&self, index: usize) -> usize {
        assume_assert!(index < self.max_blocks());
        Self::CAPACITY
    }

    unsafe fn cumulative_capacity(&self, blocks: usize) -> usize {
        assume_assert!(blocks <= self.max_blocks());
        Self::CAPACITY * blocks
    }

    unsafe fn is_threshold_point(&self, len: usize) -> bool {
        len % Self::CAPACITY == 0
    }

    unsafe fn translate_index(&self, index: usize) -> (usize, usize) {
        assume_assert!(index < self.max_blocks());
        (index / Self::CAPACITY, index % Self::CAPACITY)
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
