A pointer-stable list, like `Vec<Box<T>>` but much more efficient.

Specifically `StableList` guarantees that the memory location of `list[index]` is the same for the entire lifetime of `list`, regardless of elements being added or removed.
Since `StableList` doesn't relocate elements, this also means that the worst case performance of [`push`ing](StableList::push) is only as slow as allocating a new chunk of memory.
Although `StableList` is fast[^fast], `Vec` is faster and easier to work with (since it dereferences to a slice).
Use `Vec` unless you need pointer stability.

# Examples

`Vec` is not pointer-stable:

```
let mut vec = vec![1];
let first_location = &vec[0] as *const _;

// Vec grow's its internal buffer, moving `vec[0]`
vec.push(2);
let second_location = &vec[0] as *const _;

// Not necessarily the case:
// assert_eq!(first_location, second_location);
```

`StableList` *is* pointer-stable:

```
# use collections::StableList;
#
let mut list = StableList::from_iter([1]);
let first_location = &list[0] as *const _;

// Does not invalidate the pointer to `list[0]`
list.push(2);
let second_location = &list[0] as *const _;

assert_eq!(first_location, second_location);
```

Unlike `Vec<Box<T>>`, each *index* has a stable memory location --- not each element:

```
# use collections::StableList;
#
let mut list = StableList::from_iter([1]);
let first_location = &list[0] as *const _;

list.pop();
list.push(2);
let second_location = &list[0] as *const _;

// Different elements, but same index, so same memory location
assert_ne!(list[0], 1);
assert_eq!(first_location, second_location);
```

```
# use collections::StableList;
#
let mut list = StableList::from_iter([1, 2]);
let first_location = &list[1] as *const _;

list.remove(0);
let second_location = &list[0] as *const _;

// Same element, different indices, so different memory locations
assert_eq!(list[0], 2);
assert_ne!(first_location, second_location);
```

# Implementation

Since memory addresses must remain stable, `StableList` can never deallocate memory used to back an index.
This means keeping a single, linear element buffer is out of the question.
Instead of relocating elements, `StableList` keeps multiple buffers for its elements.
This makes pretty much every operation, other than pushing, more complicated and therefore a bit slower.
In particular, indexing with [`StableList::index`] requires more work than [`Vec::index`], since the function needs to compute which buffer the element lies in, and which index within the buffer the element lies at.
Thankfully, iterating over a `StableList` is not implemented in terms of indexing, so it avoids most of this overhead, but is still slower than iterating over a `Vec`.

In deciding how large each buffer should be, it is important to keep the performance of random accesses in mind.
If buffers where sized randomly, indexing would require a binary search over all the buffer sizes.
Instead, a doubling strategy is used by default, where the first buffer has a statically known size, and each subsequent buffer doubles the total capacity of the list.
Although this is rigid, it allows indexing to be implemented fairly quickly with a bit of branchless bit-hacking.

# Configuration

Two things can be configured via generic parameters on `StableList<T, S, A>`: the growth strategy and allocator (the `S` and `A` respectively).

The growth strategy must currently be one of the built-in supported options - either doubling or flat (linear) growth.
See the docs for [`DoublingGrowthStrategy`] and [`FlatGrowthStrategy`] for more information.
The default growth strategy is designed to work decently with whatever you throw at it.

The allocator parameter controls where the list gets its memory from.
See the [`allocator-api2` crate](https://crates.io/crates/allocator-api2) for more info.

If a `T` is zero-sized, no allocations will ever be made and instead the list is backed by a single zero-sized block with `usize::MAX` capacity.

[^fast]: For some definition of fast