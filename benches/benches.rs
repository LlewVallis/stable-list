use collections::StableList;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use std::iter;

const ITERATE_LEN: usize = 100_000;

const RANDOM_ACCESS_LEN: usize = 2usize.pow(17);
const RANDOM_ACCESS_STRIDE: usize = 1001;

const LINEAR_ACCESS_LEN: usize = 100_000;

const BUILD_LIST_LEN: usize = 100_000;

const POP_MANY_LEN: usize = 100_000;

fn iterate_list_iter(c: &mut Criterion) {
    c.bench_function("iterate_iter", |b| {
        let list = StableList::from_iter(0..ITERATE_LEN);

        b.iter(|| {
            let list = black_box(&list);

            for x in list.iter() {
                black_box(*x);
            }
        });
    });
}

fn iterate_list_index(c: &mut Criterion) {
    c.bench_function("iterate_index", |b| {
        let list = StableList::from_iter(0..ITERATE_LEN);

        b.iter(|| {
            let list = black_box(&list);

            for i in 0..ITERATE_LEN {
                black_box(list[i]);
            }
        });
    });
}

fn iterate_vec(c: &mut Criterion) {
    c.bench_function("iterate_vec", |b| {
        let list = Vec::from_iter(0..ITERATE_LEN);

        b.iter(|| {
            let list = black_box(&list);

            for x in list.iter() {
                black_box(*x);
            }
        });
    });
}

fn random_access_list(c: &mut Criterion) {
    c.bench_function("random_access_list", |b| {
        let list = StableList::from_iter(0..RANDOM_ACCESS_LEN);

        b.iter(|| {
            let mut index = RANDOM_ACCESS_STRIDE;

            while index != 0 {
                let list = black_box(&list);
                black_box(list[black_box(index)]);
                index = (index + RANDOM_ACCESS_STRIDE) % RANDOM_ACCESS_LEN;
            }
        })
    });
}

fn random_access_vec(c: &mut Criterion) {
    c.bench_function("random_access_vec", |b| {
        let list = Vec::from_iter(0..RANDOM_ACCESS_LEN);

        b.iter(|| {
            let mut index = RANDOM_ACCESS_STRIDE;

            while index != 0 {
                let list = black_box(&list);
                black_box(list[black_box(index)]);
                index = (index + RANDOM_ACCESS_STRIDE) % RANDOM_ACCESS_LEN;
            }
        })
    });
}

fn random_access_baseline(c: &mut Criterion) {
    c.bench_function("random_access_baseline", |b| {
        b.iter(|| {
            let mut index = RANDOM_ACCESS_STRIDE;

            while index != 0 {
                black_box(index);
                index = (index + RANDOM_ACCESS_STRIDE) % RANDOM_ACCESS_LEN;
            }
        })
    });
}

fn linear_access_list(c: &mut Criterion) {
    let list = StableList::from_iter(0..RANDOM_ACCESS_LEN);

    c.bench_function("linear_access_list", |b| {
        b.iter(|| {
            for i in 0..LINEAR_ACCESS_LEN {
                black_box(list[black_box(i)]);
            }
        })
    });
}

fn linear_access_vec(c: &mut Criterion) {
    let list = Vec::from_iter(0..RANDOM_ACCESS_LEN);

    c.bench_function("linear_access_vec", |b| {
        b.iter(|| {
            for i in 0..LINEAR_ACCESS_LEN {
                black_box(list[black_box(i)]);
            }
        })
    });
}

fn build_list_push(c: &mut Criterion) {
    c.bench_function("build_list_push", |b| {
        b.iter(|| {
            let mut list = StableList::new();

            let mut iter = 0..BUILD_LIST_LEN;
            while let Some(i) = black_box(iter.next()) {
                list.push(i);
            }

            black_box(list)
        })
    });
}

fn build_vec_push(c: &mut Criterion) {
    c.bench_function("build_vec_push", |b| {
        b.iter(|| {
            let mut list = Vec::new();

            let mut iter = 0..BUILD_LIST_LEN;
            while let Some(i) = black_box(iter.next()) {
                list.push(i);
            }

            black_box(list)
        })
    });
}

fn build_list_extend(c: &mut Criterion) {
    c.bench_function("build_list_extend", |b| {
        b.iter(|| {
            let mut list = StableList::new();
            let iter = (0..BUILD_LIST_LEN).map(black_box);
            list.extend(iter);
            black_box(list)
        })
    });
}

fn build_vec_extend(c: &mut Criterion) {
    c.bench_function("build_vec_extend", |b| {
        b.iter(|| {
            let mut list = Vec::new();
            let iter = (0..BUILD_LIST_LEN).map(black_box);
            list.extend(iter);
            black_box(list)
        })
    });
}

fn build_list_extend_zeros(c: &mut Criterion) {
    c.bench_function("build_list_extend_zeros", |b| {
        b.iter(|| {
            let mut list = StableList::new();
            list.extend(iter::repeat(0usize).take(BUILD_LIST_LEN));
            black_box(list)
        })
    });
}

fn build_vec_extend_zeros(c: &mut Criterion) {
    c.bench_function("build_vec_extend_zeros", |b| {
        b.iter(|| {
            let mut list = Vec::new();
            list.extend(iter::repeat(0usize).take(BUILD_LIST_LEN));
            black_box(list)
        })
    });
}

fn pop_many_list(c: &mut Criterion) {
    c.bench_function("pop_many_list", |b| {
        b.iter_batched(
            || StableList::from_iter(0..POP_MANY_LEN),
            |list| {
                let mut list = black_box(list);
                while black_box(list.pop()).is_some() {}
            },
            BatchSize::LargeInput,
        )
    });
}

fn pop_many_vec(c: &mut Criterion) {
    c.bench_function("pop_many_vec", |b| {
        b.iter_batched(
            || Vec::from_iter(0..POP_MANY_LEN),
            |list| {
                let mut list = black_box(list);
                while black_box(list.pop()).is_some() {}
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(iterate, iterate_list_iter, iterate_list_index, iterate_vec);

criterion_group!(
    random_access,
    random_access_list,
    random_access_vec,
    random_access_baseline
);

criterion_group!(linear_access, linear_access_list, linear_access_vec,);

criterion_group!(
    build,
    build_list_push,
    build_vec_push,
    build_list_extend,
    build_vec_extend,
    build_list_extend_zeros,
    build_vec_extend_zeros,
);

criterion_group!(pop_many, pop_many_list, pop_many_vec,);

criterion_main!(iterate, random_access, linear_access, build, pop_many);
