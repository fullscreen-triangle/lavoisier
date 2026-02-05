//! Benchmarks for ternary address operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lavoisier_mass_computing::prelude::*;

fn bench_address_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("address_creation");

    for depth in [6, 12, 18, 24, 36].iter() {
        let address_str: String = "012".repeat(*depth / 3 + 1).chars().take(*depth).collect();

        group.bench_with_input(
            BenchmarkId::new("from_string", depth),
            &address_str,
            |b, addr| {
                b.iter(|| TernaryAddress::from_str(black_box(addr)).unwrap())
            },
        );
    }

    group.finish();
}

fn bench_scoord_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("scoord_conversion");

    let addr = TernaryAddress::from_str("012102012102012102").unwrap();
    let coord = SEntropyCoord::new(0.25, 0.5, 0.75).unwrap();

    group.bench_function("to_scoord", |b| {
        b.iter(|| black_box(&addr).to_scoord())
    });

    group.bench_function("from_scoord_18", |b| {
        b.iter(|| TernaryAddress::from_scoord(black_box(&coord), 18).unwrap())
    });

    group.bench_function("from_scoord_36", |b| {
        b.iter(|| TernaryAddress::from_scoord(black_box(&coord), 36).unwrap())
    });

    group.finish();
}

fn bench_address_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("address_operations");

    let addr1 = TernaryAddress::from_str("012102012102").unwrap();
    let addr2 = TernaryAddress::from_str("210012210012").unwrap();

    group.bench_function("extend", |b| {
        b.iter(|| black_box(&addr1).extend(black_box(&addr2)))
    });

    group.bench_function("fragment", |b| {
        b.iter(|| black_box(&addr1).fragment_at(6).unwrap())
    });

    group.bench_function("prefix", |b| {
        b.iter(|| black_box(&addr1).prefix(6).unwrap())
    });

    group.bench_function("is_prefix_of", |b| {
        let prefix = addr1.prefix(6).unwrap();
        b.iter(|| black_box(&prefix).is_prefix_of(black_box(&addr1)))
    });

    group.finish();
}

fn bench_cell_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("cell_bounds");

    for depth in [6, 12, 18, 24].iter() {
        let address_str: String = "012102".repeat(*depth / 6 + 1).chars().take(*depth).collect();
        let addr = TernaryAddress::from_str(&address_str).unwrap();

        group.bench_with_input(
            BenchmarkId::new("cell_bounds", depth),
            &addr,
            |b, addr| {
                b.iter(|| black_box(addr).cell_bounds())
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_address_creation,
    bench_scoord_conversion,
    bench_address_operations,
    bench_cell_bounds
);

criterion_main!(benches);
