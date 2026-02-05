//! Benchmarks for spectrum synthesis operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use lavoisier_mass_computing::prelude::*;
use lavoisier_mass_computing::extractor::SpectrumExtractor;
use lavoisier_mass_computing::massscript::MassScriptInterpreter;

fn bench_spectrum_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrum_extraction");

    let extractor = SpectrumExtractor::default();

    // Single extraction at various depths
    for depth in [6, 12, 18, 24].iter() {
        let address_str: String = "012102".repeat(*depth / 6 + 1).chars().take(*depth).collect();
        let addr = TernaryAddress::from_str(&address_str).unwrap();

        group.bench_with_input(
            BenchmarkId::new("single", depth),
            &addr,
            |b, addr| {
                b.iter(|| extractor.extract(black_box(addr)))
            },
        );
    }

    group.finish();
}

fn bench_batch_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_extraction");

    let extractor = SpectrumExtractor::default();

    // Generate random-ish addresses
    fn make_addresses(count: usize) -> Vec<TernaryAddress> {
        (0..count)
            .map(|i| {
                let s = format!("{:0>18}", format!("{:o}", i * 1337 % 3u64.pow(18) as usize));
                // Convert octal-ish to ternary (just mod 3 each digit)
                let ternary: String = s.chars()
                    .map(|c| char::from_digit(c.to_digit(10).unwrap_or(0) % 3, 10).unwrap())
                    .collect();
                TernaryAddress::from_str(&ternary[..18.min(ternary.len())]).unwrap()
            })
            .collect()
    }

    for count in [10, 100, 1000, 10000].iter() {
        let addresses = make_addresses(*count);

        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel", count),
            &addresses,
            |b, addrs| {
                b.iter(|| extractor.extract_batch(black_box(addrs)))
            },
        );
    }

    group.finish();
}

fn bench_observable_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("observable_functions");

    let extractor = SpectrumExtractor::default();

    group.bench_function("mass_from_sk", |b| {
        b.iter(|| extractor.mass_from_sk(black_box(0.5)))
    });

    group.bench_function("retention_from_st", |b| {
        b.iter(|| extractor.retention_time_from_st(black_box(0.5)))
    });

    group.bench_function("fragments", |b| {
        b.iter(|| extractor.fragments_from_scoord(black_box(0.5), black_box(0.8)))
    });

    group.bench_function("isotope_pattern", |b| {
        b.iter(|| extractor.isotope_pattern_from_sk(black_box(0.5)))
    });

    group.finish();
}

fn bench_massscript(c: &mut Criterion) {
    let mut group = c.benchmark_group("massscript");

    let simple_script = r#"
partition 012102012102012102
observe
"#;

    let complex_script = r#"
partition PC_34_1 201102012021012102
observe PC_34_1
fragment PC_34_1 at 6
extend by 012012
observe
extend by 102102
observe
detect
"#;

    group.bench_function("parse_simple", |b| {
        b.iter(|| lavoisier_mass_computing::massscript::MassScript::parse(black_box(simple_script)).unwrap())
    });

    group.bench_function("parse_complex", |b| {
        b.iter(|| lavoisier_mass_computing::massscript::MassScript::parse(black_box(complex_script)).unwrap())
    });

    group.bench_function("execute_simple", |b| {
        b.iter(|| {
            let mut interpreter = MassScriptInterpreter::new();
            interpreter.execute_str(black_box(simple_script)).unwrap()
        })
    });

    group.bench_function("execute_complex", |b| {
        b.iter(|| {
            let mut interpreter = MassScriptInterpreter::new();
            interpreter.execute_str(black_box(complex_script)).unwrap()
        })
    });

    group.finish();
}

fn bench_partition_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition_state");

    let coord = SEntropyCoord::new(0.25, 0.5, 0.75).unwrap();
    let addr = TernaryAddress::from_str("012102012102012102").unwrap();

    group.bench_function("from_scoord", |b| {
        b.iter(|| lavoisier_mass_computing::partition::PartitionState::from_scoord(black_box(&coord)))
    });

    group.bench_function("from_address", |b| {
        b.iter(|| lavoisier_mass_computing::partition::PartitionState::from_address(black_box(&addr)))
    });

    let state = lavoisier_mass_computing::partition::PartitionState::from_scoord(&coord);
    group.bench_function("capacity", |b| {
        b.iter(|| black_box(&state).capacity())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_spectrum_extraction,
    bench_batch_extraction,
    bench_observable_functions,
    bench_massscript,
    bench_partition_state
);

criterion_main!(benches);
