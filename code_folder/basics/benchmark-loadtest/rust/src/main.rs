use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn my_function() {
    let mut result = 0;
    for i in 0..100 {
        result += i;
    }
}

fn my_benchmark(c: &mut Criterion) {
    c.bench_function("my_function", |b| b.iter(|| my_function()));
}

criterion_group!(benches, my_benchmark);
criterion_main!(benches);
