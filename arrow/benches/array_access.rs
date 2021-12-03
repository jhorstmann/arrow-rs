// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#[macro_use]
extern crate criterion;

use criterion::{Criterion, Throughput};
use arrow::array::{Array, Int64Array};
use arrow::datatypes::Int64Type;
use arrow::util::bench_util::create_primitive_array;

extern crate arrow;

#[inline]
fn is_set(bitmap: Option<&[u8]>, i: usize) -> bool {
    if let Some(bitmap) = bitmap {
        (bitmap[i / 8] >> (i % 8)) & 0b1 == 1
    } else {
        false
    }
}

#[inline]
unsafe fn is_set_unsafe(bitmap: Option<&[u8]>, i: usize) -> bool {
    if let Some(bitmap) = bitmap {
        (bitmap.get_unchecked(i / 8) >> (i % 8)) & 0b1 == 1
    } else {
        false
    }
}

#[inline(never)]
fn sliding_window_sum_arrow(input: &Int64Array, output: &mut [i64]) {
    // assert!(output.len() >= input.len()-1);

    output[..input.len() - 1].iter_mut().enumerate().for_each(|(i, x)| {
        let mut sum = 0;
        if input.is_valid(i) {
            sum += input.value(i);
        }
        if input.is_valid(i + 1) {
            sum += input.value(i + 1);
        }
        *x = sum;
    });
}

#[inline(never)]
fn sliding_window_sum_native(input: &[i64], valid: Option<&[u8]>, output: &mut [i64]) {
    let valid = valid.map(|b| &b[0..(input.len()+7)/8]);

    output[..input.len() - 1].iter_mut().enumerate().for_each(|(i, x)| {
        let mut sum = 0;
        if is_set(valid, i) {
            sum += input[i];
        }
        if is_set(valid, i+1) {
            sum += input[i+1];
        }
        *x = sum;
    });
}

#[inline(never)]
fn sliding_window_sum_native_unsafe(input: &[i64], valid: Option<&[u8]>, output: &mut [i64]) {
    let valid = valid.map(|b| &b[0..(input.len()+7)/8]);

    output[..input.len() - 1].iter_mut().enumerate().for_each(|(i, x)| unsafe {
        let mut sum = 0;
        if is_set_unsafe(valid, i) {
            sum += input.get_unchecked(i);
        }
        if is_set_unsafe(valid, i+1) {
            sum += input.get_unchecked(i+1);
        }
        *x = sum;
    });
}


fn array_access_benchmark(c: &mut Criterion) {
    let array = create_primitive_array::<Int64Type>(4096, 0.125);
    let mut output = vec![0_i64; array.len() - 1];

    c.benchmark_group("array_access")
        .throughput(Throughput::Bytes((array.len() * std::mem::size_of::<i64>() + array.len()/8) as u64))
        .bench_function("arrow", |b| {
            b.iter(|| sliding_window_sum_arrow(&array, &mut output))
        })
        .bench_function("native", |b| {
            let valid = unsafe { array.data().null_buffer().unwrap().typed_data() };
            b.iter(|| sliding_window_sum_native(&array.values(), Some(valid), &mut output))
        })
        .bench_function("native_unsafe", |b| {
            let valid = unsafe { array.data().null_buffer().unwrap().typed_data() };
            b.iter(|| sliding_window_sum_native_unsafe(&array.values(), Some(valid), &mut output))
        })
    ;
}


criterion_group!(benches, array_access_benchmark);
criterion_main!(benches);
