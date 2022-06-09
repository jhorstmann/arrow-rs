#![feature(stdsimd)]
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
extern crate arrow;

use std::iter;
use std::sync::Arc;

use arrow::compute::{filter_record_batch, FilterBuilder, FilterPredicate};
use arrow::record_batch::RecordBatch;
use arrow::util::bench_util::*;

use arrow::array::*;
use arrow::compute::filter;
use arrow::datatypes::{DataType, Field, Float32Type, Int32Type, Schema, UInt8Type};

use criterion::{criterion_group, criterion_main, Criterion};
use arrow::buffer::MutableBuffer;

fn bench_filter(data_array: &dyn Array, filter_array: &BooleanArray) {
    criterion::black_box(filter(data_array, filter_array).unwrap());
}

fn bench_built_filter(filter: &FilterPredicate, array: &dyn Array) {
    criterion::black_box(filter.filter(array).unwrap());
}

fn filter_u8_avx512(array: &PrimitiveArray<UInt8Type>, filter: &BooleanArray) -> PrimitiveArray<UInt8Type> {
    use std::arch::x86_64::*;

    let filter_buffer = &filter.data().buffers()[0];

    let chunks = filter_buffer.bit_chunks(filter.offset(), filter.len());
    let count = chunks.iter().chain(iter::once(chunks.remainder_bits())).map(|chunk| chunk.count_ones() as usize).sum::<usize>();

    let data_buffer = &array.data().buffers()[0];
    let mut output_data_buffer = MutableBuffer::with_capacity(count);

    let chunks = filter_buffer.bit_chunks(filter.offset(), filter.len());
    let mut input_offset = 0;
    let mut output_offset = 0;
    chunks.iter().for_each(|filter_chunk| unsafe {
        // if filter_chunk != 0 {
            let data_chunk = _mm512_loadu_epi8(data_buffer.as_ptr().add(input_offset) as *const i8);
            _mm512_mask_compressstoreu_epi8(output_data_buffer.as_mut_ptr().add(output_offset), filter_chunk, data_chunk);
            output_offset += filter_chunk.count_ones() as usize;
        // }
        input_offset += 64;
    });
    if chunks.remainder_len() > 0 {
        let filter_chunk = chunks.remainder_bits();
        unsafe {
            let data_chunk = _mm512_maskz_loadu_epi8(filter_chunk, data_buffer.as_ptr().add(input_offset) as *const i8);
            _mm512_mask_compressstoreu_epi8(output_data_buffer.as_mut_ptr().add(output_offset), filter_chunk, data_chunk);
            // output_offset += filter_chunk.count_ones() as usize;
            // input_offset += 64;
        }
    }

    let data = unsafe {
        ArrayData::new_unchecked(
            DataType::UInt8,
            count,
            None,
            None,
            0,
            vec![output_data_buffer.into()],
            vec![],
        )
    };

    PrimitiveArray::<UInt8Type>::from(data)
}

fn filter_i32_avx512(array: &PrimitiveArray<Int32Type>, filter: &BooleanArray) -> PrimitiveArray<UInt8Type> {
    use std::arch::x86_64::*;

    let filter_buffer = &filter.data().buffers()[0];

    let chunks = filter_buffer.bit_chunks(filter.offset(), filter.len());
    let count = chunks.iter().chain(iter::once(chunks.remainder_bits())).map(|chunk| chunk.count_ones() as usize).sum::<usize>();

    let data_buffer = &array.data().buffers()[0];
    let mut output_data_buffer = MutableBuffer::with_capacity(count*std::mem::size_of::<i32>());

    let chunks = filter_buffer.bit_chunks(filter.offset(), filter.len());
    let mut input_offset = 0;
    let mut output_offset = 0;
    chunks.iter().for_each(|filter_chunk| {
        if filter_chunk != 0 {
            for i in 0..4 {
                let filter_chunk = ((filter_chunk >> (i * 16)) & 0xFFFF) as u16;
                unsafe {
                    let data_chunk = _mm512_loadu_epi32(data_buffer.as_ptr().add(input_offset) as *const i32);
                    _mm512_mask_compressstoreu_epi32(output_data_buffer.as_mut_ptr().add(output_offset), filter_chunk, data_chunk);
                }
                output_offset += filter_chunk.count_ones() as usize;
                input_offset += 16;
            }
        } else {
            input_offset += 64;
        }
    });
    if chunks.remainder_len() > 0 {
        let filter_chunk = chunks.remainder_bits();
        if filter_chunk != 0 {
            for i in 0..4 {
                let filter_chunk = ((filter_chunk >> (i * 16)) & 0xFFFF) as u16;
                unsafe {
                    let data_chunk = _mm512_maskz_loadu_epi32(filter_chunk, data_buffer.as_ptr().add(input_offset) as *const i32);
                    _mm512_mask_compressstoreu_epi32(output_data_buffer.as_mut_ptr().add(output_offset), filter_chunk, data_chunk);
                    output_offset += filter_chunk.count_ones() as usize;
                    input_offset += 16;
                }
            }
        }
    }

    let data = unsafe {
        ArrayData::new_unchecked(
            DataType::UInt8,
            count,
            None,
            None,
            0,
            vec![output_data_buffer.into()],
            vec![],
        )
    };

    PrimitiveArray::<UInt8Type>::from(data)
}

fn add_benchmark(c: &mut Criterion) {
    let size = 65536;
    let filter_array = create_boolean_array(size, 0.0, 0.5);
    let dense_filter_array = create_boolean_array(size, 0.0, 0.9);
    let sparse_filter_array = create_boolean_array(size, 0.0, 0.1);

    let filter = FilterBuilder::new(&filter_array).optimize().build();
    let dense_filter = FilterBuilder::new(&dense_filter_array).optimize().build();
    let sparse_filter = FilterBuilder::new(&sparse_filter_array).optimize().build();

    let data_array = create_primitive_array::<UInt8Type>(size, 0.0);

    c.bench_function("filter optimize (kept 1/2)", |b| {
        b.iter(|| FilterBuilder::new(&filter_array).optimize().build())
    });

    c.bench_function("filter optimize high selectivity (kept 1023/1024)", |b| {
        b.iter(|| FilterBuilder::new(&dense_filter_array).optimize().build())
    });

    c.bench_function("filter optimize low selectivity (kept 1/1024)", |b| {
        b.iter(|| FilterBuilder::new(&sparse_filter_array).optimize().build())
    });

    c.bench_function("filter u8 (kept 1/2)", |b| {
        b.iter(|| bench_filter(&data_array, &filter_array))
    });
    c.bench_function("filter u8 high selectivity (kept 1023/1024)", |b| {
        b.iter(|| bench_filter(&data_array, &dense_filter_array))
    });
    c.bench_function("filter u8 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_filter(&data_array, &sparse_filter_array))
    });

    c.bench_function("filter u8 avx512 (kept 1/2)", |b| {
        b.iter(|| filter_u8_avx512(&data_array, &filter_array))
    });
    c.bench_function("filter u8 avx512 high selectivity (kept 1023/1024)", |b| {
        b.iter(|| filter_u8_avx512(&data_array, &dense_filter_array))
    });
    c.bench_function("filter u8 avx512 low selectivity (kept 1/1024)", |b| {
        b.iter(|| filter_u8_avx512(&data_array, &sparse_filter_array))
    });

    c.bench_function("filter context u8 (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function("filter context u8 high selectivity (kept 1023/1024)", |b| {
        b.iter(|| bench_built_filter(&dense_filter, &data_array))
    });
    c.bench_function("filter context u8 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_built_filter(&sparse_filter, &data_array))
    });

    let data_array = create_primitive_array::<Int32Type>(size, 0.0);
    c.bench_function("filter i32 (kept 1/2)", |b| {
        b.iter(|| bench_filter(&data_array, &filter_array))
    });
    c.bench_function("filter i32 high selectivity (kept 1023/1024)", |b| {
        b.iter(|| bench_filter(&data_array, &dense_filter_array))
    });
    c.bench_function("filter i32 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_filter(&data_array, &sparse_filter_array))
    });

    c.bench_function("filter i32 avx512 (kept 1/2)", |b| {
        b.iter(|| filter_i32_avx512(&data_array, &filter_array))
    });
    c.bench_function("filter i32 avx512 high selectivity (kept 1023/1024)", |b| {
        b.iter(|| filter_i32_avx512(&data_array, &dense_filter_array))
    });
    c.bench_function("filter i32 avx512 low selectivity (kept 1/1024)", |b| {
        b.iter(|| filter_i32_avx512(&data_array, &sparse_filter_array))
    });

    c.bench_function("filter context i32 (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context i32 high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function("filter context i32 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_built_filter(&sparse_filter, &data_array))
    });

    let data_array = create_primitive_array::<Int32Type>(size, 0.5);
    c.bench_function("filter context i32 w NULLs (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context i32 w NULLs high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context i32 w NULLs low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let data_array = create_primitive_array::<UInt8Type>(size, 0.5);
    c.bench_function("filter context u8 w NULLs (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context u8 w NULLs high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context u8 w NULLs low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let data_array = create_primitive_array::<Float32Type>(size, 0.5);
    c.bench_function("filter f32 (kept 1/2)", |b| {
        b.iter(|| bench_filter(&data_array, &filter_array))
    });
    c.bench_function("filter context f32 (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context f32 high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function("filter context f32 low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_built_filter(&sparse_filter, &data_array))
    });

    let data_array = create_string_array::<i32>(size, 0.5);
    c.bench_function("filter context string (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context string high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function("filter context string low selectivity (kept 1/1024)", |b| {
        b.iter(|| bench_built_filter(&sparse_filter, &data_array))
    });

    let data_array = create_string_dict_array::<Int32Type>(size, 0.0);
    c.bench_function("filter context string dictionary (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context string dictionary high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context string dictionary low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let data_array = create_string_dict_array::<Int32Type>(size, 0.5);
    c.bench_function("filter context string dictionary w NULLs (kept 1/2)", |b| {
        b.iter(|| bench_built_filter(&filter, &data_array))
    });
    c.bench_function(
        "filter context string dictionary w NULLs high selectivity (kept 1023/1024)",
        |b| b.iter(|| bench_built_filter(&dense_filter, &data_array)),
    );
    c.bench_function(
        "filter context string dictionary w NULLs low selectivity (kept 1/1024)",
        |b| b.iter(|| bench_built_filter(&sparse_filter, &data_array)),
    );

    let data_array = create_primitive_array::<Float32Type>(size, 0.0);

    let field = Field::new("c1", data_array.data_type().clone(), true);
    let schema = Schema::new(vec![field]);

    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(data_array)]).unwrap();

    c.bench_function("filter single record batch", |b| {
        b.iter(|| filter_record_batch(&batch, &filter_array))
    });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
