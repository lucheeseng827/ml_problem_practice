extern crate rand;
extern crate rustacuda;
extern crate rustacuda_core;
extern crate rustacuda_derive;

use rand::Rng;
use rustacuda::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    rustacuda::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let matrix_size = 3;

    let mut rng = rand::thread_rng();

    let mut matrix_a: Vec<f32> = (0..matrix_size * matrix_size)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();

    let mut matrix_b: Vec<f32> = (0..matrix_size * matrix_size)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();

    let mut matrix_dot: Vec<f32> = vec![0.0; matrix_size * matrix_size];
    let mut matrix_cross: Vec<f32> = vec![0.0; matrix_size * matrix_size];

    let module_data = CString::new(include_str!("matrix_ops.ptx"))?;
    let module = Module::load_from_string(&ctx, &module_data)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut device_matrix_a = DeviceBuffer::from_slice(&ctx, &matrix_a)?;
    let mut device_matrix_b = DeviceBuffer::from_slice(&ctx, &matrix_b)?;
    let mut device_matrix_dot = DeviceBuffer::from_slice(&ctx, &matrix_dot)?;
    let mut device_matrix_cross = DeviceBuffer::from_slice(&ctx, &matrix_cross)?;

    let block_size = 3;
    let grid_size = 1;
    let args = [DeviceBuffer::slice_as_ptr(&device_matrix_a) as *const f32,
                DeviceBuffer::slice_as_ptr(&device_matrix_b) as *const f32,
                DeviceBuffer::slice_as_ptr(&device_matrix_dot) as *mut f32,
                DeviceBuffer::slice_as_ptr(&device_matrix_cross) as *mut f32,
                &matrix_size];

    unsafe {
        launch!(module.matrix_ops<<<grid_size, block_size, 0, stream>>>(*args))?;
    }

    stream.synchronize()?;

    device_matrix_dot.copy_to(&mut matrix_dot)?;
    device_matrix_cross.copy_to(&mut matrix_cross)?;

    println!("Matrix A: {:?}", matrix_a);
    println!("Matrix B: {:?}", matrix_b);
    println!("Dot Product: {:?}", matrix_dot);
    println!("Cross Product: {:?}", matrix_cross);

    Ok(())
}
