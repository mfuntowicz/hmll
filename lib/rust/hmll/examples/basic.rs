//! Basic example of loading data from a single model file.

use hmll::{Device, LoaderKind, Source, WeightLoader};
use std::env;
use std::str::FromStr;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the file path from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <model_file>", args[0]);
        eprintln!("Example: {} model.safetensors", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];
    let start = usize::from_str(&args[2]).expect("<start> parameter should be a number");
    let end = usize::from_str(&args[3]).expect("<end> parameter should be a number");

    println!("Opening file: {}", file_path);

    // Open the source file
    let source = Source::open(file_path)?;
    println!("✓ File opened successfully");
    println!("  Size: {} bytes ({:.2} MB)", source.size(), source.size() as f64 / 1_048_576.0);

    // Store in an array to ensure proper lifetime
    let sources = [source];

    // Create a weight loader
    println!("\nCreating weight loader...");
    let mut loader = WeightLoader::new(&sources, Device::Cpu, LoaderKind::Auto)?;
    println!("✓ Loader created successfully");
    println!("  Device: {}", loader.device());
    println!("  Number of sources: {}", loader.num_sources());

    // Fetch some data from the beginning of the file
    let fetch_size = end - start;
    let actual_fetch_size = fetch_size.min(sources[0].size());
    println!("\nFetching {} bytes ({:.2} MB)...", actual_fetch_size, actual_fetch_size as f64 / 1_048_576.0);

    let start_time = Instant::now();
    let buffer = loader.fetch(start..end, 0)?;
    let elapsed = start_time.elapsed();

    println!("✓ Data fetched successfully");
    println!("  Buffer size: {} bytes", buffer.len());
    println!("  Buffer device: {}", buffer.device());
    println!("  Fetch time: {:.3}s", elapsed.as_secs_f64());

    // Calculate throughput
    let throughput_bytes_per_sec = buffer.len() as f64 / elapsed.as_secs_f64();
    let throughput_mb_per_sec = throughput_bytes_per_sec / 1_048_576.0;
    let throughput_gb_per_sec = throughput_bytes_per_sec / 1_073_741_824.0;

    println!("\n📊 Throughput:");
    println!("  {:.2} MB/s", throughput_mb_per_sec);
    println!("  {:.2} GB/s", throughput_gb_per_sec);

    // Access the data (for CPU buffers)
    if let Some(data) = buffer.as_slice() {
        println!("\n✓ Buffer accessible as slice");

        // Print first 64 bytes as hex
        let preview_len = 64usize.min(data.len());
        println!("  First {} bytes (hex):", preview_len);
        print!("  ");
        for (i, byte) in data[..preview_len].iter().enumerate() {
            print!("{:02x} ", byte);
            if (i + 1) % 16 == 0 && i < preview_len - 1 {
                print!("\n  ");
            }
        }
        println!();
    }

    println!("\n✓ All operations completed successfully!");

    Ok(())
}
