//! Example of loading data from multiple sharded model files.

use hmll::{Device, LoaderKind, Source, WeightLoader};
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get file paths from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file1> [file2] [file3] ...", args[0]);
        eprintln!(
            "Example: {} model-00001.safetensors model-00002.safetensors",
            args[0]
        );
        std::process::exit(1);
    }

    let file_paths = &args[1..];

    println!("Opening {} file(s)...", file_paths.len());

    // Open all source files
    let mut sources = Vec::new();
    let mut total_size = 0u64;

    for (i, path) in file_paths.iter().enumerate() {
        print!("  [{}] Opening: {}... ", i, path);
        match Source::open(path) {
            Ok(source) => {
                let size = source.size();
                total_size += size as u64;
                println!("✓ ({} bytes)", size);
                sources.push(source);
            }
            Err(e) => {
                eprintln!("✗ Failed: {}", e);
                return Err(e.into());
            }
        }
    }

    println!("\n✓ All files opened successfully");
    println!(
        "  Total size: {} bytes ({:.2} MB)",
        total_size,
        total_size as f64 / 1_048_576.0
    );

    // Create a weight loader for all sources
    println!("\nCreating weight loader for {} sources...", sources.len());
    let mut loader = WeightLoader::new(sources, Device::Cpu, LoaderKind::Auto)?;
    println!("✓ Loader created successfully");

    // Display information about each source
    println!("\nSource information:");
    for i in 0..loader.num_sources() {
        if let Some(info) = loader.source_info(i) {
            println!(
                "  [{}] Size: {} bytes ({:.2} MB)",
                i,
                info.size,
                info.size as f64 / 1_048_576.0
            );
        }
    }

    // Fetch data from each file with throughput measurement
    println!("\nFetching data from each source...");
    let mut total_fetch_time = 0.0;
    let mut total_fetched_bytes = 0;

    for i in 0..loader.num_sources() {
        if let Some(info) = loader.source_info(i) {
            let fetch_size = 512usize.min(info.size);
            print!("  [{}] Fetching {} bytes... ", i, fetch_size);

            let start_time = Instant::now();
            match loader.fetch(0..fetch_size, i) {
                Ok(buffer) => {
                    let elapsed = start_time.elapsed();
                    total_fetch_time += elapsed.as_secs_f64();
                    total_fetched_bytes += buffer.len();

                    let throughput_mb = (buffer.len() as f64 / 1_048_576.0) / elapsed.as_secs_f64();
                    println!(
                        "✓ {} bytes in {:.3}s ({:.2} MB/s)",
                        buffer.len(),
                        elapsed.as_secs_f64(),
                        throughput_mb
                    );

                    // Show a preview of the data
                    if let Some(data) = buffer.as_slice() {
                        let preview_len = 16usize.min(data.len());
                        print!("      Preview: ");
                        for byte in &data[..preview_len] {
                            print!("{:02x} ", byte);
                        }
                        if preview_len < data.len() {
                            print!("...");
                        }
                        println!();
                    }
                }
                Err(e) => {
                    eprintln!("✗ Failed: {}", e);
                    return Err(e.into());
                }
            }
        }
    }

    // Demonstrate fetching from different ranges in different files
    println!("\nDemonstrating random access across files...");

    for i in 0..loader.num_sources().min(3) {
        if let Some(info) = loader.source_info(i) {
            if info.size >= 2048 {
                let ranges = [(0, 256), (512, 768), (1024, 1280)];

                for (start, end) in ranges {
                    if end <= info.size {
                        print!("  [{}] Range {}..{}... ", i, start, end);
                        match loader.fetch(start..end, i) {
                            Ok(buffer) => println!("✓ {} bytes", buffer.len()),
                            Err(e) => println!("✗ {}", e),
                        }
                    }
                }
            }
        }
    }

    println!("\n✓ All operations completed successfully!");
    println!("\nSummary:");
    println!("  Files loaded: {}", loader.num_sources());
    println!(
        "  Total file size: {:.2} MB",
        total_size as f64 / 1_048_576.0
    );
    println!("  Device: {}", loader.device());

    if total_fetched_bytes > 0 && total_fetch_time > 0.0 {
        let avg_throughput_mb = (total_fetched_bytes as f64 / 1_048_576.0) / total_fetch_time;
        println!("\n📊 Overall Throughput:");
        println!(
            "  Total fetched: {:.2} MB",
            total_fetched_bytes as f64 / 1_048_576.0
        );
        println!("  Total time: {:.3}s", total_fetch_time);
        println!("  Average: {:.2} MB/s", avg_throughput_mb);
    }

    Ok(())
}
