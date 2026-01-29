"""
Nsight Compute Profiler Integration
Runs ncu on CUDA kernels and parses the profiling results
"""

import subprocess
import json
import os
import tempfile
import sys
from pathlib import Path
from typing import Dict, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gpu_filters
    import numpy as np
except ImportError:
    gpu_filters = None
    np = None


def check_ncu_available() -> bool:
    """Check if ncu (Nsight Compute) is available in PATH"""
    try:
        result = subprocess.run(
            ['ncu', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def profile_kernel_with_ncu(
    img_array: np.ndarray,
    filter_type: str,
    level: int,
    sigma: Optional[float] = None,
    radius: int = 3
) -> Dict[str, Any]:
    """
    Profile a kernel execution using Nsight Compute
    
    Args:
        img_array: Input image as numpy array
        filter_type: 'gaussian', 'box', or 'sobel'
        level: Optimization level (1 or 2 for all filters)
        sigma: Sigma for Gaussian blur (if applicable)
        radius: Kernel radius (not used for sobel)
    
    Returns:
        Dictionary with profiling metrics from ncu
    """
    if not check_ncu_available():
        raise RuntimeError("ncu (Nsight Compute) not found in PATH. Please install Nsight Compute.")
    
    if gpu_filters is None:
        raise RuntimeError("gpu_filters module not available")
    
    # Create temporary directory for profiling output
    # Use a persistent temp dir so we can inspect files if needed
    tmpdir = tempfile.mkdtemp(prefix="ncu_profile_")
    try:
        output_file = os.path.join(tmpdir, "profile.json")
        
        # Determine kernel names - we profile both horizontal and vertical passes
        if filter_type == "gaussian":
            if level == 1:
                # Level 1 kernels: gaussianBlurHorizontalNaive, gaussianBlurVerticalNaive
                kernel_pattern = "gaussianBlur.*Naive"  # Match both horizontal and vertical
            else:  # level == 2
                # Level 2 kernels: gaussianBlurHorizontalLevel2, gaussianBlurVerticalLevel2
                # Use pattern that matches both horizontal and vertical Level2 kernels
                kernel_pattern = "gaussianBlur.*Level2"  # Match both horizontal and vertical Level 2 kernels
        elif filter_type == "box":
            if level == 1:
                kernel_pattern = "boxBlur.*Naive"  # Match both horizontal and vertical naive kernels
            else:  # level == 2
                kernel_pattern = "boxBlur.*Shared"  # Match both horizontal and vertical shared memory kernels
        else:  # sobel
            # Sobel only has one kernel (not separable like blur filters)
            if level == 1:
                kernel_pattern = "sobelEdgeDetectionNaive"
            else:  # level == 2
                kernel_pattern = "sobelEdgeDetectionShared"
        
        # Create a Python script that will be profiled
        profiler_script = os.path.join(tmpdir, "profile_kernel.py")
        with open(profiler_script, 'w') as f:
            f.write(f"""
import sys
sys.path.insert(0, '{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')
import gpu_filters
import numpy as np

# Load test image
img_array = np.load('{os.path.join(tmpdir, "input.npy")}')

# Run the filter
if '{filter_type}' == 'gaussian':
    result = gpu_filters.gaussian_blur(
        img_array,
        sigma={sigma if sigma else 2.0},
        radius={radius},
        level={level}
    )
elif '{filter_type}' == 'box':
    result = gpu_filters.box_blur(
        img_array,
        radius={radius},
        level={level}
    )
else:  # sobel
    result = gpu_filters.sobel_edge_detection(
        img_array,
        level={level}
    )
""")
        
        # Save input image
        np.save(os.path.join(tmpdir, "input.npy"), img_array)
        
        # Run ncu profiling
        # ncu needs to profile the Python process that runs the kernel
        # Profile all kernels (both horizontal and vertical passes)
        # Note: ncu exports to .ncu-rep format, then we convert to JSON
        ncu_report_file = os.path.join(tmpdir, "profile.ncu-rep")
        ncu_cmd = [
            'ncu',
            '--set', 'full',  # Collect all metrics
            '--kernel-name', f'regex:.*{kernel_pattern}.*',  # Match kernel pattern with regex
            '--launch-skip', '0',
            '--launch-count', '10',  # Profile multiple launches to catch both kernels
            '--export', ncu_report_file,
            '--force-overwrite',
            'python3', profiler_script
        ]
        
        try:
            # Set environment to ensure CUDA is available
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
            
            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 120 second timeout (profiling is slower)
                cwd=tmpdir,
                env=env
            )
            
            # Debug: Print stderr if there's an error
            if result.returncode != 0 and result.stderr:
                print(f"ncu stderr: {result.stderr[:500]}")
            if result.stdout:
                print(f"ncu stdout: {result.stdout[:500]}")
            
            # Check for ncu report file
            report_file = ncu_report_file
            if not os.path.exists(report_file):
                # Search for .ncu-rep files in tmpdir (ncu might use different name)
                for f in os.listdir(tmpdir):
                    if f.endswith('.ncu-rep'):
                        report_file = os.path.join(tmpdir, f)
                        break
            
            if report_file and os.path.exists(report_file):
                print(f"Found ncu report: {report_file}")
                # Convert .ncu-rep to CSV format (more reliable than JSON)
                csv_file = output_file.replace('.json', '.csv')
                csv_cmd = [
                    'ncu',
                    '--import', report_file,
                    '--csv',  # Export as CSV
                    '--force-overwrite'
                ]
                
                csv_result = subprocess.run(
                    csv_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir
                )
                
                # Check for CSV output in stdout
                csv_output = csv_result.stdout
                if csv_output and len(csv_output) > 100:
                    # Save CSV for debugging
                    csv_file_path = os.path.join(tmpdir, "ncu_output.csv")
                    with open(csv_file_path, 'w') as f:
                        f.write(csv_output)
                    # Parse CSV output to extract metrics
                    csv_metrics = parse_ncu_csv(csv_output)
                    if csv_metrics and any(csv_metrics.values()):
                        # Pass the parsed metrics (which includes kernel_durations dict)
                        return csv_metrics
                
                # If CSV didn't work, try the old JSON export method
                json_cmd2 = [
                    'ncu',
                    '--import', report_file,
                    '--export', output_file,
                    '--force-overwrite'
                ]
                
                json_result2 = subprocess.run(
                    json_cmd2,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir
                )
                
                if json_result2.returncode != 0:
                    print(f"Warning: Failed to convert ncu report: {json_result2.stderr[:300] if json_result2.stderr else json_result2.stdout[:300]}")
                
                # Extract specific metrics using ncu query
                # Query key metrics we care about
                key_metrics = [
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
                    "dram__throughput.avg.bytes_per_second",
                    "smsp__warps_active.avg.pct_of_peak_sustained_active",
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
                ]
                
                # Try to get metrics using ncu --query-metrics
                query_results = {}
                for metric in key_metrics:
                    query_cmd = ['ncu', '--import', report_file, '--query-metrics', metric]
                    query_result = subprocess.run(query_cmd, capture_output=True, text=True, timeout=10)
                    if query_result.returncode == 0 and query_result.stdout:
                        # Parse the metric value from output
                        for line in query_result.stdout.split('\n'):
                            if metric in line or ':' in line:
                                parts = line.split(':')
                                if len(parts) >= 2:
                                    query_results[metric] = parts[-1].strip()
                
                # Also try --print with specific sections
                print_cmd = ['ncu', '--import', report_file, '--page', 'summary', '--print', 'sum']
                print_result = subprocess.run(print_cmd, capture_output=True, text=True, timeout=30)
                
                if print_result.returncode == 0 and print_result.stdout:
                    # Save raw output for debugging
                    debug_file = os.path.join(tmpdir, "ncu_print_output.txt")
                    with open(debug_file, 'w') as f:
                        f.write(print_result.stdout)
                    print(f"Saved ncu output to: {debug_file}")
                    
                    text_metrics = parse_ncu_text_output(print_result.stdout)
                    # Add query results
                    if query_results:
                        text_metrics["queried_metrics"] = query_results
                    
                    # Return metrics if we found any, otherwise return raw output
                    if any(text_metrics.values()) or query_results:
                        return text_metrics
                    else:
                        # Return raw output for manual inspection
                        return {
                            "raw_output": print_result.stdout[:10000],  # First 10000 chars
                            "profiling_success": True,
                            "report_file": report_file,
                            "debug_file": debug_file
                        }
                    
                    # Return basic info that we have a report
                    return {
                        "report_file": report_file,
                        "conversion_error": json_result2.stderr[:200] if json_result2.stderr else "Unknown",
                        "profiling_success": True,
                        "raw_stdout": json_result2.stdout[:1000] if json_result2.stdout else None
                    }
                
                # Parse JSON output
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            profile_data = json.load(f)
                        return parse_ncu_output(profile_data)
                    except json.JSONDecodeError as e:
                        # If JSON parsing fails, check if file has content
                        with open(output_file, 'r') as f:
                            content = f.read()
                        if len(content) > 100:  # File has content
                            print(f"Warning: ncu JSON output is not valid, but has content ({len(content)} bytes)")
                            # Try to extract some info from text output
                            return {"raw_output": content[:1000], "parse_error": str(e), "profiling_success": True}
                        else:
                            raise RuntimeError(f"ncu JSON file is empty or invalid: {e}")
                else:
                    raise RuntimeError("ncu did not produce JSON export file")
            else:
                # No output file - check stderr for clues
                error_msg = result.stderr if result.stderr else "Unknown error"
                if "Permission denied" in error_msg or "cannot attach" in error_msg.lower():
                    raise RuntimeError("ncu cannot attach to process. Try running with sudo or check permissions.")
                elif "not found" in error_msg.lower():
                    raise RuntimeError("ncu cannot find the target process or kernel")
                else:
                    raise RuntimeError(f"ncu did not produce output file. Exit code: {result.returncode}, stderr: {error_msg[:200]}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("ncu profiling timed out after 120 seconds")
        except FileNotFoundError:
            raise RuntimeError("ncu command not found. Is Nsight Compute installed?")
        except Exception as e:
            # Don't delete tmpdir on error so user can inspect
            print(f"Profiling error - temp files kept in: {tmpdir}")
            raise RuntimeError(f"Profiling failed: {str(e)}")
    finally:
        # Keep tmpdir for inspection - comment out to auto-delete
        # import shutil
        # shutil.rmtree(tmpdir, ignore_errors=True)
        pass


def parse_ncu_output(profile_data: Dict) -> Dict[str, Any]:
    """
    Parse ncu JSON output and extract relevant metrics
    
    Args:
        profile_data: JSON data from ncu
    
    Returns:
        Dictionary with extracted metrics
    """
    metrics = {
        "occupancy": {},
        "memory": {},
        "warp": {},
        "execution": {},
        "throughput": {}
    }
    
    try:
        # Navigate through ncu JSON structure
        # The structure varies by ncu version, so we need to be flexible
        if "Ranges" in profile_data:
            for range_data in profile_data["Ranges"]:
                if "Kernels" in range_data:
                    for kernel in range_data["Kernels"]:
                        # Extract metrics from kernel data
                        if "Metrics" in kernel:
                            for metric_name, metric_value in kernel["Metrics"].items():
                                # Categorize metrics
                                if "occupancy" in metric_name.lower():
                                    metrics["occupancy"][metric_name] = metric_value
                                elif "memory" in metric_name.lower() or "dram" in metric_name.lower():
                                    metrics["memory"][metric_name] = metric_value
                                elif "warp" in metric_name.lower():
                                    metrics["warp"][metric_name] = metric_value
                                elif "throughput" in metric_name.lower():
                                    metrics["throughput"][metric_name] = metric_value
                                else:
                                    metrics["execution"][metric_name] = metric_value
                        
                        # Extract kernel configuration
                        if "LaunchConfiguration" in kernel:
                            config = kernel["LaunchConfiguration"]
                            metrics["config"] = {
                                "grid_x": config.get("gridX", 0),
                                "grid_y": config.get("gridY", 0),
                                "grid_z": config.get("gridZ", 0),
                                "block_x": config.get("blockX", 0),
                                "block_y": config.get("blockY", 0),
                                "block_z": config.get("blockZ", 0),
                                "shared_memory": config.get("sharedMemory", 0),
                                "registers_per_thread": config.get("registersPerThread", 0)
                            }
        
        # Also check for direct metric access (different ncu versions)
        if "MetricValues" in profile_data:
            for metric in profile_data["MetricValues"]:
                name = metric.get("MetricName", "")
                value = metric.get("Value", 0)
                
                if "occupancy" in name.lower():
                    metrics["occupancy"][name] = value
                elif "memory" in name.lower() or "dram" in name.lower():
                    metrics["memory"][name] = value
                elif "warp" in name.lower():
                    metrics["warp"][name] = value
                elif "throughput" in name.lower():
                    metrics["throughput"][name] = value
                else:
                    metrics["execution"][name] = value
        
    except Exception as e:
        print(f"Warning: Error parsing ncu output: {e}")
        # Return raw data if parsing fails
        metrics["raw"] = profile_data
    
    return metrics


def parse_ncu_csv(csv_output: str) -> Dict[str, Any]:
    """Parse ncu CSV output to extract metrics"""
    import csv as csv_module
    from io import StringIO
    
    metrics = {
        "occupancy": {},
        "memory": {},
        "warp": {},
        "execution": {},
        "throughput": {},
        "config": {},
        "kernel_durations": {}  # Track duration per kernel
    }
    
    # Parse CSV using csv module to handle quoted fields properly
    reader = csv_module.DictReader(StringIO(csv_output))
    
    # Track unique kernels and their durations
    kernels_seen = set()
    all_kernels_in_csv = set()  # Track all kernels mentioned in CSV
    
    for row in reader:
        # Track all kernel names we see in the CSV (regardless of metric)
        # Try different possible column names for kernel name
        kernel_name = (row.get("Kernel Name", "") or 
                      row.get("KernelName", "") or 
                      row.get("kernel_name", "") or 
                      row.get("kernelName", "") or
                      row.get("Name", "")).strip()
        if kernel_name and kernel_name.lower() not in ["kernel name", "name", ""]:
            all_kernels_in_csv.add(kernel_name)
        metric_name = row.get("Metric Name", "").strip()
        metric_value = row.get("Metric Value", "").strip()
        section_name = row.get("Section Name", "").strip()
        
        if not metric_name or not metric_value:
            continue
        
        # Extract numeric value (remove commas, handle percentages)
        try:
            # Remove commas and percentage signs
            value_str = metric_value.replace(',', '').replace('%', '').strip()
            if value_str:
                # Try to parse as float
                try:
                    numeric_value = float(value_str)
                    # Store both as number and original string
                    metric_value = numeric_value
                except ValueError:
                    # Keep as string if not numeric
                    pass
        except:
            pass
        
        # Store kernel config
        if "Block Size" in metric_name:
            metrics["config"]["block_size"] = metric_value
        elif "Grid Size" in metric_name:
            metrics["config"]["grid_size"] = metric_value
        
        # Categorize metrics by section and name
        metric_lower = metric_name.lower()
        section_lower = section_name.lower()
        
        # Occupancy metrics
        if "occupancy" in metric_lower or "active warps" in metric_lower or "eligible warps" in metric_lower:
            metrics["occupancy"][metric_name] = metric_value
        
        # Memory metrics
        elif "memory" in metric_lower or "dram" in metric_lower or "l1" in metric_lower or "l2" in metric_lower or "cache" in metric_lower:
            metrics["memory"][metric_name] = metric_value
        elif "memory" in section_lower:
            metrics["memory"][metric_name] = metric_value
        
        # Warp metrics
        elif "warp" in metric_lower:
            metrics["warp"][metric_name] = metric_value
        elif "warp" in section_lower:
            metrics["warp"][metric_name] = metric_value
        
        # Throughput metrics
        elif "throughput" in metric_lower or "bandwidth" in metric_lower:
            metrics["throughput"][metric_name] = metric_value
        
        # Execution/Compute metrics
        elif "sm" in metric_lower or "compute" in metric_lower or "instruction" in metric_lower or "cycle" in metric_lower:
            metrics["execution"][metric_name] = metric_value
        elif "compute" in section_lower or "instruction" in section_lower:
            metrics["execution"][metric_name] = metric_value
        
        # Duration/time metrics - extract per kernel
        # Duration is in "GPU Speed Of Light Throughput" section
        elif metric_name == "Duration" and section_name == "GPU Speed Of Light Throughput":
            # Duration is reported per kernel, track it
            # Try different possible column names for kernel name
            kernel_name = (row.get("Kernel Name", "") or 
                          row.get("KernelName", "") or 
                          row.get("kernel_name", "") or 
                          row.get("kernelName", "") or
                          row.get("Name", "")).strip()
            # Filter out header-like values
            if kernel_name and kernel_name.lower() not in ["kernel name", "name", ""]:
                try:
                    # Extract numeric value and handle different units
                    if isinstance(metric_value, (int, float)):
                        raw_value = float(metric_value)
                        # If value is very large (> 1e6), likely in nanoseconds
                        # If value is very small (< 0.001), might be in seconds
                        if raw_value > 1e6:
                            duration_ms = raw_value / 1e6  # Convert ns to ms
                        elif raw_value < 0.001:
                            duration_ms = raw_value * 1000  # Convert s to ms
                        else:
                            duration_ms = raw_value  # Assume ms
                    else:
                        # Parse string value with units
                        value_str = str(metric_value).replace('"', '').replace(',', '').strip()
                        value_lower = value_str.lower()
                        
                        # Extract number
                        import re
                        number_match = re.search(r'[\d.]+', value_str)
                        if number_match:
                            raw_value = float(number_match.group())
                            
                            # Determine unit
                            if 'ns' in value_lower or 'nanosecond' in value_lower:
                                duration_ms = raw_value / 1e6  # ns to ms
                            elif 'us' in value_lower or 'μs' in value_lower or 'microsecond' in value_lower:
                                duration_ms = raw_value / 1e3  # us to ms
                            elif 's' in value_lower and 'ms' not in value_lower and 'us' not in value_lower and 'ns' not in value_lower:
                                duration_ms = raw_value * 1000  # s to ms
                            else:
                                # Assume milliseconds
                                duration_ms = raw_value
                        else:
                            continue  # Could not parse number
                    
                    # Store duration (remove the 100ms filter - actual kernel times can be longer)
                    # Only filter out obviously wrong values (> 1 hour)
                    if duration_ms < 3600000.0:  # Less than 1 hour
                        if kernel_name not in metrics["kernel_durations"]:
                            metrics["kernel_durations"][kernel_name] = duration_ms
                            kernels_seen.add(kernel_name)
                            # Extracted duration for kernel
                except (ValueError, TypeError) as e:
                    # Debug: print error for troubleshooting
                    # print(f"Warning: Could not parse duration for {kernel_name}: {metric_value}, error: {e}")
                    pass
            
            metrics["execution"][metric_name] = metric_value
        # Also track Elapsed Cycles to calculate time from cycles if Duration is unreliable
        elif metric_name == "Elapsed Cycles" and section_name == "GPU Speed Of Light Throughput":
            # Try different possible column names for kernel name
            kernel_name = (row.get("Kernel Name", "") or 
                          row.get("KernelName", "") or 
                          row.get("kernel_name", "") or 
                          row.get("kernelName", "") or
                          row.get("Name", "")).strip()
            # Filter out header-like values
            if kernel_name and kernel_name.lower() not in ["kernel name", "name", ""]:
                try:
                    if isinstance(metric_value, (int, float)):
                        cycles = float(metric_value)
                    else:
                        cycles = float(str(metric_value).replace('"', '').replace(',', '').strip())
                    
                    # Store cycles for this kernel (we'll calculate time later if needed)
                    if "kernel_cycles" not in metrics:
                        metrics["kernel_cycles"] = {}
                    if kernel_name not in metrics["kernel_cycles"]:
                        metrics["kernel_cycles"][kernel_name] = cycles
                        # Extracted cycles for kernel
                except (ValueError, TypeError):
                    # Could not parse cycles
                    pass
            
            metrics["execution"][metric_name] = metric_value
        elif "duration" in metric_lower or "time" in metric_lower or "elapsed" in metric_lower:
            # Other time-related metrics
            metrics["execution"][metric_name] = metric_value
        elif "time" in metric_lower or "elapsed" in metric_lower:
            metrics["execution"][metric_name] = metric_value
        else:
            # Default to execution
            metrics["execution"][metric_name] = metric_value
    
    # Calculate total execution time from all kernels
    # If we have cycles but missing durations, calculate from cycles
    # Try to get GPU max clock frequency (more reliable than current clock)
    # Default to ~1.5 GHz if not available
    gpu_freq_ghz = 1.5  # Default assumption
    try:
        import subprocess
        # Try max clock first (more stable)
        freq_result = subprocess.run(['nvidia-smi', '--query-gpu=clocks.max.gr', '--format=csv,noheader'], 
                                   capture_output=True, text=True, timeout=2)
        if freq_result.returncode == 0 and freq_result.stdout:
            freq_str = freq_result.stdout.strip().replace(' MHz', '').replace(' MHz', '').strip()
            if freq_str:
                freq_mhz = float(freq_str)
                if freq_mhz > 100:  # Sanity check (should be > 100 MHz)
                    gpu_freq_ghz = freq_mhz / 1000.0
                    # Detected GPU max frequency
    except Exception:
        # Could not detect GPU frequency, using default
        pass
    
    if "kernel_cycles" in metrics and metrics["kernel_cycles"]:
        # Calculate duration from cycles for kernels missing duration
        for kernel_name, cycles in metrics["kernel_cycles"].items():
            if kernel_name not in metrics.get("kernel_durations", {}):
                # Time = cycles / frequency
                # Note: cycles are per SM, but we need total cycles across all SMs
                # For now, use cycles directly (this might need adjustment based on SM count)
                duration_ms = (cycles / (gpu_freq_ghz * 1e9)) * 1000.0
                if "kernel_durations" not in metrics:
                    metrics["kernel_durations"] = {}
                metrics["kernel_durations"][kernel_name] = duration_ms
                # Calculated duration from cycles
    
    # If we have some kernel durations but not all, estimate missing ones
    # For blur filters, horizontal and vertical passes are typically similar
    if metrics["kernel_durations"]:
        horizontal_kernels = [k for k in metrics["kernel_durations"].keys() if "Horizontal" in k]
        # Find vertical kernels that were profiled but don't have valid durations
        all_kernels = metrics.get("_all_kernels", set())
        missing_vertical = [k for k in all_kernels if "Vertical" in k and k not in metrics["kernel_durations"]]
        
        # If we have horizontal duration but missing vertical, estimate it
        if horizontal_kernels and missing_vertical:
            horizontal_kernel = horizontal_kernels[0]
            horizontal_duration = metrics["kernel_durations"][horizontal_kernel]
            
            # Try to use cycles ratio if available
            if "kernel_cycles" in metrics:
                horizontal_cycles = metrics["kernel_cycles"].get(horizontal_kernel, 0)
                for vertical_kernel in missing_vertical:
                    vertical_cycles = metrics["kernel_cycles"].get(vertical_kernel, 0)
                    if horizontal_cycles > 0 and vertical_cycles > 0:
                        # Estimate vertical duration based on cycles ratio
                        estimated_duration = horizontal_duration * (vertical_cycles / horizontal_cycles)
                        metrics["kernel_durations"][vertical_kernel] = estimated_duration
                        # Estimated duration from cycles ratio
                        continue
            
            # Fallback: assume similar duration (they process same amount of data)
            for vertical_kernel in missing_vertical:
                if vertical_kernel not in metrics["kernel_durations"]:
                    metrics["kernel_durations"][vertical_kernel] = horizontal_duration
                    # Estimated duration (assumed similar to horizontal)
    
    # Store all kernels we found in CSV (even if we don't have durations for all)
    if all_kernels_in_csv:
        metrics["_all_kernels"] = all_kernels_in_csv
        # If we have kernels but no durations, try to calculate from cycles
        if not metrics.get("kernel_durations") and "kernel_cycles" in metrics:
            # We already tried to calculate from cycles above, but let's make sure
            # all kernels with cycles get durations
            if "kernel_durations" not in metrics:
                metrics["kernel_durations"] = {}
            for kernel_name in all_kernels_in_csv:
                if kernel_name in metrics.get("kernel_cycles", {}) and kernel_name not in metrics["kernel_durations"]:
                    cycles = metrics["kernel_cycles"][kernel_name]
                    # Calculate duration from cycles (using frequency detected above)
                    duration_ms = (cycles / (gpu_freq_ghz * 1e9)) * 1000.0
                    metrics["kernel_durations"][kernel_name] = duration_ms
    
    # If we still have kernels but no durations, at least report which kernels were profiled
    if all_kernels_in_csv and not metrics.get("kernel_durations"):
        # Try to extract from any metric that has kernel name
        # Look for "Kernel Duration" metrics that might have kernel names
        for key, value in metrics.get("execution", {}).items():
            if "kernel" in key.lower() and "duration" in key.lower():
                # This might contain kernel-specific duration info
                pass
    
    # Clean up internal tracking but keep _all_kernels for now (used below)
    # if "_all_kernels" in metrics:
    #     del metrics["_all_kernels"]
    
    # If we have kernel durations, calculate totals
    if metrics.get("kernel_durations"):
        total_duration = sum(metrics["kernel_durations"].values())
        metrics["total_kernel_duration_ms"] = total_duration
        metrics["kernels_profiled"] = list(metrics["kernel_durations"].keys())
        metrics["total_kernels"] = len(metrics["kernel_durations"])
    elif all_kernels_in_csv:
        # At least report which kernels were found, even without durations
        metrics["kernels_profiled"] = list(all_kernels_in_csv)
        metrics["total_kernels"] = len(all_kernels_in_csv)
        # Try to estimate total time from cycles if available
        if "kernel_cycles" in metrics and metrics["kernel_cycles"]:
            total_cycles = sum(metrics["kernel_cycles"].values())
            estimated_total_ms = (total_cycles / (gpu_freq_ghz * 1e9)) * 1000.0
            metrics["total_kernel_duration_ms"] = estimated_total_ms
            metrics["time_ms"] = estimated_total_ms
        # If we have cycles per kernel, calculate per-kernel durations
        if "kernel_cycles" in metrics and metrics["kernel_cycles"]:
            if "kernel_durations" not in metrics:
                metrics["kernel_durations"] = {}
            for kernel_name, cycles in metrics["kernel_cycles"].items():
                if kernel_name not in metrics["kernel_durations"]:
                    duration_ms = (cycles / (gpu_freq_ghz * 1e9)) * 1000.0
                    metrics["kernel_durations"][kernel_name] = duration_ms
            # Recalculate total now that we have per-kernel durations
            if metrics["kernel_durations"]:
                total_duration = sum(metrics["kernel_durations"].values())
                metrics["total_kernel_duration_ms"] = total_duration
                metrics["kernels_profiled"] = list(metrics["kernel_durations"].keys())
                metrics["total_kernels"] = len(metrics["kernel_durations"])
    
    # Clean up internal tracking
    if "_all_kernels" in metrics:
        del metrics["_all_kernels"]
    
    return metrics


def parse_ncu_text_output(text_output: str) -> Dict[str, Any]:
    """Parse ncu text output (from --print sum) to extract metrics"""
    metrics = {
        "occupancy": {},
        "memory": {},
        "warp": {},
        "execution": {},
        "throughput": {}
    }
    
    if not text_output or len(text_output) < 50:
        return metrics
    
    # Look for common metric patterns in text output
    lines = text_output.split('\n')
    current_section = None
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()
        
        # Detect sections
        if "occupancy" in line_lower:
            current_section = "occupancy"
        elif "memory" in line_lower or "dram" in line_lower:
            current_section = "memory"
        elif "warp" in line_lower:
            current_section = "warp"
        elif "throughput" in line_lower or "bandwidth" in line_lower:
            current_section = "throughput"
        
        # Parse metric lines (typically "Metric Name: value" or "Metric Name = value")
        if ':' in line_stripped or '=' in line_stripped:
            separator = ':' if ':' in line_stripped else '='
            parts = line_stripped.split(separator, 1)
            if len(parts) == 2:
                metric_name = parts[0].strip()
                metric_value_str = parts[1].strip()
                
                # Try to extract numeric value
                try:
                    # Remove units and extract number
                    value_str = metric_value_str.split()[0] if metric_value_str.split() else metric_value_str
                    value_str = value_str.replace('%', '').replace(',', '')
                    value = float(value_str)
                    
                    # Categorize based on metric name
                    metric_lower = metric_name.lower()
                    if "occupancy" in metric_lower:
                        metrics["occupancy"][metric_name] = value
                    elif "warp" in metric_lower:
                        metrics["warp"][metric_name] = value
                    elif "memory" in metric_lower or "dram" in metric_lower:
                        metrics["memory"][metric_name] = value
                    elif "throughput" in metric_lower or "bandwidth" in metric_lower:
                        metrics["throughput"][metric_name] = value
                    else:
                        metrics["execution"][metric_name] = value
                except (ValueError, IndexError):
                    # Store as string if not numeric
                    if current_section and current_section in metrics:
                        metrics[current_section][metric_name] = metric_value_str
    
    return metrics


def get_common_ncu_metrics(metrics: Dict[str, Any], ncu_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Extract commonly used metrics in a standardized format
    
    Returns a dictionary with key metrics that can be displayed in UI
    """
    if not metrics or not isinstance(metrics, dict):
        return {}
    common = {}
    
    # Occupancy metrics
    occupancy_data = metrics.get("occupancy", {})
    if occupancy_data:
        # Look for active warps, eligible warps, occupancy
        for key, value in occupancy_data.items():
            key_lower = key.lower()
            try:
                if isinstance(value, (int, float)):
                    val = float(value)
                else:
                    val = float(str(value).replace('%', '').replace(',', ''))
                
                if "active warps" in key_lower:
                    common["active_warps_per_scheduler"] = val
                elif "eligible warps" in key_lower:
                    common["eligible_warps_per_scheduler"] = val
                elif "occupancy" in key_lower:
                    common["occupancy_pct"] = val
            except (ValueError, TypeError):
                pass
    
    # Memory metrics
    memory_data = metrics.get("memory", {})
    if memory_data:
        for key, value in memory_data.items():
            key_lower = key.lower()
            try:
                if isinstance(value, (int, float)):
                    val = float(value)
                else:
                    val = float(str(value).replace('%', '').replace(',', '').replace('Gbyte/s', '').replace('Ghz', ''))
                
                if "memory throughput" in key_lower and "gbyte" in str(value).lower():
                    common["memory_throughput_gbps"] = val
                elif "dram throughput" in key_lower:
                    common["dram_throughput_pct"] = val
                elif "l1" in key_lower and "hit rate" in key_lower:
                    common["l1_hit_rate_pct"] = val
                elif "l2" in key_lower and "hit rate" in key_lower:
                    common["l2_hit_rate_pct"] = val
                elif "mem busy" in key_lower or "mem pipes busy" in key_lower:
                    common["memory_busy_pct"] = val
            except (ValueError, TypeError):
                pass
    
    # Warp efficiency
    warp_data = metrics.get("warp", {})
    if warp_data:
        for key, value in warp_data.items():
            key_lower = key.lower()
            try:
                if isinstance(value, (int, float)):
                    val = float(value)
                else:
                    val = float(str(value).replace('%', '').replace(',', ''))
                
                if "active threads" in key_lower:
                    common["avg_active_threads_per_warp"] = val
                elif "efficiency" in key_lower:
                    common["warp_efficiency_pct"] = val
            except (ValueError, TypeError):
                pass
    
    # Execution metrics
    exec_data = metrics.get("execution", {})
    if exec_data:
        for key, value in exec_data.items():
            key_lower = key.lower()
            try:
                if isinstance(value, (int, float)):
                    val = float(value)
                else:
                    val = float(str(value).replace('%', '').replace(',', '').replace('ms', '').replace('us', ''))
                
                # Extract actual kernel execution time (without profiling overhead)
                # Note: ncu reports duration per kernel, we need to sum both passes
                if "duration" in key_lower:
                    # ncu reports duration in ms for the actual kernel execution
                    if "ms" in str(value).lower() or isinstance(value, (int, float)):
                        # Store individual kernel duration
                        if "kernel_durations" not in common:
                            common["kernel_durations"] = []
                        common["kernel_durations"].append(val)
                    elif "us" in str(value).lower() or "μs" in str(value):
                        # Convert microseconds to milliseconds
                        val_ms = val / 1000.0
                        if "kernel_durations" not in common:
                            common["kernel_durations"] = []
                        common["kernel_durations"].append(val_ms)
                elif "elapsed cycles" in key_lower:
                    # Can calculate time from cycles if we have frequency
                    # This is the actual kernel execution cycles
                    common["elapsed_cycles"] = val
                elif "sm busy" in key_lower:
                    common["sm_busy_pct"] = val
                elif "compute throughput" in key_lower or "sm throughput" in key_lower:
                    common["compute_throughput_pct"] = val
            except (ValueError, TypeError):
                pass
    
    # Extract total kernel duration from ncu_data if available
    # Check in metrics dict first (from parse_ncu_csv), then in ncu_data
    if ncu_data and "total_kernel_duration_ms" in ncu_data:
        common["time_ms"] = ncu_data["total_kernel_duration_ms"]
        common["kernel_duration_ms"] = ncu_data["total_kernel_duration_ms"]
        if "kernels_profiled" in ncu_data:
            common["kernels_profiled"] = ncu_data["kernels_profiled"]
            common["total_kernels"] = len(ncu_data["kernels_profiled"])
    elif "total_kernel_duration_ms" in metrics:
        common["time_ms"] = metrics["total_kernel_duration_ms"]
        common["kernel_duration_ms"] = metrics["total_kernel_duration_ms"]
        if "kernels_profiled" in metrics:
            common["kernels_profiled"] = metrics["kernels_profiled"]
            common["total_kernels"] = len(metrics["kernels_profiled"])
    elif "kernel_durations" in metrics and isinstance(metrics["kernel_durations"], dict):
        # Sum durations from dict (kernel_name -> duration)
        # This is the per-kernel duration dict from parse_ncu_csv
        total = sum(metrics["kernel_durations"].values())
        common["time_ms"] = total
        common["kernel_duration_ms"] = total
        common["total_kernels"] = len(metrics["kernel_durations"])
        common["kernels_profiled"] = list(metrics["kernel_durations"].keys())
    elif "kernel_durations" in common and isinstance(common["kernel_durations"], list):
        # Sum individual kernel durations from list
        if common["kernel_durations"]:
            common["time_ms"] = sum(common["kernel_durations"])
            common["kernel_duration_ms"] = common["time_ms"]
            common["total_kernels"] = len(common["kernel_durations"])
    
    return common

