// Configuration
const API_BASE_URL = 'http://localhost:8000';

// State
let currentImage = null;
let currentImageBase64 = null;
let charts = {
    time: null,
    bandwidth: null
};

// Image modal navigation state
let modalImageIndex = 0;
let modalImages = []; // Array of {src, caption} objects

// DOM Elements
const dropZone = document.getElementById('dropZone');
const dropZoneContent = document.getElementById('dropZoneContent');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const imageInfo = document.getElementById('imageInfo');
const fileInput = document.getElementById('fileInput');
const removeImageBtn = document.getElementById('removeImage');
const filterType = document.getElementById('filterType');
const sigmaControl = document.getElementById('sigmaControl');
const sigmaSlider = document.getElementById('sigmaSlider');
const sigmaValue = document.getElementById('sigmaValue');
const radiusSlider = document.getElementById('radiusSlider');
const radiusValue = document.getElementById('radiusValue');
const processBtn = document.getElementById('processBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');
const originalImg = document.getElementById('originalImg');
const result1Img = document.getElementById('result1Img');
const result2Img = document.getElementById('result2Img');
const result1Container = document.getElementById('result1Container');
const result2Container = document.getElementById('result2Container');
const metricsTable = document.getElementById('metricsTable');
const metricsBody = document.getElementById('metricsBody');
const level1Header = document.getElementById('level1Header');
const level2Header = document.getElementById('level2Header');
const speedupHeader = document.getElementById('speedupHeader');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const closeErrorBtn = document.getElementById('closeError');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateFilterParams();
    checkAPIHealth();
    
    // Ensure process button is disabled initially
    if (processBtn) {
        processBtn.disabled = true;
    }
});

// Event Listeners
function setupEventListeners() {
    // File input
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('dragleave', handleDragLeave);
    
    // Remove image
    removeImageBtn.addEventListener('click', removeImage);
    
    // Filter controls
    filterType.addEventListener('change', updateFilterParams);
    sigmaSlider.addEventListener('input', () => {
        sigmaValue.textContent = sigmaSlider.value;
    });
    radiusSlider.addEventListener('input', () => {
        radiusValue.textContent = radiusSlider.value;
    });
    
    // Process button
    processBtn.addEventListener('click', processImage);
    
    // Error close
    closeErrorBtn.addEventListener('click', () => {
        errorMessage.classList.add('hidden');
    });
    
    // Modal close button
    const closeModalBtn = document.getElementById('closeModal');
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent modal backdrop click
            closeImageModal();
        });
    }
}

// File handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        loadImage(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('border-blue-500');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('border-blue-500');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('border-blue-500');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        loadImage(file);
    } else {
        showError('Please drop a valid image file');
    }
}

function loadImage(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        currentImageBase64 = e.target.result;
        previewImg.src = currentImageBase64;
        originalImg.src = currentImageBase64;
        
        // Get image dimensions
        const img = new Image();
        img.onload = () => {
            imageInfo.textContent = `${img.width} × ${img.height} pixels`;
            currentImage = {
                file: file,
                width: img.width,
                height: img.height,
                base64: currentImageBase64
            };
            
            // Show preview, hide drop zone content
            dropZoneContent.classList.add('hidden');
            imagePreview.classList.remove('hidden');
            processBtn.disabled = false;
        };
        img.src = currentImageBase64;
    };
    
    reader.readAsDataURL(file);
}

function removeImage() {
    currentImage = null;
    currentImageBase64 = null;
    fileInput.value = '';
    dropZoneContent.classList.remove('hidden');
    imagePreview.classList.add('hidden');
    resultsSection.classList.add('hidden');
    if (processBtn) processBtn.disabled = true;
}

// Filter parameters
function updateFilterParams() {
    const filter = filterType.value;
    const isGaussian = filter === 'gaussian';
    const isSobel = filter === 'sobel';
    
    // Show/hide sigma control (only for Gaussian)
    sigmaControl.classList.toggle('hidden', !isGaussian);
    
    // Show/hide radius control (not needed for Sobel)
    const radiusControl = document.getElementById('radiusControl');
    if (radiusControl) {
        radiusControl.classList.toggle('hidden', isSobel);
    }
}

// API calls
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        
        if (!data.gpu_available) {
            // Show as a warning, not an error, and auto-dismiss after 8 seconds
            showError('GPU module not available. GPU processing will not work until CUDA bindings are built.', 8000);
        }
    } catch (error) {
        showError('Cannot connect to API server. Make sure backend is running on port 8000.');
    }
}

async function processImage() {
    if (!currentImage) {
        showError('Please upload an image first');
        return;
    }
    
    loadingIndicator.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    
    try {
        // Process with ALL available optimization levels
        const allResults = await callProcessAllAPI();
        
        // Display results for all levels
        displayAllResults(allResults);
        
    } catch (error) {
        showError(`Processing failed: ${error.message}`);
    } finally {
        loadingIndicator.classList.add('hidden');
    }
}

async function callProcessAllAPI() {
    const filter = filterType.value;
    const sigma = filter === 'gaussian' ? parseFloat(sigmaSlider.value) : null;
    const radius = parseInt(radiusSlider.value);
    
    // Extract base64 data (remove data URL prefix if present)
    let imageData = currentImageBase64;
    if (imageData.includes(',')) {
        imageData = imageData.split(',')[1];
    }
    
    const requestBody = {
        image: imageData,
        filter: filter,
        level: 1,  // Level doesn't matter for process-all, but required by API
        radius: radius,
        enable_profiling: true  // Always enable profiling for detailed stats
    };
    
    if (sigma !== null) {
        requestBody.sigma = sigma;
    }
    
    const response = await fetch(`${API_BASE_URL}/api/process-all`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API request failed');
    }
    
    return await response.json();
}

// Display results for all levels
function displayAllResults(allResults) {
    // Set original image
    originalImg.src = allResults.original_image;
    
    // Get all level results
    const levelKeys = Object.keys(allResults.results).sort(); // Sort to get level_1, level_2, etc.
    const results = levelKeys.map(key => ({
        level: allResults.results[key].info.level_number,
        levelName: allResults.results[key].info.level,
        data: allResults.results[key]
    }));
    
    // Display images dynamically
    displayAllImages(results);
    
    // Display metrics for all levels
    displayAllMetrics(results);
    
    // Show results section
    resultsSection.classList.remove('hidden');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function displayAllImages(results) {
    const imageComparison = document.getElementById('imageComparison');
    
    // Clear existing result containers (keep original)
    const existingContainers = imageComparison.querySelectorAll('[id^="result"]');
    existingContainers.forEach(container => {
        if (container.id !== 'result1Container' && container.id !== 'result2Container') {
            container.remove();
        }
    });
    
    // Update grid to accommodate all images (original + results)
    const totalImages = 1 + results.length; // Original + results
    const gridCols = totalImages <= 3 ? totalImages : 3;
    imageComparison.className = `grid grid-cols-1 md:grid-cols-${gridCols} gap-6`;
    
    // Display each result
    results.forEach((result, index) => {
        let container;
        if (index === 0) {
            // Use first container
            container = result1Container;
            container.classList.remove('hidden');
        } else if (index === 1) {
            // Use second container
            container = result2Container;
            container.classList.remove('hidden');
        } else {
            // Create new container
            container = document.createElement('div');
            container.id = `result${index + 1}Container`;
            imageComparison.appendChild(container);
        }
        
        const img = container.querySelector('img') || document.createElement('img');
        img.id = `result${index + 1}Img`;
        img.src = result.data.processed_image;
        img.alt = `Level ${result.level} Result`;
        img.className = 'w-full rounded-lg shadow-md cursor-pointer hover:opacity-90 transition-opacity';
        img.onclick = () => expandImage(img);
        
        const heading = container.querySelector('h3') || document.createElement('h3');
        heading.className = 'text-sm font-medium text-gray-700 mb-2';
        heading.textContent = `Level ${result.level}: ${result.levelName.charAt(0).toUpperCase() + result.levelName.slice(1).replace('_', ' ')}`;
        
        if (!container.querySelector('h3')) {
            container.insertBefore(heading, img);
        }
        if (!container.querySelector('img')) {
            container.appendChild(img);
        }
    });
    
    // Hide unused containers
    if (results.length < 2) {
        result2Container.classList.add('hidden');
    }
}

function displayAllMetrics(results) {
    if (results.length === 0) return;
    
    // Update table headers dynamically
    const thead = metricsTable.querySelector('thead tr');
    thead.innerHTML = '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metric</th>';
    
    results.forEach((result, index) => {
        const th = document.createElement('th');
        th.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
        th.id = `level${result.level}Header`;
        th.textContent = `Level ${result.level}`;
        thead.appendChild(th);
    });
    
    // Add speedup column if we have multiple levels
    if (results.length > 1) {
        const speedupTh = document.createElement('th');
        speedupTh.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
        speedupTh.id = 'speedupHeader';
        speedupTh.textContent = 'Speedup vs Level 1';
        thead.appendChild(speedupTh);
    }
    
    // Calculate speedups (relative to level 1) - always use real execution time from CUDA events
    // time_ms is the real execution time (from CUDA events), not profiled time
    const baselineTime = results[0].data.metrics.time_ms || 0;
    const speedups = results.map(result => {
        const time = result.data.metrics.time_ms || 0;
        return result.level === 1 ? null : (baselineTime / time).toFixed(2);
    });
    
    // Helper function to format metric values with appropriate precision and units
    function formatValue(val, key = '') {
        if (val === undefined || val === null) return '<span class="text-gray-400">N/A</span>';
        if (typeof val === 'boolean') return val ? '<span class="text-green-600">✓</span>' : '<span class="text-red-600">✗</span>';
        if (typeof val === 'string') {
            // If it's a long string (like kernel names), truncate it
            if (val.length > 50) {
                return `<span title="${val}">${val.substring(0, 47)}...</span>`;
            }
            return val;
        }
        if (Array.isArray(val)) {
            if (val.length === 0) return '<span class="text-gray-400">[]</span>';
            // Format array values nicely
            if (val.length <= 3) {
                return val.map(v => typeof v === 'number' ? formatNumber(v, key) : String(v)).join(', ');
            }
            return `[${val.length} items]`;
        }
        if (typeof val === 'number') {
            return formatNumber(val, key);
        }
        return String(val);
    }
    
    // Helper to format numbers with appropriate precision
    function formatNumber(val, key = '') {
        const keyLower = key.toLowerCase();
        
        // Percentages
        if (keyLower.includes('pct') || keyLower.includes('percent') || keyLower.includes('occupancy') || 
            keyLower.includes('efficiency') || keyLower.includes('hit_rate') || keyLower.includes('busy')) {
            return `${val.toFixed(2)}%`;
        }
        
        // Large numbers (cycles, bytes)
        if (keyLower.includes('cycles') || keyLower.includes('bytes')) {
            if (val >= 1e9) return `${(val / 1e9).toFixed(2)}B`;
            if (val >= 1e6) return `${(val / 1e6).toFixed(2)}M`;
            if (val >= 1e3) return `${(val / 1e3).toFixed(2)}K`;
            return val.toFixed(0);
        }
        
        // Time values
        if (keyLower.includes('time') || keyLower.includes('duration') || keyLower.includes('ms')) {
            if (val >= 1000) return `${(val / 1000).toFixed(2)}s`;
            if (val >= 1) return `${val.toFixed(3)}ms`;
            if (val >= 0.001) return `${(val * 1000).toFixed(2)}μs`;
            return `${(val * 1000000).toFixed(2)}ns`;
        }
        
        // Throughput/bandwidth
        if (keyLower.includes('throughput') || keyLower.includes('bandwidth') || keyLower.includes('gbps')) {
            return `${val.toFixed(2)} GB/s`;
        }
        
        // General numbers
        if (val >= 1000) return val.toLocaleString('en-US', { maximumFractionDigits: 2 });
        if (val >= 1) return val.toFixed(3);
        if (val >= 0.01) return val.toFixed(4);
        return val.toFixed(6);
    }
    
    // Helper function to format metric names in a more readable way
    function formatMetricName(key) {
        // Remove section prefixes (e.g., "memory_", "occupancy_")
        let name = key;
        const sectionPrefixes = ['occupancy_', 'memory_', 'warp_', 'execution_', 'throughput_', 'config_'];
        for (const prefix of sectionPrefixes) {
            if (name.startsWith(prefix)) {
                name = name.substring(prefix.length);
                break;
            }
        }
        
        // Handle special cases for better readability
        const replacements = {
            'time_ms': 'Execution Time',
            'execution_time': 'Execution Time',
            'execution_time_ms': 'Execution Time',
            'execution_duration': 'Execution Time',
            'duration': 'Execution Time',  // Map Duration to Execution Time when real time is available
            'kernel_duration_ms': 'Kernel Duration',
            'elapsed_cycles': 'Elapsed Cycles',
            'occupancy_pct': 'Occupancy',
            'active_warps_per_scheduler': 'Active Warps/Scheduler',
            'eligible_warps_per_scheduler': 'Eligible Warps/Scheduler',
            'memory_throughput_gbps': 'Memory Throughput',
            'dram_throughput_pct': 'DRAM Throughput',
            'l1_hit_rate_pct': 'L1 Cache Hit Rate',
            'l2_hit_rate_pct': 'L2 Cache Hit Rate',
            'memory_busy_pct': 'Memory Busy',
            'warp_efficiency_pct': 'Warp Efficiency',
            'avg_active_threads_per_warp': 'Avg Active Threads/Warp',
            'sm_busy_pct': 'SM Busy',
            'compute_throughput_pct': 'Compute Throughput',
            'total_kernels': 'Total Kernels',
            'kernels_profiled': 'Kernels Profiled',
            'kernel_durations': 'Kernel Durations',
            'block_size': 'Block Size',
            'grid_size': 'Grid Size'
        };
        
        if (replacements[name]) {
            return replacements[name];
        }
        
        // General formatting
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase())
            .replace(/\bPct\b/g, ' (%)')
            .replace(/\bGbps\b/g, ' (GB/s)')
            .replace(/\bMs\b/g, ' (ms)')
            .replace(/\bUs\b/g, ' (μs)')
            .replace(/\bCycles\b/g, ' (cycles)')
            .replace(/\bAvg\b/g, 'Average')
            .replace(/\bPer\b/g, 'per')
            .replace(/\bTotal\b/g, 'Total')
            .replace(/\bActive\b/g, 'Active')
            .replace(/\bElapsed\b/g, 'Elapsed');
    }
    
    // Extract all metrics from ncu_data or metrics
    let allMetrics = new Set();
    results.forEach(result => {
        const metrics = result.data.metrics;
        
        // First, add all direct metrics (skip internal fields)
        Object.keys(metrics).forEach(key => {
            if (!['ncu_data', 'profiled_time_ms', 'ncu_profiled_time_ms'].includes(key)) {
                allMetrics.add(key);
            }
        });
        
        // Then, extract metrics from ncu_data structure if available
        // Flatten the nested structure to make all metrics accessible
        if (metrics.ncu_data && typeof metrics.ncu_data === 'object') {
            const ncuData = metrics.ncu_data;
            
            // Extract from categorized sections and add them with readable names
            ['occupancy', 'memory', 'warp', 'execution', 'throughput', 'config'].forEach(section => {
                if (ncuData[section] && typeof ncuData[section] === 'object') {
                    Object.keys(ncuData[section]).forEach(key => {
                        // Create a prefixed key to avoid conflicts
                        const prefixedKey = `${section}_${key.toLowerCase().replace(/\s+/g, '_')}`;
                        allMetrics.add(prefixedKey);
                    });
                }
            });
            
            // Extract kernel durations
            if (ncuData.kernel_durations && typeof ncuData.kernel_durations === 'object') {
                Object.keys(ncuData.kernel_durations).forEach(key => {
                    allMetrics.add(`kernel_duration_${key.toLowerCase().replace(/[^a-z0-9]/g, '_')}`);
                });
            }
            
            // Also extract total_kernel_duration_ms and kernels_profiled if they exist
            if (ncuData.total_kernel_duration_ms !== undefined) {
                allMetrics.add('total_kernel_duration_ms');
            }
            if (ncuData.kernels_profiled !== undefined) {
                allMetrics.add('kernels_profiled');
            }
        }
    });
    
    // Helper to get metric value from result
    function getMetricValue(result, key) {
        const metrics = result.data.metrics;
        const keyLower = key.toLowerCase();
        
        // Check if this metric will be displayed as "Execution Time"
        // This must happen BEFORE we check ncu_data to prevent profiled times from being used
        const formattedName = formatMetricName(key);
        const willDisplayAsExecutionTime = 
            formattedName === 'Execution Time' || 
            formattedName.toLowerCase().includes('execution time');
        
        // For execution time metrics, ALWAYS use real execution time from CUDA events
        // Never use profiled time from ncu_data
        // Check for various time-related metric patterns
        const isTimeMetric = 
            key === 'time_ms' || 
            key === 'execution_time' || 
            keyLower === 'execution time' ||
            keyLower === 'duration' ||
            keyLower.includes('execution_time') ||
            keyLower.includes('execution time') ||
            keyLower.includes('execution_duration') ||
            (keyLower.includes('duration') && !keyLower.includes('kernel') && !keyLower.includes('elapsed')) ||
            willDisplayAsExecutionTime;
        
        if (isTimeMetric || willDisplayAsExecutionTime) {
            // Always return real execution time from top-level metrics
            // This is the actual kernel execution time from CUDA events, not profiled time
            if (metrics['time_ms'] !== undefined && metrics['time_ms'] !== null) {
                return metrics['time_ms'];
            }
            // Don't fall back to ncu_data for time metrics - we want real execution time only
            return undefined;
        }
        
        // Check direct metrics first (these are the common metrics from get_common_ncu_metrics)
        if (metrics[key] !== undefined && metrics[key] !== null) {
            return metrics[key];
        }
        
        // Check ncu_data structure for nested metrics
        if (metrics.ncu_data && typeof metrics.ncu_data === 'object') {
            const ncuData = metrics.ncu_data;
            
            // Check top-level ncu_data fields
            if (ncuData[key] !== undefined && ncuData[key] !== null) {
                return ncuData[key];
            }
            
            // Check if it's a prefixed key (section_key format)
            const parts = key.split('_');
            if (parts.length >= 2) {
                const section = parts[0];
                const originalKey = parts.slice(1).join('_');
                
                if (ncuData[section] && typeof ncuData[section] === 'object') {
                    // Try to find matching key (case-insensitive, space-insensitive)
                    const sectionData = ncuData[section];
                    for (const [k, v] of Object.entries(sectionData)) {
                        const normalizedKey = k.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '');
                        const normalizedOriginal = originalKey.toLowerCase().replace(/[^a-z0-9_]/g, '');
                        if (normalizedKey === normalizedOriginal) {
                            return v;
                        }
                    }
                }
            }
            
            // Check kernel durations
            if (key.startsWith('kernel_duration_') && ncuData.kernel_durations) {
                const kernelName = key.replace('kernel_duration_', '').replace(/_/g, ' ');
                for (const [k, v] of Object.entries(ncuData.kernel_durations)) {
                    if (k.toLowerCase().replace(/[^a-z0-9]/g, '_') === kernelName.replace(/[^a-z0-9]/g, '_')) {
                        return v;
                    }
                }
            }
        }
        
        return undefined;
    }
    
    // Organize metrics by category
    // Include common metric patterns to categorize dynamically discovered metrics
    const categories = {
        'Execution Time': ['time_ms', 'kernel_duration_ms', 'kernel_durations', 'elapsed_cycles', 'total_kernels', 'kernels_profiled', 'duration', 'cycles'],
        'Occupancy': ['occupancy', 'active_warps', 'eligible_warps'],
        'Memory': ['memory', 'dram', 'l1', 'l2', 'cache', 'throughput'],
        'Warp': ['warp', 'threads'],
        'Compute': ['sm', 'compute', 'instruction'],
        'Configuration': ['block', 'grid', 'config', 'shared'],
        'Other': []
    };
    
    // Categorize all metrics (match by pattern)
    const categorized = {};
    allMetrics.forEach(key => {
        let categorized_flag = false;
        const keyLower = key.toLowerCase();
        for (const [cat, patterns] of Object.entries(categories)) {
            if (patterns.some(pattern => keyLower.includes(pattern.toLowerCase()))) {
                if (!categorized[cat]) categorized[cat] = [];
                categorized[cat].push(key);
                categorized_flag = true;
                break;
            }
        }
        if (!categorized_flag) {
            if (!categorized['Other']) categorized['Other'] = [];
            categorized['Other'].push(key);
        }
    });
    
    // Build metrics HTML organized by category
    let metricsHTML = '';
    
    // Add summary header showing total metrics count
    const totalMetricsCount = Object.values(categorized).reduce((sum, arr) => sum + arr.length, 0);
    if (totalMetricsCount > 0) {
        metricsHTML += `
        <tr class="bg-gradient-to-r from-blue-50 to-indigo-50 border-b-2 border-blue-300">
            <td colspan="${results.length + (results.length > 1 ? 2 : 1)}" class="px-6 py-4">
                <div class="flex items-center justify-between">
                    <div>
                        <span class="text-lg font-bold text-blue-900">📊 Nsight Compute Metrics</span>
                        <span class="text-sm text-blue-700 ml-2">(${totalMetricsCount} metrics)</span>
                    </div>
                    <div class="text-xs text-blue-600">
                        Real kernel execution times from CUDA events (without profiling overhead)
                    </div>
                </div>
            </td>
        </tr>
        `;
    }
    
    // Define category icons and descriptions
    const categoryInfo = {
        'Execution Time': { icon: '⏱️', desc: 'Kernel execution timing metrics' },
        'Configuration': { icon: '⚙️', desc: 'Launch configuration parameters' },
        'Occupancy': { icon: '📈', desc: 'Warp occupancy and scheduling' },
        'Memory': { icon: '💾', desc: 'Memory access patterns and throughput' },
        'Warp': { icon: '🔄', desc: 'Warp execution efficiency' },
        'Compute': { icon: '⚡', desc: 'Compute unit utilization' },
        'Other': { icon: '📋', desc: 'Additional metrics' }
    };
    
    // Display categories in order
    const categoryOrder = ['Execution Time', 'Configuration', 'Occupancy', 'Memory', 'Warp', 'Compute', 'Other'];
    
    categoryOrder.forEach(category => {
        if (!categorized[category] || categorized[category].length === 0) return;
        
        const info = categoryInfo[category] || { icon: '📊', desc: '' };
        const metricCount = categorized[category].length;
        
        // Category header with icon and description
        metricsHTML += `
        <tr class="bg-gradient-to-r from-gray-100 to-gray-50 border-t-2 border-gray-300">
            <td colspan="${results.length + (results.length > 1 ? 2 : 1)}" class="px-6 py-3">
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-2">
                        <span class="text-lg">${info.icon}</span>
                        <span class="text-sm font-bold text-gray-800 uppercase tracking-wide">${category}</span>
                        <span class="text-xs text-gray-600">(${metricCount} metrics)</span>
                    </div>
                    ${info.desc ? `<span class="text-xs text-gray-500 italic">${info.desc}</span>` : ''}
                </div>
            </td>
        </tr>
        `;
        
        // Metrics in this category - highlight important ones
        const importantMetrics = ['time_ms', 'occupancy_pct', 'memory_throughput_gbps', 'dram_throughput_pct', 
                                  'l1_hit_rate_pct', 'l2_hit_rate_pct', 'warp_efficiency_pct', 'sm_busy_pct'];
        
        categorized[category].forEach(key => {
            // For Execution Time category, skip redundant profiled time metrics
            if (category === 'Execution Time') {
                const keyLower = key.toLowerCase();
                // Skip "Duration" and similar profiled metrics if we have real execution time
                const isProfiledTimeMetric = (
                    keyLower === 'duration' || 
                    keyLower.includes('execution_duration') ||
                    (keyLower.includes('duration') && !keyLower.includes('kernel') && !keyLower.includes('elapsed'))
                ) && key !== 'time_ms' && key !== 'execution_time';
                
                if (isProfiledTimeMetric) {
                    // Check if we have real execution time available
                    const hasRealTime = results.some(r => {
                        const realTime = r.data.metrics.time_ms;
                        return realTime !== undefined && realTime !== null && realTime > 0;
                    });
                    // Skip profiled duration metrics if we have real execution time
                    if (hasRealTime) {
                        return;
                    }
                }
            }
            
            // Check if any result has this metric (with a value, not just undefined)
            const hasMetric = results.some(r => {
                const val = getMetricValue(r, key);
                return val !== undefined && val !== null && val !== '';
            });
            if (!hasMetric) return;
            
            const isImportant = importantMetrics.some(imp => key.includes(imp.replace('_', '')));
            const rowClass = isImportant ? 'bg-blue-50 hover:bg-blue-100' : 'hover:bg-gray-50';
            const fontClass = isImportant ? 'font-semibold text-gray-900' : 'font-medium text-gray-800';
            
            metricsHTML += `
            <tr class="${rowClass} transition-colors">
                <td class="px-6 py-3 ${fontClass}">
                    ${isImportant ? '<span class="text-blue-600 mr-2">★</span>' : ''}
                    ${formatMetricName(key)}
                </td>
                ${results.map(result => {
                    // getMetricValue already handles Execution Time metrics correctly
                    // It checks formatMetricName internally and returns real execution time
                    const val = getMetricValue(result, key);
                    const formattedVal = formatValue(val, key);
                    // Add color coding for percentages
                    let cellClass = 'px-6 py-3 text-sm text-gray-700';
                    if (typeof val === 'number' && key.toLowerCase().includes('pct')) {
                        if (val >= 80) cellClass += ' text-green-700 font-semibold';
                        else if (val >= 50) cellClass += ' text-yellow-700';
                        else if (val > 0) cellClass += ' text-orange-700';
                    }
                    return `<td class="${cellClass}">${formattedVal}</td>`;
                }).join('')}
                ${results.length > 1 ? speedups.map(s => {
                    if (s && key === 'time_ms' && s !== 'NaN' && !isNaN(parseFloat(s))) {
                        const speedup = parseFloat(s);
                        const speedupClass = speedup > 1.5 ? 'text-green-600 font-bold' : speedup > 1.1 ? 'text-green-500' : 'text-gray-600';
                        return `<td class="px-6 py-3 text-sm ${speedupClass}">${s}×</td>`;
                    }
                    return '<td class="px-6 py-3 text-sm text-gray-400">-</td>';
                }).join('') : ''}
            </tr>
            `;
        });
    });
    
    // If no metrics found, show message
    if (metricsHTML === '') {
        metricsHTML = `
        <tr>
            <td colspan="${results.length + (results.length > 1 ? 2 : 1)}" class="px-6 py-4 text-center text-sm text-gray-500">
                No Nsight Compute metrics available. Enable profiling to see detailed metrics.
            </td>
        </tr>
        `;
    }
    
    metricsBody.innerHTML = metricsHTML;
    
    // Update charts (still use time_ms and bandwidth_gbps for charts)
    updateAllCharts(results);
}

function updateAllCharts(results) {
    // Destroy existing charts
    if (charts.time) charts.time.destroy();
    if (charts.bandwidth) charts.bandwidth.destroy();
    
    const labels = results.map(r => `Level ${r.level}`);
    // Always use real execution time from CUDA events (time_ms)
    // This is the actual kernel execution time without profiling overhead
    const timeData = results.map(r => r.data.metrics.time_ms || 0);
    // Use bandwidth_gbps from metrics (calculated from real execution time)
    const bandwidthData = results.map(r => {
        if (r.data.metrics.bandwidth_gbps) return r.data.metrics.bandwidth_gbps;
        // Calculate from real execution time if bandwidth not available
        const time = r.data.metrics.time_ms || 0;
        if (time && r.data.info) {
            const size = r.data.info.width * r.data.info.height * r.data.info.channels * 4; // 2 passes * 2 (input+temp)
            return (size / (time / 1000.0)) / (1024.0 ** 3); // GB/s
        }
        return 0;
    });
    
    // Color palette for multiple levels
    const colors = [
        { bg: 'rgba(59, 130, 246, 0.5)', border: 'rgb(59, 130, 246)' },
        { bg: 'rgba(16, 185, 129, 0.5)', border: 'rgb(16, 185, 129)' },
        { bg: 'rgba(245, 158, 11, 0.5)', border: 'rgb(245, 158, 11)' },
        { bg: 'rgba(239, 68, 68, 0.5)', border: 'rgb(239, 68, 68)' }
    ];
    
    // Time chart
    const timeCanvas = document.getElementById('timeChart');
    if (!timeCanvas) {
        console.error('timeChart canvas not found');
        return;
    }
    const timeCtx = timeCanvas.getContext('2d');
    charts.time = new Chart(timeCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Execution Time (ms)',
                data: timeData,
                backgroundColor: results.map((_, i) => colors[i % colors.length].bg),
                borderColor: results.map((_, i) => colors[i % colors.length].border),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    // Bandwidth chart
    const bandwidthCanvas = document.getElementById('bandwidthChart');
    if (!bandwidthCanvas) {
        console.error('bandwidthChart canvas not found');
        return;
    }
    const bandwidthCtx = bandwidthCanvas.getContext('2d');
    charts.bandwidth = new Chart(bandwidthCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Memory Bandwidth (GB/s)',
                data: bandwidthData,
                backgroundColor: results.map((_, i) => colors[i % colors.length].bg),
                borderColor: results.map((_, i) => colors[i % colors.length].border),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    // Log chart creation for debugging
    console.log('Charts created:', { timeData, bandwidthData, labels });
}

// Error handling
function showError(message, autoHideMs = 5000) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
    
    // Auto-hide after specified time (default 5 seconds)
    setTimeout(() => {
        errorMessage.classList.add('hidden');
    }, autoHideMs);
}

// Build array of available images for navigation
function buildImageArray() {
    const images = [];
    
    // Original image
    const originalImg = document.getElementById('originalImg');
    if (originalImg && originalImg.src) {
        images.push({
            src: originalImg.src,
            caption: 'Original Image',
            id: 'originalImg'
        });
    }
    
    // Find all result images dynamically
    const imageComparison = document.getElementById('imageComparison');
    if (imageComparison) {
        const resultContainers = imageComparison.querySelectorAll('[id^="result"][id$="Container"]:not([id="result1Container"]):not([id="result2Container"]), #result1Container, #result2Container');
        
        resultContainers.forEach(container => {
            if (!container.classList.contains('hidden')) {
                const img = container.querySelector('img');
                const heading = container.querySelector('h3');
                
                if (img && img.src) {
                    const caption = heading ? heading.textContent + ' Result' : 'Result';
                    images.push({
                        src: img.src,
                        caption: caption,
                        id: img.id
                    });
                }
            }
        });
        
        // Also check result1Container and result2Container specifically
        const result1Img = document.getElementById('result1Img');
        const result1Container = document.getElementById('result1Container');
        if (result1Img && result1Img.src && result1Container && !result1Container.classList.contains('hidden')) {
            const heading = result1Container.querySelector('h3');
            const caption = heading ? heading.textContent + ' Result' : 'Level 1 Result';
            if (!images.find(img => img.id === 'result1Img')) {
                images.push({
                    src: result1Img.src,
                    caption: caption,
                    id: 'result1Img'
                });
            }
        }
        
        const result2Img = document.getElementById('result2Img');
        const result2Container = document.getElementById('result2Container');
        if (result2Img && result2Img.src && result2Container && !result2Container.classList.contains('hidden')) {
            const heading = result2Container.querySelector('h3');
            const caption = heading ? heading.textContent + ' Result' : 'Level 2 Result';
            if (!images.find(img => img.id === 'result2Img')) {
                images.push({
                    src: result2Img.src,
                    caption: caption,
                    id: 'result2Img'
                });
            }
        }
    }
    
    return images;
}

// Image expansion modal
function expandImage(imgElement) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    
    if (!modal || !modalImage) return;
    
    // Build array of available images
    modalImages = buildImageArray();
    
    if (modalImages.length === 0) return;
    
    // Find which image was clicked
    let clickedIndex = 0;
    for (let i = 0; i < modalImages.length; i++) {
        if (modalImages[i].id === imgElement.id) {
            clickedIndex = i;
            break;
        }
    }
    
    modalImageIndex = clickedIndex;
    showModalImage(modalImageIndex);
    
    // Show modal
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden'; // Prevent background scrolling
    
    // Keyboard navigation handler
    const handleKeyboard = (e) => {
        if (e.key === 'Escape') {
            closeImageModal();
            document.removeEventListener('keydown', handleKeyboard);
        } else if (e.key === 'ArrowLeft') {
            e.preventDefault();
            navigateImage(-1);
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            navigateImage(1);
        }
    };
    document.addEventListener('keydown', handleKeyboard);
}

// Show image at specific index in modal
function showModalImage(index) {
    if (modalImages.length === 0 || index < 0 || index >= modalImages.length) return;
    
    const modalImage = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    const modalCounter = document.getElementById('modalCounter');
    const prevBtn = document.getElementById('prevImageBtn');
    const nextBtn = document.getElementById('nextImageBtn');
    
    modalImageIndex = index;
    const imageData = modalImages[index];
    
    modalImage.src = imageData.src;
    modalCaption.textContent = imageData.caption;
    
    // Update counter
    if (modalCounter) {
        modalCounter.textContent = `${index + 1} / ${modalImages.length}`;
    }
    
    // Show/hide navigation buttons
    if (prevBtn) {
        prevBtn.style.display = modalImages.length > 1 ? 'flex' : 'none';
    }
    if (nextBtn) {
        nextBtn.style.display = modalImages.length > 1 ? 'flex' : 'none';
    }
}

// Navigate to next/previous image
function navigateImage(direction) {
    if (modalImages.length <= 1) return;
    
    let newIndex = modalImageIndex + direction;
    
    // Wrap around
    if (newIndex < 0) {
        newIndex = modalImages.length - 1;
    } else if (newIndex >= modalImages.length) {
        newIndex = 0;
    }
    
    showModalImage(newIndex);
}

function closeImageModal(event) {
    // If event is provided and target is the image container, don't close
    if (event && event.target && event.target.closest('.flex.flex-col.items-center')) {
        return;
    }
    
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.classList.add('hidden');
        document.body.style.overflow = ''; // Restore scrolling
        modalImages = [];
        modalImageIndex = 0;
    }
}

// Make functions globally available
window.expandImage = expandImage;
window.closeImageModal = closeImageModal;
window.navigateImage = navigateImage;

