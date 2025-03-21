/**
 * Benchmark tests for the Prime Compression algorithm against industry standards
 * 
 * This test suite focuses on comparing the performance and compression ratios
 * of the Prime Compression algorithm against common industry standards like
 * GZIP, Brotli, LZMA, and others when available.
 */

// Import the compression module
const compression = require('../prime-compression.js');

// Add the performance.now polyfill if needed
if (typeof performance === 'undefined') {
  global.performance = {
    now: function() {
      return Date.now();
    }
  };
}

// Import Node.js modules for standard compression algorithms
let zlib, lzma, brotli;
try {
  zlib = require('zlib');
} catch (e) {
  console.warn('zlib not available for comparison benchmarks');
}

try {
  lzma = require('lzma-native');
} catch (e) {
  console.warn('lzma-native not available, skipping LZMA benchmarks');
}

// Polyfill for TextEncoder/TextDecoder if in Node.js environment
if (typeof TextEncoder === 'undefined') {
  const util = require('util');
  global.TextEncoder = util.TextEncoder;
  global.TextDecoder = util.TextDecoder;
}

// Helper function to format byte sizes
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// Helper function to format times
function formatTime(ms) {
  if (ms < 1) return `${(ms * 1000).toFixed(2)} μs`;
  if (ms < 1000) return `${ms.toFixed(2)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

// Data generation utilities
const TestDataGenerator = {
  // Generate text data
  generateText(size, type = 'english') {
    // Common English words for text generation
    const words = [
      'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 
      'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
      'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
      'prime', 'compression', 'algorithm', 'spectral', 'analysis', 'quantum', 
      'framework', 'data', 'pattern', 'recognition', 'fourier', 'transform',
      'optimization', 'vector', 'matrix', 'numeric', 'compute', 'processor'
    ];
    
    // Generate text according to type
    let text = '';
    switch (type) {
      case 'english':
        // Generate quasi-random English text
        while (text.length < size) {
          const sentenceLength = 5 + Math.floor(Math.random() * 10);
          let sentence = '';
          
          for (let i = 0; i < sentenceLength; i++) {
            const wordIndex = Math.floor(Math.random() * words.length);
            const word = words[wordIndex];
            
            if (i === 0) {
              // Capitalize first word
              sentence += word.charAt(0).toUpperCase() + word.slice(1);
            } else {
              sentence += word;
            }
            
            if (i < sentenceLength - 1) {
              sentence += ' ';
            }
          }
          
          sentence += '. ';
          text += sentence;
        }
        break;
        
      case 'json':
        // Generate JSON-like data
        const jsonObj = {
          data: [],
          metadata: {
            version: '1.0',
            type: 'prime-benchmark',
            size: size,
            description: 'Test data for Prime Compression benchmark',
            framework: 'Prime Framework',
            algorithms: ['spectral', 'pattern', 'dictionary', 'statistical']
          }
        };
        
        // Add some array data
        for (let i = 0; i < Math.min(1000, size / 50); i++) {
          jsonObj.data.push({
            id: i,
            value: Math.random() * 100,
            name: `Item ${i}`,
            active: i % 2 === 0,
            properties: {
              category: i % 5,
              tags: ['prime', 'test', i % 3 === 0 ? 'important' : 'normal'],
              created: new Date().toISOString()
            }
          });
        }
        
        text = JSON.stringify(jsonObj, null, 2);
        break;
        
      case 'html':
        // Generate HTML-like data
        text = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Prime Compression Benchmark</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      max-width: 800px;
      margin: 0 auto;
      padding: 20px;
    }
    h1 { color: #333; }
    .data-container { border: 1px solid #ddd; padding: 15px; }
    .result { margin-bottom: 10px; }
    .result-good { color: green; }
    .result-bad { color: red; }
  </style>
</head>
<body>
  <h1>Prime Compression Benchmark Results</h1>
  <div class="data-container">`;
        
        // Generate content to fill up to size
        while (text.length < size - 100) {
          text += `
    <div class="result">
      <h2>Test Case #${Math.floor(text.length / 100)}</h2>
      <p>Data size: ${Math.floor(Math.random() * 1000)} bytes</p>
      <p>Compression ratio: ${(Math.random() * 20).toFixed(2)}x</p>
      <p class="result-${Math.random() > 0.7 ? 'bad' : 'good'}">
        ${Math.random() > 0.7 ? 'Failed to meet expected ratio' : 'Compression successful'}
      </p>
    </div>`;
        }
        
        text += `
  </div>
  <script>
    document.addEventListener('DOMContentLoaded', function() {
      console.log('Benchmark page loaded');
      const results = document.querySelectorAll('.result');
      let totalSuccess = 0;
      
      results.forEach(result => {
        if (result.querySelector('.result-good')) {
          totalSuccess++;
        }
      });
      
      console.log(\`Success rate: \${(totalSuccess / results.length * 100).toFixed(2)}%\`);
    });
  </script>
</body>
</html>`;
        break;
    }
    
    // Truncate or pad to exact size
    if (text.length > size) {
      text = text.substring(0, size);
    } else while (text.length < size) {
      text += ' ';
    }
    
    // Convert to Uint8Array
    const encoder = new TextEncoder();
    return encoder.encode(text.substring(0, size));
  },

  // Generate binary data
  generateBinary(size, pattern = 'random') {
    const data = new Uint8Array(size);
    
    switch (pattern) {
      case 'random':
        // Random data (less compressible)
        for (let i = 0; i < size; i++) {
          data[i] = Math.floor(Math.random() * 256);
        }
        break;
        
      case 'sequential':
        // Sequential data (more compressible)
        for (let i = 0; i < size; i++) {
          data[i] = i % 256;
        }
        break;
        
      case 'repeated':
        // Repeated pattern (highly compressible)
        const patternLength = 16;
        const repeatedPattern = new Uint8Array(patternLength);
        for (let i = 0; i < patternLength; i++) {
          repeatedPattern[i] = Math.floor(Math.random() * 256);
        }
        
        for (let i = 0; i < size; i++) {
          data[i] = repeatedPattern[i % patternLength];
        }
        break;
        
      case 'zeros':
        // All zeros (maximally compressible)
        data.fill(0);
        break;
        
      case 'sine':
        // Sine wave (compressible with spectral)
        const frequency = 0.05;
        const amplitude = 127;
        const center = 128;
        
        for (let i = 0; i < size; i++) {
          data[i] = Math.round(center + amplitude * Math.sin(2 * Math.PI * frequency * i));
        }
        break;
        
      case 'mixed':
        // Mixed patterns (to test adaptability)
        const sectionSize = Math.floor(size / 5);
        
        // Section 1: Random data
        for (let i = 0; i < sectionSize; i++) {
          data[i] = Math.floor(Math.random() * 256);
        }
        
        // Section 2: Sequential data
        for (let i = 0; i < sectionSize; i++) {
          data[sectionSize + i] = i % 256;
        }
        
        // Section 3: Sine wave
        for (let i = 0; i < sectionSize; i++) {
          data[sectionSize * 2 + i] = Math.round(128 + 127 * Math.sin(2 * Math.PI * 0.05 * i));
        }
        
        // Section 4: Repeated pattern
        const pattern = [1, 2, 3, 4, 5, 4, 3, 2];
        for (let i = 0; i < sectionSize; i++) {
          data[sectionSize * 3 + i] = pattern[i % pattern.length];
        }
        
        // Section 5: Zeros
        for (let i = 0; i < size - (sectionSize * 4); i++) {
          data[sectionSize * 4 + i] = 0;
        }
        break;
    }
    
    return data;
  },
  
  // Generate real-world-like data (image)
  generateImageLike(width, height) {
    const size = width * height * 3; // RGB
    const data = new Uint8Array(size);
    
    // Create a gradient background
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pos = (y * width + x) * 3;
        data[pos] = Math.floor((x / width) * 255);       // R
        data[pos + 1] = Math.floor((y / height) * 255);  // G
        data[pos + 2] = Math.floor(((x + y) / (width + height)) * 255); // B
      }
    }
    
    // Add some "shapes" - rectangles of solid color
    for (let i = 0; i < 5; i++) {
      const rectX = Math.floor(Math.random() * (width - 50));
      const rectY = Math.floor(Math.random() * (height - 50));
      const rectW = 20 + Math.floor(Math.random() * 30);
      const rectH = 20 + Math.floor(Math.random() * 30);
      const color = [
        Math.floor(Math.random() * 256),
        Math.floor(Math.random() * 256),
        Math.floor(Math.random() * 256)
      ];
      
      for (let y = rectY; y < rectY + rectH && y < height; y++) {
        for (let x = rectX; x < rectX + rectW && x < width; x++) {
          const pos = (y * width + x) * 3;
          data[pos] = color[0];
          data[pos + 1] = color[1];
          data[pos + 2] = color[2];
        }
      }
    }
    
    return data;
  }
};

// Benchmark runner
const BenchmarkRunner = {
  // Run a compression benchmark
  runBenchmark(dataType, dataSize, dataPattern = null) {
    console.log(`\n=== Benchmarking ${dataType} data (${formatBytes(dataSize)}) ===`);
    
    // Generate test data
    let data;
    switch (dataType) {
      case 'text':
        data = TestDataGenerator.generateText(dataSize, dataPattern || 'english');
        break;
      case 'binary':
        data = TestDataGenerator.generateBinary(dataSize, dataPattern || 'random');
        break;
      case 'image':
        // Approximate dimensions for given size
        const pixelCount = Math.floor(dataSize / 3);
        const dimension = Math.floor(Math.sqrt(pixelCount));
        data = TestDataGenerator.generateImageLike(dimension, dimension);
        break;
      default:
        throw new Error(`Unknown data type: ${dataType}`);
    }
    
    // Results object
    const results = {
      dataType,
      dataSize,
      dataPattern,
      prime: { time: 0, size: 0, ratio: 0, strategy: '' },
      gzip: { time: 0, size: 0, ratio: 0 },
      brotli: { time: 0, size: 0, ratio: 0 },
      lzma: { time: 0, size: 0, ratio: 0 }
    };
    
    // Benchmark Prime Compression
    console.log("Running Prime Compression...");
    try {
      const startTime = performance.now();
      const compressed = compression.compress(data);
      results.prime.time = performance.now() - startTime;
      
      results.prime.size = compressed.compressedSize || 
                         (compressed.compressedVector ? compressed.compressedVector.length : 0);
      results.prime.ratio = compressed.compressionRatio || (data.length / results.prime.size);
      results.prime.strategy = compressed.strategy || compressed.compressionMethod || 'unknown';
      
      console.log(`Prime: ${results.prime.ratio.toFixed(2)}x ratio in ${formatTime(results.prime.time)}`);
      console.log(`Strategy: ${results.prime.strategy}`);
      
      // Verify decompression works
      const decompStartTime = performance.now();
      const decompressed = compression.decompress(compressed);
      const decompTime = performance.now() - decompStartTime;
      
      // Check integrity (just first/last few bytes for speed)
      const integrityPassed = 
        decompressed.length === data.length &&
        decompressed[0] === data[0] &&
        decompressed[1] === data[1] &&
        decompressed[data.length - 2] === data[data.length - 2] &&
        decompressed[data.length - 1] === data[data.length - 1];
      
      console.log(`Decompression time: ${formatTime(decompTime)}`);
      console.log(`Integrity check: ${integrityPassed ? 'PASSED' : 'FAILED'}`);
      
      if (!integrityPassed) {
        console.error(`Original length: ${data.length}, Decompressed length: ${decompressed.length}`);
        results.prime.error = 'Decompression integrity check failed';
      }
    } catch (e) {
      console.error(`Prime compression error: ${e.message}`);
      results.prime.error = e.message;
    }
    
    // Benchmark GZIP if available
    if (zlib) {
      console.log("\nRunning GZIP...");
      try {
        const buffer = Buffer.from(data);
        
        // Compress
        const startTime = performance.now();
        const compressed = zlib.gzipSync(buffer, { level: 9 });
        results.gzip.time = performance.now() - startTime;
        
        results.gzip.size = compressed.length;
        results.gzip.ratio = data.length / compressed.length;
        
        console.log(`GZIP: ${results.gzip.ratio.toFixed(2)}x ratio in ${formatTime(results.gzip.time)}`);
        
        // Decompress to verify
        const decompStartTime = performance.now();
        const decompressed = zlib.gunzipSync(compressed);
        const decompTime = performance.now() - decompStartTime;
        
        console.log(`Decompression time: ${formatTime(decompTime)}`);
      } catch (e) {
        console.error(`GZIP error: ${e.message}`);
        results.gzip.error = e.message;
      }
    }
    
    // Benchmark Brotli if available
    if (zlib && zlib.brotliCompressSync) {
      console.log("\nRunning Brotli...");
      try {
        const buffer = Buffer.from(data);
        
        // Compress
        const startTime = performance.now();
        const compressed = zlib.brotliCompressSync(buffer, { 
          params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 11 }
        });
        results.brotli.time = performance.now() - startTime;
        
        results.brotli.size = compressed.length;
        results.brotli.ratio = data.length / compressed.length;
        
        console.log(`Brotli: ${results.brotli.ratio.toFixed(2)}x ratio in ${formatTime(results.brotli.time)}`);
        
        // Decompress to verify
        const decompStartTime = performance.now();
        const decompressed = zlib.brotliDecompressSync(compressed);
        const decompTime = performance.now() - decompStartTime;
        
        console.log(`Decompression time: ${formatTime(decompTime)}`);
      } catch (e) {
        console.error(`Brotli error: ${e.message}`);
        results.brotli.error = e.message;
      }
    }
    
    // Benchmark LZMA if available
    if (lzma && lzma.compressSync) {
      console.log("\nRunning LZMA...");
      try {
        const buffer = Buffer.from(data);
        
        // Compress
        const startTime = performance.now();
        const compressed = lzma.compressSync(buffer, { preset: 9 });
        results.lzma.time = performance.now() - startTime;
        
        results.lzma.size = compressed.length;
        results.lzma.ratio = data.length / compressed.length;
        
        console.log(`LZMA: ${results.lzma.ratio.toFixed(2)}x ratio in ${formatTime(results.lzma.time)}`);
        
        // Decompress to verify
        const decompStartTime = performance.now();
        const decompressed = lzma.decompressSync(compressed);
        const decompTime = performance.now() - decompStartTime;
        
        console.log(`Decompression time: ${formatTime(decompTime)}`);
      } catch (e) {
        console.error(`LZMA error: ${e.message}`);
        results.lzma.error = e.message;
      }
    }
    
    // Compare results
    console.log("\n--- Comparison ---");
    
    // Create array of successful algorithms for comparison
    const algorithms = [];
    if (!results.prime.error) algorithms.push({name: 'Prime', ratio: results.prime.ratio, time: results.prime.time});
    if (!results.gzip.error) algorithms.push({name: 'GZIP', ratio: results.gzip.ratio, time: results.gzip.time});
    if (!results.brotli.error) algorithms.push({name: 'Brotli', ratio: results.brotli.ratio, time: results.brotli.time});
    if (!results.lzma.error) algorithms.push({name: 'LZMA', ratio: results.lzma.ratio, time: results.lzma.time});
    
    if (algorithms.length > 1) {
      // Sort by compression ratio (highest first)
      algorithms.sort((a, b) => b.ratio - a.ratio);
      console.log("Best compression ratio:");
      for (let i = 0; i < algorithms.length; i++) {
        const alg = algorithms[i];
        console.log(`${i + 1}. ${alg.name}: ${alg.ratio.toFixed(2)}x`);
      }
      
      // Sort by compression time (fastest first)
      algorithms.sort((a, b) => a.time - b.time);
      console.log("\nFastest compression:");
      for (let i = 0; i < algorithms.length; i++) {
        const alg = algorithms[i];
        console.log(`${i + 1}. ${alg.name}: ${formatTime(alg.time)}`);
      }
      
      // Calculate space-time efficiency (ratio / log(time))
      // This favors algorithms with good compression ratios that aren't too slow
      algorithms.forEach(alg => {
        alg.efficiency = alg.ratio / Math.log10(1 + alg.time);
      });
      
      algorithms.sort((a, b) => b.efficiency - a.efficiency);
      console.log("\nBest space-time efficiency (ratio/log(time)):");
      for (let i = 0; i < algorithms.length; i++) {
        const alg = algorithms[i];
        console.log(`${i + 1}. ${alg.name}: ${alg.efficiency.toFixed(2)}`);
      }
    } else if (algorithms.length === 1) {
      console.log(`Only ${algorithms[0].name} completed successfully`);
    } else {
      console.log("No algorithms completed successfully");
    }
    
    return results;
  }
};

// Define benchmark test cases
const benchmarkTestCases = [
  // Text data tests
  { dataType: 'text', dataSize: 10 * 1024, dataPattern: 'english' }, // 10KB
  { dataType: 'text', dataSize: 100 * 1024, dataPattern: 'english' }, // 100KB
  { dataType: 'text', dataSize: 50 * 1024, dataPattern: 'json' }, // 50KB JSON
  { dataType: 'text', dataSize: 100 * 1024, dataPattern: 'html' }, // 100KB HTML
  
  // Binary data tests
  { dataType: 'binary', dataSize: 10 * 1024, dataPattern: 'random' }, // 10KB random
  { dataType: 'binary', dataSize: 10 * 1024, dataPattern: 'sequential' }, // 10KB sequential
  { dataType: 'binary', dataSize: 10 * 1024, dataPattern: 'repeated' }, // 10KB repeated
  { dataType: 'binary', dataSize: 10 * 1024, dataPattern: 'sine' }, // 10KB sine wave
  { dataType: 'binary', dataSize: 100 * 1024, dataPattern: 'mixed' }, // 100KB mixed
  
  // Image-like data tests
  { dataType: 'image', dataSize: 100 * 1024 }, // ~100KB image (182x182 RGB)
  { dataType: 'image', dataSize: 500 * 1024 }  // ~500KB image (408x408 RGB)
];

// Run all benchmark tests
function runAllBenchmarks() {
  console.log("=================================================");
  console.log(" Prime Compression Algorithm - Benchmark Suite ");
  console.log("=================================================");
  
  const results = [];
  
  // Run each benchmark test
  for (const testCase of benchmarkTestCases) {
    try {
      const result = BenchmarkRunner.runBenchmark(
        testCase.dataType,
        testCase.dataSize,
        testCase.dataPattern
      );
      results.push(result);
    } catch (e) {
      console.error(`Error in benchmark: ${e.message}`);
    }
  }
  
  // Generate overall summary
  console.log("\n=================================================");
  console.log("                 Overall Summary                  ");
  console.log("=================================================");
  
  // Average compression ratios
  const avgRatios = {
    prime: 0,
    gzip: 0,
    brotli: 0,
    lzma: 0
  };
  
  // Count of successful tests per algorithm
  const successCounts = {
    prime: 0,
    gzip: 0,
    brotli: 0,
    lzma: 0
  };
  
  // Tally up results
  for (const result of results) {
    if (!result.prime.error) {
      avgRatios.prime += result.prime.ratio;
      successCounts.prime++;
    }
    
    if (!result.gzip.error) {
      avgRatios.gzip += result.gzip.ratio;
      successCounts.gzip++;
    }
    
    if (!result.brotli.error) {
      avgRatios.brotli += result.brotli.ratio;
      successCounts.brotli++;
    }
    
    if (!result.lzma.error) {
      avgRatios.lzma += result.lzma.ratio;
      successCounts.lzma++;
    }
  }
  
  // Calculate averages
  if (successCounts.prime > 0) avgRatios.prime /= successCounts.prime;
  if (successCounts.gzip > 0) avgRatios.gzip /= successCounts.gzip;
  if (successCounts.brotli > 0) avgRatios.brotli /= successCounts.brotli;
  if (successCounts.lzma > 0) avgRatios.lzma /= successCounts.lzma;
  
  // Print summary
  console.log("Average Compression Ratios:");
  if (successCounts.prime > 0) console.log(`Prime: ${avgRatios.prime.toFixed(2)}x (${successCounts.prime}/${results.length} tests)`);
  if (successCounts.gzip > 0) console.log(`GZIP: ${avgRatios.gzip.toFixed(2)}x (${successCounts.gzip}/${results.length} tests)`);
  if (successCounts.brotli > 0) console.log(`Brotli: ${avgRatios.brotli.toFixed(2)}x (${successCounts.brotli}/${results.length} tests)`);
  if (successCounts.lzma > 0) console.log(`LZMA: ${avgRatios.lzma.toFixed(2)}x (${successCounts.lzma}/${results.length} tests)`);
  
  // Prime-specific strategy analysis
  if (successCounts.prime > 0) {
    console.log("\nPrime Compression Strategy Analysis:");
    
    const strategyCounts = {};
    for (const result of results) {
      if (!result.prime.error && result.prime.strategy) {
        if (!strategyCounts[result.prime.strategy]) {
          strategyCounts[result.prime.strategy] = {
            count: 0,
            totalRatio: 0
          };
        }
        
        strategyCounts[result.prime.strategy].count++;
        strategyCounts[result.prime.strategy].totalRatio += result.prime.ratio;
      }
    }
    
    for (const [strategy, data] of Object.entries(strategyCounts)) {
      const avgRatio = data.totalRatio / data.count;
      console.log(`${strategy}: Used ${data.count} times, avg ratio ${avgRatio.toFixed(2)}x`);
    }
  }
  
  console.log("\n=================================================");
  console.log("           Benchmark Tests Completed              ");
  console.log("=================================================");
  
  return results;
}

// Execute benchmarks
runAllBenchmarks();