/**
 * Edge case tests for the Prime Compression algorithm
 * 
 * This test suite focuses on challenging edge cases and unusual data patterns
 * that might cause problems for compression algorithms. It validates the robustness
 * of the compression algorithm under extreme conditions.
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

// Function to calculate a checksum for data integrity validation
function calculateChecksum(data) {
  let hash = 0;
  
  // Handle different data types
  const buffer = data instanceof Uint8Array ? data : Array.from(data);
  
  for (let i = 0; i < buffer.length; i++) {
    const byte = buffer[i];
    // Using a simple FNV-1a-like algorithm
    hash = ((hash ^ byte) * 16777619) >>> 0;
  }
  
  // Convert to hex string for easier comparison
  return (hash >>> 0).toString(16).padStart(8, '0');
}

// Edge case data generators
const EdgeCaseGenerator = {
  // Tiny data (1-4 bytes)
  generateTinyData(size = 1) {
    const data = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = 42 + i;
    }
    return data;
  },
  
  // All same value
  generateConstantData(size = 1024, value = 123) {
    const data = new Uint8Array(size);
    data.fill(value);
    return data;
  },
  
  // Alternating pattern (0, 255, 0, 255, ...)
  generateAlternatingData(size = 1024) {
    const data = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = i % 2 === 0 ? 0 : 255;
    }
    return data;
  },
  
  // Counter with reset (0, 1, 2, ..., 255, 0, 1, ...)
  generateCounterWithReset(size = 1024) {
    const data = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = i % 256;
    }
    return data;
  },
  
  // Prime numbers sequence
  generatePrimeSequence(size = 1024) {
    const data = new Uint8Array(size);
    
    const isPrime = (n) => {
      if (n <= 1) return false;
      if (n <= 3) return true;
      if (n % 2 === 0 || n % 3 === 0) return false;
      
      for (let i = 5; i * i <= n; i += 6) {
        if (n % i === 0 || n % (i + 2) === 0) return false;
      }
      
      return true;
    };
    
    let primeCount = 0;
    let num = 2;
    
    while (primeCount < size) {
      if (isPrime(num)) {
        data[primeCount] = num % 256;
        primeCount++;
      }
      num++;
    }
    
    return data;
  },
  
  // Fibonacci sequence
  generateFibonacci(size = 1024) {
    const data = new Uint8Array(size);
    
    let a = 1, b = 1;
    data[0] = a % 256;
    
    for (let i = 1; i < size; i++) {
      data[i] = b % 256;
      const next = a + b;
      a = b;
      b = next;
    }
    
    return data;
  },
  
  // Random data with hidden pattern
  generateHiddenPattern(size = 1024, patternSize = 16, interval = 100) {
    const data = new Uint8Array(size);
    
    // Fill with random data
    for (let i = 0; i < size; i++) {
      data[i] = Math.floor(Math.random() * 256);
    }
    
    // Generate a pattern
    const pattern = new Uint8Array(patternSize);
    for (let i = 0; i < patternSize; i++) {
      pattern[i] = Math.floor(Math.random() * 256);
    }
    
    // Insert pattern at intervals
    for (let pos = 0; pos < size - patternSize; pos += interval) {
      for (let i = 0; i < patternSize; i++) {
        if (pos + i < size) {
          data[pos + i] = pattern[i];
        }
      }
    }
    
    return data;
  },
  
  // Nearly-sorted data with few outliers
  generateNearlySorted(size = 1024, outlierPercent = 5) {
    const data = new Uint8Array(size);
    
    // Fill with sorted data
    for (let i = 0; i < size; i++) {
      data[i] = i % 256;
    }
    
    // Add outliers
    const outlierCount = Math.floor(size * outlierPercent / 100);
    for (let i = 0; i < outlierCount; i++) {
      const pos = Math.floor(Math.random() * size);
      data[pos] = Math.floor(Math.random() * 256);
    }
    
    return data;
  },
  
  // Step function (plateaus)
  generateStepFunction(size = 1024, steps = 10) {
    const data = new Uint8Array(size);
    
    const stepSize = Math.floor(size / steps);
    
    for (let step = 0; step < steps; step++) {
      const value = Math.floor((step / steps) * 256);
      
      for (let i = 0; i < stepSize; i++) {
        const pos = step * stepSize + i;
        if (pos < size) {
          data[pos] = value;
        }
      }
    }
    
    return data;
  },
  
  // Exponential growth
  generateExponential(size = 1024) {
    const data = new Uint8Array(size);
    
    for (let i = 0; i < size; i++) {
      // Calculate exponential value, scaled to fit in byte range
      const normPos = i / size;
      let value = Math.pow(2, 8 * normPos) - 1;
      
      // Ensure it fits in a byte
      value = Math.min(255, Math.floor(value));
      data[i] = value;
    }
    
    return data;
  },
  
  // Sawtooth wave
  generateSawtooth(size = 1024, period = 64) {
    const data = new Uint8Array(size);
    
    for (let i = 0; i < size; i++) {
      data[i] = (i % period) * (256 / period); 
    }
    
    return data;
  },
  
  // Square wave
  generateSquareWave(size = 1024, period = 50, dutyCycle = 0.5) {
    const data = new Uint8Array(size);
    
    for (let i = 0; i < size; i++) {
      const pos = i % period;
      data[i] = pos < (period * dutyCycle) ? 255 : 0;
    }
    
    return data;
  },
  
  // Highly repetitive with small changes
  generateRepetitiveWithChanges(size = 1024, patternSize = 16, changeFrequency = 0.05) {
    const data = new Uint8Array(size);
    
    // Generate a base pattern
    const pattern = new Uint8Array(patternSize);
    for (let i = 0; i < patternSize; i++) {
      pattern[i] = Math.floor(Math.random() * 256);
    }
    
    // Fill the data with the repeating pattern
    for (let i = 0; i < size; i++) {
      data[i] = pattern[i % patternSize];
      
      // Occasionally change a value
      if (Math.random() < changeFrequency) {
        data[i] = (data[i] + 1) % 256;
      }
    }
    
    return data;
  },
  
  // Binary data (only 0s and 1s)
  generateBinaryValues(size = 1024) {
    const data = new Uint8Array(size);
    
    for (let i = 0; i < size; i++) {
      data[i] = Math.random() < 0.5 ? 0 : 1;
    }
    
    return data;
  },
  
  // Data with abrupt changes in pattern
  generateAbruptChanges(size = 1024, sections = 5) {
    const data = new Uint8Array(size);
    
    const sectionSize = Math.floor(size / sections);
    
    for (let section = 0; section < sections; section++) {
      const start = section * sectionSize;
      const end = Math.min(size, (section + 1) * sectionSize);
      
      // Different pattern for each section
      switch (section % 5) {
        case 0: // Random
          for (let i = start; i < end; i++) {
            data[i] = Math.floor(Math.random() * 256);
          }
          break;
          
        case 1: // Sequential
          for (let i = start; i < end; i++) {
            data[i] = (i - start) % 256;
          }
          break;
          
        case 2: // Constant
          for (let i = start; i < end; i++) {
            data[i] = section * 40;
          }
          break;
          
        case 3: // Sine wave
          for (let i = start; i < end; i++) {
            const phase = (i - start) / (end - start) * 2 * Math.PI;
            data[i] = Math.floor(128 + 127 * Math.sin(phase * 5));
          }
          break;
          
        case 4: // Alternating
          for (let i = start; i < end; i++) {
            data[i] = (i - start) % 2 === 0 ? 0 : 255;
          }
          break;
      }
    }
    
    return data;
  },
  
  // Data with very long run length
  generateLongRuns(size = 1024, runCount = 5) {
    const data = new Uint8Array(size);
    
    // Calculate run length
    const avgRunLength = Math.floor(size / runCount);
    
    let pos = 0;
    for (let run = 0; run < runCount && pos < size; run++) {
      // Vary run length a bit
      const runLength = Math.floor(avgRunLength * (0.8 + Math.random() * 0.4));
      
      // Random value for this run
      const value = Math.floor(Math.random() * 256);
      
      // Fill the run
      for (let i = 0; i < runLength && pos < size; i++) {
        data[pos++] = value;
      }
    }
    
    return data;
  }
};

// Test runner
function runEdgeCaseTest(name, dataGenerator, size = 1024) {
  console.log(`\n=== Test: ${name} ===`);
  
  try {
    // Generate test data
    const data = dataGenerator(size);
    const originalChecksum = calculateChecksum(data);
    
    console.log(`Generated ${data.length} bytes of test data`);
    
    // Compress the data
    console.log(`Compressing...`);
    const startTime = performance.now();
    const compressed = compression.compress(data);
    const compressionTime = performance.now() - startTime;
    
    // Print compression stats
    console.log(`Compressed size: ${compressed.compressedSize || compressed.compressedVector.length} bytes`);
    console.log(`Compression ratio: ${compressed.compressionRatio.toFixed(2)}x`);
    console.log(`Compression strategy: ${compressed.strategy || compressed.compressionMethod || "unknown"}`);
    console.log(`Compression time: ${compressionTime.toFixed(2)} ms`);
    
    // Decompress the data
    console.log(`Decompressing...`);
    const startDecompTime = performance.now();
    const decompressed = compression.decompress(compressed);
    const decompressionTime = performance.now() - startDecompTime;
    
    console.log(`Decompression time: ${decompressionTime.toFixed(2)} ms`);
    
    // Verify integrity
    const decompressedChecksum = calculateChecksum(decompressed);
    
    if (data.length !== decompressed.length) {
      console.error(`✗ Size mismatch: original=${data.length}, decompressed=${decompressed.length}`);
      return false;
    }
    
    if (originalChecksum !== decompressedChecksum) {
      console.error(`✗ Checksum mismatch: original=${originalChecksum}, decompressed=${decompressedChecksum}`);
      
      // Find the first few mismatches
      let mismatchCount = 0;
      const MAX_MISMATCHES_TO_REPORT = 5;
      
      for (let i = 0; i < data.length; i++) {
        if (data[i] !== decompressed[i]) {
          console.error(`Mismatch at byte ${i}: original=${data[i]}, decompressed=${decompressed[i]}`);
          mismatchCount++;
          
          if (mismatchCount >= MAX_MISMATCHES_TO_REPORT) {
            console.error(`... more mismatches omitted`);
            break;
          }
        }
      }
      
      return false;
    }
    
    console.log(`✓ Decompressed data matches original data (100% integrity verified)`);
    return true;
  } catch (e) {
    console.error(`✗ Test failed with error: ${e.message}`);
    return false;
  }
}

// Run all edge case tests
function runAllEdgeCaseTests() {
  console.log("=================================================");
  console.log(" Prime Compression - Edge Case Test Suite ");
  console.log("=================================================");
  
  const testCases = [
    // Tiny data tests
    { name: "1 Byte Data", generator: EdgeCaseGenerator.generateTinyData, size: 1 },
    { name: "2 Byte Data", generator: EdgeCaseGenerator.generateTinyData, size: 2 },
    { name: "3 Byte Data", generator: EdgeCaseGenerator.generateTinyData, size: 3 },
    { name: "4 Byte Data", generator: EdgeCaseGenerator.generateTinyData, size: 4 },
    
    // Simple pattern tests
    { name: "All Zeros", generator: EdgeCaseGenerator.generateConstantData, args: [1024, 0] },
    { name: "All Same Value (123)", generator: EdgeCaseGenerator.generateConstantData, args: [1024, 123] },
    { name: "All Same Value (255)", generator: EdgeCaseGenerator.generateConstantData, args: [1024, 255] },
    { name: "Alternating Values (0, 255)", generator: EdgeCaseGenerator.generateAlternatingData },
    { name: "Counter with Reset", generator: EdgeCaseGenerator.generateCounterWithReset },
    
    // Mathematical sequences
    { name: "Prime Number Sequence", generator: EdgeCaseGenerator.generatePrimeSequence },
    { name: "Fibonacci Sequence", generator: EdgeCaseGenerator.generateFibonacci },
    { name: "Exponential Growth", generator: EdgeCaseGenerator.generateExponential },
    
    // Wave patterns
    { name: "Sawtooth Wave", generator: EdgeCaseGenerator.generateSawtooth },
    { name: "Square Wave", generator: EdgeCaseGenerator.generateSquareWave },
    
    // Mixed patterns
    { name: "Hidden Pattern in Random Data", generator: EdgeCaseGenerator.generateHiddenPattern },
    { name: "Nearly Sorted with Outliers", generator: EdgeCaseGenerator.generateNearlySorted },
    { name: "Step Function", generator: EdgeCaseGenerator.generateStepFunction },
    { name: "Repetitive with Small Changes", generator: EdgeCaseGenerator.generateRepetitiveWithChanges },
    { name: "Binary Values Only", generator: EdgeCaseGenerator.generateBinaryValues },
    { name: "Abrupt Pattern Changes", generator: EdgeCaseGenerator.generateAbruptChanges },
    { name: "Long Run Lengths", generator: EdgeCaseGenerator.generateLongRuns }
  ];
  
  const results = [];
  
  for (const testCase of testCases) {
    const args = testCase.args || [testCase.size || 1024];
    const generator = (...generatorArgs) => testCase.generator(...generatorArgs);
    const result = runEdgeCaseTest(testCase.name, generator, ...args);
    
    results.push({
      name: testCase.name,
      passed: result
    });
  }
  
  // Print summary
  console.log("\n=================================================");
  console.log("                 Test Summary                    ");
  console.log("=================================================");
  
  let passed = 0;
  for (const result of results) {
    console.log(`${result.passed ? '✓' : '✗'} ${result.name}`);
    if (result.passed) passed++;
  }
  
  console.log(`\nPassed: ${passed}/${results.length} (${(passed / results.length * 100).toFixed(2)}%)`);
  console.log("=================================================");
  
  return passed === results.length;
}

// Run all the tests
runAllEdgeCaseTests();