/**
 * Unit tests for the Prime Compression module
 */

// Import the compression module directly
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

// Test data generation utility
function generateTestData(size, pattern = 'random') {
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
      
    default:
      throw new Error(`Unknown test data pattern: ${pattern}`);
  }
  
  return data;
}

// Helper function to verify compression/decompression
function verifyCompression(data, expectedCompressionRatio = null) {
  // Compress the data
  console.log(`Compressing ${data.length} bytes...`);
  const startTime = performance.now();
  const compressed = compression.compress(data);
  const compressionTime = performance.now() - startTime;
  
  // Print compression statistics
  console.log(`Compressed size: ${compressed.compressedVector ? compressed.compressedVector.length : compressed.compressedSize} bytes`);
  console.log(`Compression ratio: ${compressed.compressionRatio.toFixed(2)}x`);
  console.log(`Terminating base: ${compressed.terminatingBase || "N/A"}`);
  console.log(`Compression time: ${compressionTime.toFixed(2)} ms`);
  
  // Verify expected compression ratio if provided
  if (expectedCompressionRatio !== null) {
    if (compressed.compressionRatio < expectedCompressionRatio) {
      console.warn(`Warning: Compression ratio (${compressed.compressionRatio.toFixed(2)}) is below expected (${expectedCompressionRatio})`);
    } else {
      console.log(`✓ Compression ratio meets or exceeds expected value`);
    }
  }
  
  // Decompress the data
  console.log(`Decompressing...`);
  const decompStartTime = performance.now();
  const decompressed = compression.decompress(compressed);
  const decompressionTime = performance.now() - decompStartTime;
  
  console.log(`Decompression time: ${decompressionTime.toFixed(2)} ms`);
  
  // Verify the data is correctly decompressed with multiple integrity checks
  if (data.length !== decompressed.length) {
    throw new Error(`Decompressed length (${decompressed.length}) doesn't match original (${data.length})`);
  }
  
  // Full byte-by-byte comparison
  let matches = true;
  let mismatchCount = 0;
  const MAX_MISMATCHES_TO_REPORT = 3;
  
  for (let i = 0; i < data.length; i++) {
    if (data[i] !== decompressed[i]) {
      matches = false;
      mismatchCount++;
      
      if (mismatchCount <= MAX_MISMATCHES_TO_REPORT) {
        console.error(`Mismatch at byte ${i}: original=${data[i]}, decompressed=${decompressed[i]}`);
      }
      
      // Stop after finding several mismatches to avoid flooding the console
      if (mismatchCount > 10) {
        console.error(`... (${mismatchCount - MAX_MISMATCHES_TO_REPORT} more mismatches)`);
        break;
      }
    }
  }
  
  // Calculate checksums for both original and decompressed data
  const originalChecksum = calculateChecksum(data);
  const decompressedChecksum = calculateChecksum(decompressed);
  
  const checksumMatch = originalChecksum === decompressedChecksum;
  
  if (!checksumMatch) {
    console.error(`Checksum mismatch: original=${originalChecksum}, decompressed=${decompressedChecksum}`);
  } else {
    console.log(`✓ Checksums match: ${originalChecksum}`);
  }
  
  if (matches && checksumMatch) {
    console.log(`✓ Decompressed data matches original data (100% integrity verified)`);
    return true;
  } else {
    const matchPercentage = ((data.length - mismatchCount) / data.length * 100).toFixed(2);
    console.error(`✗ Decompression integrity check failed (${matchPercentage}% bytes matched)`);
    return false;
  }
}

// Test basic compression functionality
function testBasicCompression() {
  console.log("\n=== Test Basic Compression ===");
  
  // Small random data test
  const data = generateTestData(1024, 'random');
  
  // Should be able to compress and decompress without errors
  return verifyCompression(data);
}

// Test different data patterns
function testDataPatterns() {
  console.log("\n=== Test Different Data Patterns ===");
  let allPassed = true;
  
  const testCases = [
    { pattern: 'zeros', size: 1024, expectedRatio: 50 },
    { pattern: 'repeated', size: 1024, expectedRatio: 10 },
    { pattern: 'sequential', size: 1024, expectedRatio: 5 },
    { pattern: 'random', size: 1024, expectedRatio: 1 }
  ];
  
  for (const testCase of testCases) {
    console.log(`\nTesting ${testCase.pattern} data pattern:`);
    const data = generateTestData(testCase.size, testCase.pattern);
    const passed = verifyCompression(data, testCase.expectedRatio);
    allPassed = allPassed && passed;
  }
  
  return allPassed;
}

// Test varying data sizes
function testDataSizes() {
  console.log("\n=== Test Varying Data Sizes ===");
  let allPassed = true;
  
  const sizes = [16, 64, 256, 1024];
  
  for (const size of sizes) {
    console.log(`\nTesting data size ${size} bytes:`);
    // Use repeated pattern for predictable compression
    const data = generateTestData(size, 'repeated');
    const passed = verifyCompression(data);
    allPassed = allPassed && passed;
  }
  
  return allPassed;
}

// Test compression analysis
function testCompressionAnalysis() {
  console.log("\n=== Test Compression Analysis ===");
  
  // Test different data patterns for analysis
  const patterns = ['zeros', 'repeated', 'sequential', 'random'];
  
  for (const pattern of patterns) {
    console.log(`\nAnalyzing ${pattern} data:`);
    const data = generateTestData(1024, pattern);
    
    const analysis = compression.analyzeCompression(data);
    
    console.log(`Entropy: ${analysis.entropy.toFixed(4)}`);
    console.log(`Pattern score: ${analysis.patternScore.toFixed(4)}`);
    console.log(`Estimated terminating base: ${analysis.estimatedTerminatingBase}`);
    console.log(`Theoretical compression ratio: ${analysis.theoreticalCompressionRatio.toFixed(2)}x`);
    
    // Verify that high pattern data has lower entropy
    if (pattern === 'zeros' || pattern === 'repeated') {
      if (analysis.entropy > 0.5) {
        console.warn(`Warning: Entropy for ${pattern} data is higher than expected: ${analysis.entropy.toFixed(4)}`);
      } else {
        console.log(`✓ Entropy is appropriately low for ${pattern} data`);
      }
    }
    
    // Verify that compressibility is correctly identified
    if (pattern === 'random') {
      if (analysis.isCompressible) {
        console.warn(`Warning: Random data incorrectly identified as compressible`);
      } else {
        console.log(`✓ Random data correctly identified as less compressible`);
      }
    } else if (pattern === 'zeros') {
      if (!analysis.isCompressible) {
        console.warn(`Warning: Zero data incorrectly identified as not compressible`);
      } else {
        console.log(`✓ Zero data correctly identified as highly compressible`);
      }
    }
  }
  
  return true; // Analysis tests are informative, not pass/fail
}

// Test edge cases
function testEdgeCases() {
  console.log("\n=== Test Edge Cases ===");
  let allPassed = true;
  
  // Empty data
  try {
    compression.compress(new Uint8Array(0));
    console.error("✗ Should have rejected empty data");
    allPassed = false;
  } catch (e) {
    console.log(`✓ Correctly rejected empty data: ${e.message}`);
  }
  
  // Very small data (1 byte)
  try {
    const tinyData = new Uint8Array([42]);
    const compressed = compression.compress(tinyData);
    const decompressed = compression.decompress(compressed);
    
    if (decompressed.length === 1 && decompressed[0] === 42) {
      console.log(`✓ Correctly handled 1-byte data`);
    } else {
      console.error(`✗ Failed to correctly handle 1-byte data`);
      allPassed = false;
    }
  } catch (e) {
    console.error(`✗ Failed on 1-byte data: ${e.message}`);
    allPassed = false;
  }
  
  // Invalid compressed data
  try {
    compression.decompress({});
    console.error("✗ Should have rejected invalid compressed data");
    allPassed = false;
  } catch (e) {
    console.log(`✓ Correctly rejected invalid compressed data: ${e.message}`);
  }
  
  return allPassed;
}

// Helper function to format byte sizes
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// Performance benchmark (simplified for speed)
function benchmarkCompression() {
  console.log("\n=== Compression Performance Benchmark ===");
  
  const dataSizes = [1024, 10240]; // 1KB, 10KB
  const patterns = ['repeated', 'random'];
  
  for (const pattern of patterns) {
    console.log(`\nBenchmarking ${pattern} data:`);
    
    for (const size of dataSizes) {
      const data = generateTestData(size, pattern);
      
      console.log(`\nData size: ${size} bytes`);
      
      // Compression benchmark
      const startCompress = performance.now();
      const compressed = compression.compress(data);
      const compressTime = performance.now() - startCompress;
      
      console.log(`Compression time: ${compressTime.toFixed(2)} ms`);
      console.log(`Compression throughput: ${((size / 1024) / (compressTime / 1000)).toFixed(2)} KB/s`);
      console.log(`Compression ratio: ${compressed.compressionRatio.toFixed(2)}x`);
      
      // Decompression benchmark
      const startDecompress = performance.now();
      compression.decompress(compressed);
      const decompressTime = performance.now() - startDecompress;
      
      console.log(`Decompression time: ${decompressTime.toFixed(2)} ms`);
      console.log(`Decompression throughput: ${((size / 1024) / (decompressTime / 1000)).toFixed(2)} KB/s`);
    }
  }
  
  return true; // Benchmark is informative, not pass/fail
}

// Run all tests
function runAllTests() {
  console.log("======================================");
  console.log("   Prime Compression Module Tests");
  console.log("======================================");
  
  const results = [
    { name: "Basic Compression", passed: testBasicCompression() },
    { name: "Data Patterns", passed: testDataPatterns() },
    { name: "Data Sizes", passed: testDataSizes() },
    { name: "Compression Analysis", passed: testCompressionAnalysis() },
    { name: "Edge Cases", passed: testEdgeCases() },
    { name: "Performance Benchmark", passed: benchmarkCompression() }
  ];
  
  console.log("\n======================================");
  console.log("            Test Results");
  console.log("======================================");
  
  let allPassed = true;
  for (const result of results) {
    console.log(`${result.passed ? '✓' : '✗'} ${result.name}`);
    allPassed = allPassed && result.passed;
  }
  
  console.log("\n======================================");
  console.log(`Overall Test Result: ${allPassed ? 'PASSED' : 'FAILED'}`);
  console.log("======================================");
  
  return allPassed;
}

// Execute tests
runAllTests();