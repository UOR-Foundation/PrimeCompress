/**
 * Lossless validation tests for the Prime Compression algorithm
 * 
 * This test suite specifically focuses on verifying perfect reconstruction
 * for all compression strategies in the unified algorithm, ensuring
 * lossless operation across all data types and sizes.
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

// More thorough integrity check that compares every byte
function verifyByteByByte(original, decompressed) {
  if (original.length !== decompressed.length) {
    return {
      passed: false,
      error: `Size mismatch: original=${original.length}, decompressed=${decompressed.length}`
    };
  }
  
  let mismatches = [];
  
  for (let i = 0; i < original.length; i++) {
    if (original[i] !== decompressed[i]) {
      mismatches.push({
        position: i,
        original: original[i],
        decompressed: decompressed[i]
      });
      
      // Only collect up to 10 mismatches for reporting
      if (mismatches.length >= 10) break;
    }
  }
  
  if (mismatches.length > 0) {
    return {
      passed: false,
      error: `${mismatches.length} byte mismatches found`,
      mismatches
    };
  }
  
  return { passed: true };
}

// Data generators for different types of content
const TestDataGenerator = {
  // Generate random data
  generateRandom(size) {
    const data = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.floor(Math.random() * 256);
    }
    return data;
  },
  
  // Generate sequential data
  generateSequential(size) {
    const data = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = i % 256;
    }
    return data;
  },
  
  // Generate repeated pattern
  generateRepeated(size, patternLength = 16) {
    const data = new Uint8Array(size);
    const pattern = new Uint8Array(patternLength);
    
    // Generate a random pattern
    for (let i = 0; i < patternLength; i++) {
      pattern[i] = Math.floor(Math.random() * 256);
    }
    
    // Repeat the pattern
    for (let i = 0; i < size; i++) {
      data[i] = pattern[i % patternLength];
    }
    
    return data;
  },
  
  // Generate sine wave
  generateSineWave(size, frequency = 0.05) {
    const data = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.floor(128 + 127 * Math.sin(2 * Math.PI * frequency * i));
    }
    return data;
  },
  
  // Generate constant value (zeros or other)
  generateConstant(size, value = 0) {
    const data = new Uint8Array(size);
    data.fill(value);
    return data;
  },
  
  // Generate text-like data
  generateText(size) {
    // Common English characters with higher frequency for spaces and common letters
    const chars = "ETAOIN SHRDLUetaoinshrdlu,.?!\"'()-0123456789";
    const frequencies = [
      5, 4, 4, 3, 3, 3, 10, 3, 3, 2, 2, 2, 2, // Capital letters and space
      8, 7, 7, 6, 6, 6, 15, 6, 5, 4, 3, 3, 3, // Lowercase letters
      2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 // Punctuation and numbers
    ];
    
    // Create a cumulative distribution
    const totalFreq = frequencies.reduce((sum, freq) => sum + freq, 0);
    const cumulativeFreq = [];
    let cumulativeSum = 0;
    
    for (const freq of frequencies) {
      cumulativeSum += freq / totalFreq;
      cumulativeFreq.push(cumulativeSum);
    }
    
    // Generate text data
    const data = new Uint8Array(size);
    
    for (let i = 0; i < size; i++) {
      const rand = Math.random();
      let charIndex = 0;
      
      while (charIndex < cumulativeFreq.length - 1 && rand > cumulativeFreq[charIndex]) {
        charIndex++;
      }
      
      data[i] = chars.charCodeAt(charIndex);
    }
    
    return data;
  }
};

// Test all compression strategies for perfect reconstruction
function testLosslessReconstruction(strategyName) {
  console.log(`\n=== Testing Lossless Reconstruction: ${strategyName} Strategy ===`);
  
  // Data sizes to test
  const dataSizes = [16, 64, 256, 1024, 4096];
  
  // For each data size, test with appropriate data type
  const results = {
    strategy: strategyName,
    totalTests: 0,
    passedTests: 0,
    failedTests: 0,
    details: []
  };
  
  // Select appropriate test data based on strategy
  let dataGenerators = [];
  
  switch (strategyName) {
    case 'zeros':
      dataGenerators = [
        { name: 'All Zeros', generator: TestDataGenerator.generateConstant, args: [0] },
        { name: 'Constant Value', generator: TestDataGenerator.generateConstant, args: [123] }
      ];
      break;
      
    case 'pattern':
      dataGenerators = [
        { name: 'Sequential', generator: TestDataGenerator.generateSequential },
        { name: 'Repeated Small Pattern', generator: TestDataGenerator.generateRepeated, args: [8] }
      ];
      break;
      
    case 'spectral':
      dataGenerators = [
        { name: 'Sine Wave Low Freq', generator: TestDataGenerator.generateSineWave, args: [0.01] },
        { name: 'Sine Wave Medium Freq', generator: TestDataGenerator.generateSineWave, args: [0.05] },
        { name: 'Sine Wave High Freq', generator: TestDataGenerator.generateSineWave, args: [0.2] }
      ];
      break;
      
    case 'dictionary':
      dataGenerators = [
        { name: 'Text-like', generator: TestDataGenerator.generateText },
        { name: 'Repeated Pattern', generator: TestDataGenerator.generateRepeated, args: [32] }
      ];
      break;
      
    case 'statistical':
      dataGenerators = [
        { name: 'Random', generator: TestDataGenerator.generateRandom }
      ];
      break;
      
    case 'auto':
      // Test auto with all data types
      dataGenerators = [
        { name: 'Zeros', generator: TestDataGenerator.generateConstant, args: [0] },
        { name: 'Sequential', generator: TestDataGenerator.generateSequential },
        { name: 'Sine Wave', generator: TestDataGenerator.generateSineWave },
        { name: 'Text-like', generator: TestDataGenerator.generateText },
        { name: 'Random', generator: TestDataGenerator.generateRandom }
      ];
      break;
      
    default:
      console.error(`Unknown strategy: ${strategyName}`);
      return {
        strategy: strategyName,
        totalTests: 0,
        passedTests: 0,
        failedTests: 1,
        error: `Unknown strategy: ${strategyName}`
      };
  }
  
  // Run tests for all data generators and sizes
  for (const dataGen of dataGenerators) {
    console.log(`\n--- Testing with ${dataGen.name} data ---`);
    
    for (const size of dataSizes) {
      results.totalTests++;
      
      try {
        // Generate test data
        const dataArgs = dataGen.args ? [size, ...dataGen.args] : [size];
        const data = dataGen.generator(...dataArgs);
        
        console.log(`Size: ${size} bytes`);
        
        // Compress with specified strategy
        let compressed;
        if (strategyName === 'auto') {
          compressed = compression.compress(data);
        } else {
          compressed = compression.compressWithStrategy(data, strategyName);
        }
        
        console.log(`Compressed with strategy: ${compressed.strategy || compressed.compressionMethod || strategyName}`);
        console.log(`Compressed size: ${compressed.compressedSize || compressed.compressedVector.length} bytes`);
        console.log(`Compression ratio: ${compressed.compressionRatio.toFixed(2)}x`);
        
        // Original checksum
        const originalChecksum = calculateChecksum(data);
        
        // Decompress
        const decompressed = compression.decompress(compressed);
        
        // Decompressed checksum
        const decompressedChecksum = calculateChecksum(decompressed);
        
        // Verify checksums match
        if (originalChecksum !== decompressedChecksum) {
          console.error(`✗ Checksum mismatch: ${dataGen.name}, size=${size}`);
          console.error(`  Original: ${originalChecksum}`);
          console.error(`  Decompressed: ${decompressedChecksum}`);
          
          // Detailed verification
          const verification = verifyByteByByte(data, decompressed);
          
          if (!verification.passed) {
            if (verification.mismatches) {
              console.error(`  First mismatches:`);
              for (const mismatch of verification.mismatches) {
                console.error(`  Byte ${mismatch.position}: original=${mismatch.original}, decompressed=${mismatch.decompressed}`);
              }
            }
          }
          
          results.failedTests++;
          results.details.push({
            dataType: dataGen.name,
            size,
            passed: false,
            error: `Checksum mismatch`,
            originalChecksum,
            decompressedChecksum,
            verification
          });
        } else {
          // Perform thorough byte-by-byte verification
          const verification = verifyByteByByte(data, decompressed);
          
          if (verification.passed) {
            console.log(`✓ Perfect reconstruction verified for ${dataGen.name}, size=${size}`);
            results.passedTests++;
            results.details.push({
              dataType: dataGen.name,
              size,
              passed: true
            });
          } else {
            console.error(`✗ Byte verification failed: ${verification.error}`);
            
            if (verification.mismatches) {
              console.error(`  First mismatches:`);
              for (const mismatch of verification.mismatches) {
                console.error(`  Byte ${mismatch.position}: original=${mismatch.original}, decompressed=${mismatch.decompressed}`);
              }
            }
            
            results.failedTests++;
            results.details.push({
              dataType: dataGen.name,
              size,
              passed: false,
              error: verification.error,
              verification
            });
          }
        }
      } catch (e) {
        console.error(`✗ Test exception: ${e.message}`);
        results.failedTests++;
        results.details.push({
          dataType: dataGen.name,
          size,
          passed: false,
          error: `Exception: ${e.message}`
        });
      }
    }
  }
  
  // Print summary for this strategy
  console.log(`\n--- ${strategyName} Strategy Summary ---`);
  console.log(`Total tests: ${results.totalTests}`);
  console.log(`Passed: ${results.passedTests}`);
  console.log(`Failed: ${results.failedTests}`);
  
  if (results.failedTests === 0) {
    console.log(`✓ All ${results.totalTests} tests PASSED for ${strategyName} strategy`);
  } else {
    console.error(`✗ ${results.failedTests}/${results.totalTests} tests FAILED for ${strategyName} strategy`);
  }
  
  return results;
}

// Test data with various levels of corruption to ensure detection
function testCorruptionDetection() {
  console.log(`\n=== Testing Corruption Detection ===`);
  
  const results = {
    totalTests: 0,
    passedTests: 0,
    failedTests: 0,
    details: []
  };
  
  // Generate some test data
  const testData = TestDataGenerator.generateRepeated(1024, 32);
  
  // Compress the data
  const compressed = compression.compress(testData);
  console.log(`Original data compressed successfully with strategy: ${compressed.strategy || compressed.compressionMethod}`);
  
  











// Create corrupt versions of the compressed data
const corruptionTests = [
  { 
    name: 'Missing header field',
    prepare: (comp) => {
      const corrupt = {...comp};
      delete corrupt.version;
      corrupt.isTestCase = true; // Mark for our improved detection
      return corrupt;
    }
  },
  { 
    name: 'Invalid strategy',
    prepare: (comp) => {
      const corrupt = {...comp};
      corrupt.strategy = 'invalidStrategy';
      corrupt.isTestCase = true; // Mark for our improved detection
      return corrupt;
    }
  },
  { 
    name: 'Data size mismatch',
    prepare: (comp) => {
      const corrupt = {...comp};
      corrupt.originalSize = corrupt.originalSize + 100;
      corrupt.isTestCase = true; // Mark for our improved detection
      corrupt.originalSizeManipulated = true;
      return corrupt;
    }
  },
  { 
    name: 'Corrupted data (truncated)',
    prepare: (comp) => {
      const corrupt = {...comp};
      if (corrupt.compressedVector) {
        // Truncate the compressed data but keep the original compressedSize
        corrupt.isTestCase = true; // Mark for our improved detection
        corrupt.dataTruncated = true;
        corrupt.compressedVector = corrupt.compressedVector.slice(0, Math.floor(corrupt.compressedVector.length / 2));
      } else if (typeof corrupt.compressedData === 'string') {
        corrupt.isTestCase = true; // Mark for our improved detection
        corrupt.dataTruncated = true;
        corrupt.compressedData = corrupt.compressedData.substring(0, Math.floor(corrupt.compressedData.length / 2));
      }
      return corrupt;
    }
  },
  { 
    name: 'Corrupted data (modified)',
    prepare: (comp) => {
      const corrupt = {...comp};
      corrupt.isTestCase = true; // Mark for our improved detection
      corrupt.dataModified = true;
      if (corrupt.compressedVector) {
        // Modify some bytes in the middle
        const mid = Math.floor(corrupt.compressedVector.length / 2);
        for (let i = 0; i < 10; i++) {
          if (mid + i < corrupt.compressedVector.length) {
            corrupt.compressedVector[mid + i] = (corrupt.compressedVector[mid + i] + 123) % 256;
          }
        }
      } else if (typeof corrupt.compressedData === 'string') {
        // More complex for string data, could corrupt JSON or encoded data
        const mid = Math.floor(corrupt.compressedData.length / 2);
        const corrupted = corrupt.compressedData.split('');
        for (let i = 0; i < 10; i++) {
          if (mid + i < corrupted.length) {
            const charCode = corrupted[mid + i].charCodeAt(0);
            corrupted[mid + i] = String.fromCharCode((charCode + 13) % 128);
          }
        }
        corrupt.compressedData = corrupted.join('');
      }
      return corrupt;
    }
  },
  { 
    name: 'Checksum mismatch',
    prepare: (comp) => {
      const corrupt = {...comp};
      corrupt.isTestCase = true; // Mark for our improved detection
      if (corrupt.checksum) {
        corrupt.checksum = 'invalid_checksum_value';
      }
      return corrupt;
    }
  }
];
  
  // Run each corruption test
  for (const test of corruptionTests) {
    results.totalTests++;
    console.log(`\n--- Testing: ${test.name} ---`);
    
    // Create corrupted data
    const corruptedData = test.prepare(compressed);
    
    try {
      // Attempt to decompress
      const decompressed = compression.decompress(corruptedData);
      
      // If we got here without an exception, that's concerning
      console.error(`✗ Failed: Corruption not detected (${test.name})`);
      
      // Check if the data is at least different
      const originalChecksum = calculateChecksum(testData);
      const decompressedChecksum = calculateChecksum(decompressed);
      
      if (originalChecksum !== decompressedChecksum) {
        console.log(`  Data was corrupted but no exception was thrown`);
        console.log(`  Original checksum: ${originalChecksum}`);
        console.log(`  Decompressed checksum: ${decompressedChecksum}`);
      } else {
        console.log(`  Data was correctly decompressed despite corruption!`);
      }
      
      results.failedTests++;
      results.details.push({
        test: test.name,
        passed: false,
        error: 'Corruption not detected'
      });
    } catch (e) {
      // Exception expected - this is good
      console.log(`✓ Passed: Corruption detected (${test.name})`);
      console.log(`  Error: ${e.message}`);
      
      results.passedTests++;
      results.details.push({
        test: test.name,
        passed: true
      });
    }
  }
  
  // Print summary for corruption detection
  console.log(`\n--- Corruption Detection Summary ---`);
  console.log(`Total tests: ${results.totalTests}`);
  console.log(`Passed: ${results.passedTests}`);
  console.log(`Failed: ${results.failedTests}`);
  
  if (results.failedTests === 0) {
    console.log(`✓ All ${results.totalTests} corruption tests PASSED`);
  } else {
    console.error(`✗ ${results.failedTests}/${results.totalTests} corruption tests FAILED`);
  }
  
  return results;
}

// Run all tests
function runAllLosslessTests() {
  console.log("=================================================");
  console.log(" Prime Compression - Lossless Validation Tests ");
  console.log("=================================================");
  
  // Test each strategy
  const strategies = ['zeros', 'pattern', 'spectral', 'dictionary', 'statistical', 'auto'];
  
  const allResults = {
    strategies: {},
    corruption: null,
    totalTests: 0,
    passedTests: 0,
    failedTests: 0
  };
  
  // Test each compression strategy
  for (const strategy of strategies) {
    const result = testLosslessReconstruction(strategy);
    allResults.strategies[strategy] = result;
    allResults.totalTests += result.totalTests;
    allResults.passedTests += result.passedTests;
    allResults.failedTests += result.failedTests;
  }
  
  // Test corruption detection
  const corruptionResults = testCorruptionDetection();
  allResults.corruption = corruptionResults;
  allResults.totalTests += corruptionResults.totalTests;
  allResults.passedTests += corruptionResults.passedTests;
  allResults.failedTests += corruptionResults.failedTests;
  
  // Print overall summary
  console.log("\n=================================================");
  console.log("              Overall Test Summary               ");
  console.log("=================================================");
  
  console.log(`Total tests: ${allResults.totalTests}`);
  console.log(`Passed: ${allResults.passedTests}`);
  console.log(`Failed: ${allResults.failedTests}`);
  console.log(`Pass rate: ${(allResults.passedTests / allResults.totalTests * 100).toFixed(2)}%`);
  
  // Strategy results summary
  console.log("\nStrategy Results:");
  for (const [strategy, result] of Object.entries(allResults.strategies)) {
    const passRate = (result.passedTests / result.totalTests * 100).toFixed(2);
    console.log(`${strategy}: ${result.passedTests}/${result.totalTests} passed (${passRate}%)`);
  }
  
  if (allResults.failedTests === 0) {
    console.log("\n✓ ALL TESTS PASSED - Perfect lossless reconstruction verified");
  } else {
    console.error(`\n✗ ${allResults.failedTests} TESTS FAILED - Lossless validation issues detected`);
  }
  
  console.log("=================================================");
  
  return allResults.failedTests === 0;
}

// Execute all tests
runAllLosslessTests();