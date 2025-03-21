/**
 * Robustness tests for the Prime Compression algorithm
 * This tests the algorithm's behavior with edge cases and unusual inputs
 */

// Import the compression module
const compression = require('../prime-compression.js');

// Test robustness with various edge cases
function testRobustness() {
  console.log("======================================");
  console.log("  Prime Compression Robustness Test");
  console.log("======================================");
  
  const tests = [
    { name: "Empty array", data: new Uint8Array(0), shouldThrow: true },
    { name: "Null input", data: null, shouldThrow: true },
    { name: "Very small (1 byte)", data: new Uint8Array([42]), shouldThrow: false },
    { name: "Very small (2 bytes)", data: new Uint8Array([0, 255]), shouldThrow: false },
    { name: "All same value", data: new Uint8Array(100).fill(123), shouldThrow: false },
    { name: "Alternating binary", generateAlternating(100, 0, 1), shouldThrow: false },
    { name: "Max values", new Uint8Array(100).fill(255), shouldThrow: false },
    { name: "Min values", new Uint8Array(100).fill(0), shouldThrow: false },
    { name: "Prime numbers", generatePrimes(100), shouldThrow: false },
    { name: "Fibonacci sequence", generateFibonacci(100), shouldThrow: false },
    { name: "Exponential growth", generateExponential(100), shouldThrow: false },
    { name: "Random with repeats", generateRandomWithRepeats(100), shouldThrow: false },
    { name: "Special bytecodes", generateSpecialBytes(100), shouldThrow: false }
  ];
  
  for (const test of tests) {
    try {
      console.log(`\n--- Testing: ${test.name} ---`);
      if (test.shouldThrow) {
        try {
          compression.compress(test.data);
          console.log('❌ FAILED: Expected compression to throw an error, but it did not');
        } catch (error) {
          console.log(`✅ PASSED: Correctly threw: ${error.message}`);
        }
        continue;
      }
      
      // If not expected to throw, test the full compression/decompression cycle
      const compressed = compression.compress(test.data);
      console.log(`Compression ratio: ${compressed.compressionRatio.toFixed(2)}x`);
      console.log(`Compression method: ${compressed.specialCase || 'standard'}`);
      
      // Decompress
      const decompressed = compression.decompress(compressed);
      
      // Verify
      let matches = true;
      if (decompressed.length !== test.data.length) {
        console.log(`❌ FAILED: Length mismatch - original: ${test.data.length}, decompressed: ${decompressed.length}`);
        matches = false;
      } else {
        for (let i = 0; i < test.data.length; i++) {
          if (test.data[i] !== decompressed[i]) {
            console.log(`❌ FAILED: Mismatch at position ${i} - original: ${test.data[i]}, decompressed: ${decompressed[i]}`);
            matches = false;
            break;
          }
        }
      }
      
      if (matches) {
        console.log(`✅ PASSED: Successfully compressed and decompressed`);
      }
    } catch (error) {
      if (test.shouldThrow) {
        console.log(`✅ PASSED: Correctly threw: ${error.message}`);
      } else {
        console.log(`❌ FAILED: Unexpected error: ${error.message}`);
      }
    }
  }
  
  console.log("\n======================================");
  console.log("  Robustness Test Complete");
  console.log("======================================");
}

// Generate test data sets

function generateAlternating(size, val1, val2) {
  const data = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = i % 2 === 0 ? val1 : val2;
  }
  return data;
}

function generatePrimes(size) {
  const data = new Uint8Array(size);
  const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251];
  
  for (let i = 0; i < size; i++) {
    data[i] = primes[i % primes.length];
  }
  return data;
}

function generateFibonacci(size) {
  const data = new Uint8Array(size);
  data[0] = 0;
  if (size > 1) data[1] = 1;
  
  for (let i = 2; i < size; i++) {
    data[i] = (data[i-1] + data[i-2]) % 256; // Keep in byte range
  }
  return data;
}

function generateExponential(size) {
  const data = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = Math.min(255, Math.floor(Math.pow(1.1, i % 50))); // Reset to avoid overflow
  }
  return data;
}

function generateRandomWithRepeats(size) {
  const data = new Uint8Array(size);
  const possibleValues = [10, 20, 30, 40, 50, 60, 70, 80, 90];
  
  for (let i = 0; i < size; i++) {
    data[i] = possibleValues[Math.floor(Math.random() * possibleValues.length)];
  }
  return data;
}

function generateSpecialBytes(size) {
  const data = new Uint8Array(size);
  const specialBytes = [0, 10, 13, 26, 27, 32, 127, 128, 129, 255]; // Control characters, boundaries, etc.
  
  for (let i = 0; i < size; i++) {
    data[i] = specialBytes[i % specialBytes.length];
  }
  return data;
}

// Run tests
testRobustness();