/**
 * Performance tests for the Prime Compression algorithm
 * This tests the algorithm's performance with larger data sets
 */

// Import the compression module
const compression = require('../prime-compression.js');

// Helper function to format byte sizes
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// Helper function to format throughput
function formatThroughput(bytesPerSecond) {
  if (bytesPerSecond < 1024) return `${bytesPerSecond.toFixed(2)} B/s`;
  if (bytesPerSecond < 1024 * 1024) return `${(bytesPerSecond / 1024).toFixed(2)} KB/s`;
  return `${(bytesPerSecond / (1024 * 1024)).toFixed(2)} MB/s`;
}

// Configure test sizes
const testSizes = [
  10 * 1024,      // 10 KB
  100 * 1024,     // 100 KB
  500 * 1024      // 500 KB
];

// Configure test data patterns
const testPatterns = [
  { name: "Random", generator: (size) => randomData(size) },
  { name: "Text", generator: (size) => textData(size) },
  { name: "Sequential", generator: (size) => sequentialData(size) },
  { name: "Repeating", generator: (size) => repeatingData(size) },
  { name: "JSON", generator: (size) => structuredJsonData(size) }
];

// Generate random data
function randomData(size) {
  const data = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = Math.floor(Math.random() * 256);
  }
  return data;
}

// Generate text-like data
function textData(size) {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:!?-()[]{}'\"\n\t";
  const data = new Uint8Array(size);
  
  // Make it somewhat realistic with words and sentences
  let pos = 0;
  while (pos < size) {
    // Generate a word
    const wordLength = Math.floor(Math.random() * 10) + 1;
    for (let i = 0; i < wordLength && pos < size; i++) {
      data[pos++] = chars.charCodeAt(Math.floor(Math.random() * 52) + 0); // A-Za-z
    }
    
    // Add space or punctuation
    if (pos < size) {
      const punct = Math.random();
      if (punct < 0.8) {
        data[pos++] = 32; // space
      } else if (punct < 0.9) {
        data[pos++] = chars.charCodeAt(Math.floor(Math.random() * 10) + 52); // Punctuation
        if (pos < size) data[pos++] = 32; // space after punctuation
      } else {
        data[pos++] = 10; // newline
      }
    }
  }
  
  return data;
}

// Generate sequential data
function sequentialData(size) {
  const data = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = i % 256;
  }
  return data;
}

// Generate repeating pattern data
function repeatingData(size) {
  const patternLength = 64;
  const pattern = new Uint8Array(patternLength);
  
  // Create random pattern
  for (let i = 0; i < patternLength; i++) {
    pattern[i] = Math.floor(Math.random() * 256);
  }
  
  // Repeat pattern
  const data = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = pattern[i % patternLength];
  }
  
  return data;
}

// Generate JSON-like structured data
function structuredJsonData(size) {
  // Create an array of objects to mimic realistic JSON
  const objects = [];
  const objectCount = Math.floor(size / 100); // Roughly size objects
  
  for (let i = 0; i < objectCount; i++) {
    objects.push({
      id: i,
      name: `Item${i}`,
      active: Math.random() > 0.5,
      value: Math.floor(Math.random() * 1000) / 10,
      tags: ['tag' + Math.floor(Math.random() * 10), 'tag' + Math.floor(Math.random() * 10)],
      metadata: {
        created: new Date().toISOString(),
        priority: Math.floor(Math.random() * 5)
      }
    });
  }
  
  // Convert to JSON string
  const jsonString = JSON.stringify(objects);
  
  // Truncate or pad to reach exact size
  if (jsonString.length >= size) {
    return new Uint8Array(Buffer.from(jsonString.substring(0, size)));
  } else {
    const data = new Uint8Array(size);
    const jsonBuffer = Buffer.from(jsonString);
    
    // Copy JSON data
    for (let i = 0; i < jsonBuffer.length; i++) {
      data[i] = jsonBuffer[i];
    }
    
    // Pad with repeated JSON
    for (let i = jsonBuffer.length; i < size; i++) {
      data[i] = jsonBuffer[i % jsonBuffer.length];
    }
    
    return data;
  }
}

// Run performance test
function runPerformanceTest() {
  console.log("======================================");
  console.log("  Prime Compression Performance Test");
  console.log("======================================");
  
  for (const pattern of testPatterns) {
    console.log(`\n--- ${pattern.name} Data ---`);
    console.table([
      ["Size", "Comp Time", "Decomp Time", "Comp Method", "Ratio", "Comp Speed", "Decomp Speed"]
    ]);
    
    for (const size of testSizes) {
      const formattedSize = formatBytes(size);
      
      // Generate test data
      const data = pattern.generator(size);
      
      // Test compression
      const compressStart = performance.now();
      const compressed = compression.compress(data);
      const compressTime = performance.now() - compressStart;
      
      // Test decompression
      const decompressStart = performance.now();
      const decompressed = compression.decompress(compressed);
      const decompressTime = performance.now() - decompressStart;
      
      // Verify correctness
      let isValid = data.length === decompressed.length;
      if (isValid) {
        // Check a sample of values (checking every byte could be slow for large sizes)
        const checkInterval = Math.max(1, Math.floor(data.length / 1000));
        for (let i = 0; i < data.length; i += checkInterval) {
          if (data[i] !== decompressed[i]) {
            isValid = false;
            console.error(`Verification failed at position ${i}`);
            break;
          }
        }
      }
      
      // Calculate compression ratio and speeds
      const compressionRatio = compressed.compressionRatio;
      const compressionSpeed = formatThroughput(size / (compressTime / 1000));
      const decompressionSpeed = formatThroughput(size / (decompressTime / 1000));
      
      // Print results
      console.log(
        `${formattedSize}   ${compressTime.toFixed(2)}ms   ${decompressTime.toFixed(2)}ms   ` +
        `${compressed.specialCase || 'standard'}   ${compressionRatio.toFixed(2)}x   ` +
        `${compressionSpeed}   ${decompressionSpeed}   ${isValid ? '✓' : '✗'}`
      );
    }
  }
  
  console.log("\n======================================");
  console.log("  Performance Test Complete");
  console.log("======================================");
}

// Run tests
runPerformanceTest();