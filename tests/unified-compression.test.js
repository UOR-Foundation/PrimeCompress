/**
 * Comprehensive tests for the Unified Prime Compression Algorithm
 * 
 * This test suite validates the compression algorithm across various data types,
 * benchmarks performance against industry standards, tests edge cases, and
 * verifies perfect reconstruction for all lossless modes.
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

// Polyfill for TextEncoder/TextDecoder if in Node.js environment
if (typeof TextEncoder === 'undefined') {
  const util = require('util');
  global.TextEncoder = util.TextEncoder;
  global.TextDecoder = util.TextDecoder;
}

// Import Node.js zlib for comparison benchmarks
let zlib;
try {
  zlib = require('zlib');
} catch (e) {
  console.warn('zlib not available for comparison benchmarks');
  zlib = null;
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

// Test data generation utilities
const DataGenerator = {
  // Base method for generating test data
  generateTestData(size, pattern = 'random') {
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
  },

  // Generate sine wave data with optional noise
  generateSineWave(size, frequency = 0.05, amplitude = 127, phase = 0, noiseLevel = 0) {
    const data = new Uint8Array(size);
    const center = 128;
    
    for (let i = 0; i < size; i++) {
      // Generate sine wave
      let value = center + Math.round(amplitude * Math.sin(2 * Math.PI * frequency * i + phase));
      
      // Add noise if specified
      if (noiseLevel > 0) {
        value += Math.round((Math.random() * 2 - 1) * noiseLevel);
      }
      
      // Clamp to valid byte range
      data[i] = Math.max(0, Math.min(255, value));
    }
    
    return data;
  },

  // Generate composite signal (multiple sine waves)
  generateCompositeSignal(size, components = 3) {
    const data = new Uint8Array(size);
    const center = 128;
    
    // Generate multiple frequency components
    const frequencies = [];
    const amplitudes = [];
    const phases = [];
    
    let totalAmplitude = 0;
    for (let i = 0; i < components; i++) {
      frequencies.push(0.01 + (Math.random() * 0.09)); // 0.01 to 0.1
      const amp = 20 + Math.random() * 50;
      amplitudes.push(amp);
      totalAmplitude += amp;
      phases.push(Math.random() * Math.PI * 2);
    }
    
    // Normalize amplitudes to avoid clipping
    const normFactor = 127 / totalAmplitude;
    for (let i = 0; i < components; i++) {
      amplitudes[i] *= normFactor;
    }
    
    // Generate composite signal
    for (let i = 0; i < size; i++) {
      let value = center;
      
      for (let j = 0; j < components; j++) {
        value += amplitudes[j] * Math.sin(2 * Math.PI * frequencies[j] * i + phases[j]);
      }
      
      // Clamp to valid byte range
      data[i] = Math.max(0, Math.min(255, Math.round(value)));
    }
    
    return data;
  },

  // Generate exponential pattern
  generateExponential(size, base = 2, offset = 0) {
    const data = new Uint8Array(size);
    
    for (let i = 0; i < size; i++) {
      // Calculate the exponential value, scaled to fit in a byte
      const expValue = Math.pow(base, (i % 10) / 10);
      data[i] = Math.round((expValue % 1) * 255) + offset;
    }
    
    return data;
  },

  // Generate quasi-periodic pattern
  generateQuasiPeriodic(size) {
    const data = new Uint8Array(size);
    const primaryPeriod = 23; // Prime number
    const secondaryPeriod = 11; // Another prime
    
    for (let i = 0; i < size; i++) {
      // Combine two periods to create quasi-periodicity
      data[i] = ((i % primaryPeriod) * 10 + (i % secondaryPeriod) * 20) % 256;
    }
    
    return data;
  },

  // Generate text data
  generateText(size, type = 'english') {
    // Common English words for text generation
    const words = [
      'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 
      'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
      'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
      'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
      'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'
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
        
      case 'repeated':
        // Generate text with repeated paragraphs
        const paragraph = 'This is a sample paragraph with some repeating text. ';
        while (text.length < size) {
          text += paragraph;
        }
        break;
        
      case 'json':
        // Generate JSON-like data
        const jsonObj = {
          data: [],
          metadata: {
            version: '1.0',
            type: 'test-data',
            size: size
          }
        };
        
        // Add some array data
        for (let i = 0; i < size / 20; i++) {
          jsonObj.data.push({
            id: i,
            value: Math.random() * 100,
            name: `Item ${i}`,
            active: i % 2 === 0
          });
        }
        
        text = JSON.stringify(jsonObj, null, 2);
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

  // Generate mixed data (combination of different patterns)
  generateMixedData(size) {
    const data = new Uint8Array(size);
    
    // Divide the data into sections with different patterns
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
    const frequency = 0.05;
    const amplitude = 127;
    const center = 128;
    for (let i = 0; i < sectionSize; i++) {
      data[sectionSize * 2 + i] = Math.round(center + amplitude * Math.sin(2 * Math.PI * frequency * i));
    }
    
    // Section 4: Repeated pattern
    const patternLength = 8;
    const repeatedPattern = new Uint8Array(patternLength);
    for (let i = 0; i < patternLength; i++) {
      repeatedPattern[i] = Math.floor(Math.random() * 256);
    }
    
    for (let i = 0; i < sectionSize; i++) {
      data[sectionSize * 3 + i] = repeatedPattern[i % patternLength];
    }
    
    // Section 5: All zeros
    for (let i = 0; i < size - (sectionSize * 4); i++) {
      data[sectionSize * 4 + i] = 0;
    }
    
    return data;
  }
};

// Compression verification and benchmarking utilities
const CompressionTester = {
  // Verify compression integrity and return detailed results
  verifyCompression(data, strategy = null, options = {}) {
    const results = {
      originalSize: data.length,
      originalChecksum: calculateChecksum(data),
      startTime: performance.now(),
      compressionTime: 0,
      decompressionTime: 0,
      compressedSize: 0,
      compressionRatio: 0,
      strategySelected: null,
      integrityPassed: false,
      errors: []
    };
    
    try {
      // Compress the data with optional strategy
      const startCompress = performance.now();
      const compressed = strategy 
        ? compression.compressWithStrategy(data, strategy, options)
        : compression.compress(data, options);
      results.compressionTime = performance.now() - startCompress;
      
      results.compressedSize = compressed.compressedSize || 
                               (compressed.compressedVector ? compressed.compressedVector.length : 0);
      results.compressionRatio = compressed.compressionRatio || (data.length / results.compressedSize);
      results.strategySelected = compressed.strategy || compressed.compressionMethod || 'unknown';
      
      // Decompress the data
      const startDecompress = performance.now();
      const decompressed = compression.decompress(compressed);
      results.decompressionTime = performance.now() - startDecompress;
      
      // Verify sizes match
      if (data.length !== decompressed.length) {
        results.errors.push(`Size mismatch: original=${data.length}, decompressed=${decompressed.length}`);
      }
      
      // Calculate checksum for decompressed data
      results.decompressedChecksum = calculateChecksum(decompressed);
      
      // Verify checksums match
      if (results.originalChecksum !== results.decompressedChecksum) {
        results.errors.push(`Checksum mismatch: original=${results.originalChecksum}, decompressed=${results.decompressedChecksum}`);
      }
      
      // If no errors, integrity passed
      results.integrityPassed = results.errors.length === 0;
      
      // Full byte-by-byte check to count mismatches (limited to improve performance)
      if (!results.integrityPassed) {
        let mismatchCount = 0;
        let firstMismatches = [];
        const MAX_MISMATCHES_TO_LOG = 5;
        
        for (let i = 0; i < data.length; i++) {
          if (data[i] !== decompressed[i]) {
            mismatchCount++;
            
            if (firstMismatches.length < MAX_MISMATCHES_TO_LOG) {
              firstMismatches.push(`Byte ${i}: original=${data[i]}, decompressed=${decompressed[i]}`);
            }
            
            // Check a reasonable number of bytes to improve performance
            if (mismatchCount > 100) break;
          }
        }
        
        results.mismatchCount = mismatchCount;
        results.mismatchExamples = firstMismatches;
        results.mismatchPercentage = (mismatchCount / data.length) * 100;
      }
    } catch (e) {
      results.errors.push(`Exception: ${e.message}`);
      results.integrityPassed = false;
    }
    
    results.totalTime = performance.now() - results.startTime;
    return results;
  },

  // Benchmark against industry standards
  benchmarkAgainstStandards(data, options = {}) {
    const results = {
      originalSize: data.length,
      primeCompression: { time: 0, size: 0, ratio: 0 },
      gzip: { time: 0, size: 0, ratio: 0 },
      brotli: { time: 0, size: 0, ratio: 0 }
    };
    
    // Prime Compression benchmark
    try {
      const startTime = performance.now();
      const compressed = compression.compress(data, options);
      results.primeCompression.time = performance.now() - startTime;
      results.primeCompression.size = compressed.compressedSize || 
                                     (compressed.compressedVector ? compressed.compressedVector.length : data.length);
      results.primeCompression.ratio = compressed.compressionRatio || (data.length / results.primeCompression.size);
      results.primeCompression.strategy = compressed.strategy || compressed.compressionMethod || 'unknown';
    } catch (e) {
      results.primeCompression.error = e.message;
    }
    
    // GZIP benchmark if available
    if (zlib) {
      try {
        const buffer = Buffer.from(data);
        const startTime = performance.now();
        const compressed = zlib.gzipSync(buffer, { level: 9 });
        results.gzip.time = performance.now() - startTime;
        results.gzip.size = compressed.length;
        results.gzip.ratio = data.length / compressed.length;
      } catch (e) {
        results.gzip.error = e.message;
      }
      
      // Brotli benchmark if available
      if (zlib.brotliCompressSync) {
        try {
          const buffer = Buffer.from(data);
          const startTime = performance.now();
          const compressed = zlib.brotliCompressSync(buffer, { 
            params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 11 }
          });
          results.brotli.time = performance.now() - startTime;
          results.brotli.size = compressed.length;
          results.brotli.ratio = data.length / compressed.length;
        } catch (e) {
          results.brotli.error = e.message;
        }
      } else {
        results.brotli.error = 'Brotli not available';
      }
    } else {
      results.gzip.error = 'zlib not available';
      results.brotli.error = 'zlib not available';
    }
    
    return results;
  }
};

// Test runner for comprehensive testing
const TestRunner = {
  // For console output formatting
  colors: {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    dim: '\x1b[2m',
    underscore: '\x1b[4m',
    blink: '\x1b[5m',
    reverse: '\x1b[7m',
    hidden: '\x1b[8m',
    
    fg: {
      black: '\x1b[30m',
      red: '\x1b[31m',
      green: '\x1b[32m',
      yellow: '\x1b[33m',
      blue: '\x1b[34m',
      magenta: '\x1b[35m',
      cyan: '\x1b[36m',
      white: '\x1b[37m'
    },
    
    bg: {
      black: '\x1b[40m',
      red: '\x1b[41m',
      green: '\x1b[42m',
      yellow: '\x1b[43m',
      blue: '\x1b[44m',
      magenta: '\x1b[45m',
      cyan: '\x1b[46m',
      white: '\x1b[47m'
    }
  },
  
  // Log formatted output
  log: {
    header(text) {
      console.log(`\n${TestRunner.colors.bright}${TestRunner.colors.fg.cyan}===== ${text} =====${TestRunner.colors.reset}`);
    },
    
    subheader(text) {
      console.log(`\n${TestRunner.colors.fg.cyan}--- ${text} ---${TestRunner.colors.reset}`);
    },
    
    success(text) {
      console.log(`${TestRunner.colors.fg.green}✓ ${text}${TestRunner.colors.reset}`);
    },
    
    error(text) {
      console.log(`${TestRunner.colors.fg.red}✗ ${text}${TestRunner.colors.reset}`);
    },
    
    warning(text) {
      console.log(`${TestRunner.colors.fg.yellow}! ${text}${TestRunner.colors.reset}`);
    },
    
    info(text) {
      console.log(`${TestRunner.colors.fg.blue}i ${text}${TestRunner.colors.reset}`);
    },
    
    data(text) {
      console.log(`  ${text}`);
    }
  },
  
  // Run a test suite with multiple test cases
  runTestSuite(suiteName, testCases) {
    TestRunner.log.header(suiteName);
    
    const results = {
      name: suiteName,
      totalTests: testCases.length,
      passed: 0,
      failed: 0,
      details: []
    };
    
    for (const testCase of testCases) {
      TestRunner.log.subheader(testCase.name);
      
      const testResult = {
        name: testCase.name,
        passed: true,
        details: {}
      };
      
      try {
        // Generate test data
        const data = testCase.dataGenerator();
        TestRunner.log.info(`Generated ${formatBytes(data.length)} of test data`);
        
        // Run compression test
        const compressionResult = CompressionTester.verifyCompression(
          data, 
          testCase.strategy || null, 
          testCase.options || {}
        );
        
        testResult.details = compressionResult;
        
        // Log basic information
        TestRunner.log.data(`Original size: ${formatBytes(compressionResult.originalSize)}`);
        TestRunner.log.data(`Compressed size: ${formatBytes(compressionResult.compressedSize)}`);
        TestRunner.log.data(`Compression ratio: ${compressionResult.compressionRatio.toFixed(2)}x`);
        TestRunner.log.data(`Strategy selected: ${compressionResult.strategySelected}`);
        TestRunner.log.data(`Compression time: ${formatTime(compressionResult.compressionTime)}`);
        TestRunner.log.data(`Decompression time: ${formatTime(compressionResult.decompressionTime)}`);
        
        // Check if the compression ratio meets expectations
        if (testCase.expectedMinRatio && compressionResult.compressionRatio < testCase.expectedMinRatio) {
          testResult.passed = false;
          TestRunner.log.warning(`Compression ratio (${compressionResult.compressionRatio.toFixed(2)}x) is below expected minimum (${testCase.expectedMinRatio}x)`);
        } else if (testCase.expectedMinRatio) {
          TestRunner.log.success(`Compression ratio meets or exceeds expected minimum (${testCase.expectedMinRatio}x)`);
        }
        
        // Check if the selected strategy matches expected
        if (testCase.expectedStrategy && compressionResult.strategySelected !== testCase.expectedStrategy) {
          TestRunner.log.warning(`Selected strategy (${compressionResult.strategySelected}) doesn't match expected (${testCase.expectedStrategy})`);
        } else if (testCase.expectedStrategy) {
          TestRunner.log.success(`Correctly selected ${testCase.expectedStrategy} strategy`);
        }
        
        // Check integrity
        if (compressionResult.integrityPassed) {
          TestRunner.log.success(`Decompression integrity verified (100% match)`);
        } else {
          testResult.passed = false;
          TestRunner.log.error(`Decompression integrity check failed`);
          
          for (const error of compressionResult.errors) {
            TestRunner.log.error(`  ${error}`);
          }
          
          if (compressionResult.mismatchExamples) {
            for (const mismatch of compressionResult.mismatchExamples) {
              TestRunner.log.data(`  ${mismatch}`);
            }
          }
        }
        
        // Run benchmark comparison if requested
        if (testCase.benchmark) {
          TestRunner.log.subheader('Benchmark Comparison');
          const benchmarkResult = CompressionTester.benchmarkAgainstStandards(
            data, 
            testCase.options || {}
          );
          
          testResult.benchmark = benchmarkResult;
          
          // Prime compression
          TestRunner.log.data(`Prime Compression: ${benchmarkResult.primeCompression.ratio.toFixed(2)}x ratio in ${formatTime(benchmarkResult.primeCompression.time)}`);
          
          // GZIP
          if (benchmarkResult.gzip.error) {
            TestRunner.log.warning(`GZIP: ${benchmarkResult.gzip.error}`);
          } else {
            TestRunner.log.data(`GZIP: ${benchmarkResult.gzip.ratio.toFixed(2)}x ratio in ${formatTime(benchmarkResult.gzip.time)}`);
          }
          
          // Brotli
          if (benchmarkResult.brotli.error) {
            TestRunner.log.warning(`Brotli: ${benchmarkResult.brotli.error}`);
          } else {
            TestRunner.log.data(`Brotli: ${benchmarkResult.brotli.ratio.toFixed(2)}x ratio in ${formatTime(benchmarkResult.brotli.time)}`);
          }
          
          // Compression winner
          const algorithms = [];
          if (!benchmarkResult.primeCompression.error) algorithms.push({name: 'Prime', ratio: benchmarkResult.primeCompression.ratio});
          if (!benchmarkResult.gzip.error) algorithms.push({name: 'GZIP', ratio: benchmarkResult.gzip.ratio});
          if (!benchmarkResult.brotli.error) algorithms.push({name: 'Brotli', ratio: benchmarkResult.brotli.ratio});
          
          if (algorithms.length > 0) {
            algorithms.sort((a, b) => b.ratio - a.ratio);
            TestRunner.log.info(`Best compression ratio: ${algorithms[0].name} (${algorithms[0].ratio.toFixed(2)}x)`);
          }
        }
        
        // Final test case result
        if (testResult.passed) {
          TestRunner.log.success(`Test case PASSED`);
          results.passed++;
        } else {
          TestRunner.log.error(`Test case FAILED`);
          results.failed++;
        }
      } catch (e) {
        testResult.passed = false;
        testResult.error = e.message;
        TestRunner.log.error(`Test exception: ${e.message}`);
        results.failed++;
      }
      
      results.details.push(testResult);
    }
    
    // Suite summary
    TestRunner.log.header(`${suiteName} Summary`);
    TestRunner.log.data(`Total tests: ${results.totalTests}`);
    TestRunner.log.data(`Passed: ${results.passed}`);
    TestRunner.log.data(`Failed: ${results.failed}`);
    
    if (results.failed === 0) {
      TestRunner.log.success(`All tests PASSED`);
    } else {
      TestRunner.log.error(`${results.failed}/${results.totalTests} tests FAILED`);
    }
    
    return results;
  }
};

// Main test cases organized by compression method
const testSuites = {
  // Test specialized compression methods with their ideal data types
  specializedCompressionTests: [
    {
      name: 'Zero Compression',
      dataGenerator: () => DataGenerator.generateTestData(4096, 'zeros'),
      expectedMinRatio: 100,
      expectedStrategy: 'zeros',
      benchmark: true
    },
    {
      name: 'Sequential Pattern Compression',
      dataGenerator: () => DataGenerator.generateTestData(4096, 'sequential'),
      expectedMinRatio: 10,
      expectedStrategy: 'pattern',
      benchmark: true
    },
    {
      name: 'Repeated Pattern Compression',
      dataGenerator: () => DataGenerator.generateTestData(4096, 'repeated'),
      expectedMinRatio: 5,
      expectedStrategy: 'dictionary',
      benchmark: true
    },
    {
      name: 'Sine Wave Spectral Compression',
      dataGenerator: () => DataGenerator.generateSineWave(4096, 0.02),
      expectedMinRatio: 30,
      expectedStrategy: 'spectral',
      benchmark: true
    },
    {
      name: 'Composite Wave Spectral Compression',
      dataGenerator: () => DataGenerator.generateCompositeSignal(4096, 3),
      expectedMinRatio: 10,
      expectedStrategy: 'spectral',
      benchmark: true
    },
    {
      name: 'Text Compression',
      dataGenerator: () => DataGenerator.generateText(4096, 'english'),
      expectedMinRatio: 2,
      expectedStrategy: 'dictionary',
      benchmark: true
    },
    {
      name: 'Exponential Pattern Compression',
      dataGenerator: () => DataGenerator.generateExponential(4096),
      expectedMinRatio: 5,
      expectedStrategy: 'pattern',
      benchmark: true
    },
    {
      name: 'Quasi-Periodic Pattern Compression',
      dataGenerator: () => DataGenerator.generateQuasiPeriodic(4096),
      expectedMinRatio: 5,
      expectedStrategy: 'pattern',
      benchmark: true
    },
    {
      name: 'Random Data Statistical Compression',
      dataGenerator: () => DataGenerator.generateTestData(4096, 'random'),
      expectedMinRatio: 1,
      expectedStrategy: 'statistical',
      benchmark: true
    }
  ],
  
  // Test varying data sizes to ensure stability across sizes
  dataSizeTests: [
    {
      name: 'Tiny Data (64 bytes)',
      dataGenerator: () => DataGenerator.generateMixedData(64),
      benchmark: false
    },
    {
      name: 'Small Data (512 bytes)',
      dataGenerator: () => DataGenerator.generateMixedData(512),
      benchmark: false
    },
    {
      name: 'Medium Data (4KB)',
      dataGenerator: () => DataGenerator.generateMixedData(4 * 1024),
      benchmark: false
    },
    {
      name: 'Large Data (64KB)',
      dataGenerator: () => DataGenerator.generateMixedData(64 * 1024),
      benchmark: false
    },
    {
      name: 'Huge Data (512KB)',
      dataGenerator: () => DataGenerator.generateMixedData(512 * 1024),
      benchmark: false,
      options: { fastMode: true } // Use fast mode for large data
    }
  ],
  
  // Edge cases designed to test the robustness of the algorithm
  edgeCaseTests: [
    {
      name: 'Single Byte Data',
      dataGenerator: () => new Uint8Array([42]),
      benchmark: false
    },
    {
      name: 'Two Byte Data',
      dataGenerator: () => new Uint8Array([127, 128]),
      benchmark: false
    },
    {
      name: 'Alternating High-Low Values',
      dataGenerator: () => {
        const data = new Uint8Array(1024);
        for (let i = 0; i < data.length; i++) {
          data[i] = i % 2 === 0 ? 0 : 255;
        }
        return data;
      },
      benchmark: false
    },
    {
      name: 'All Same Value (not zero)',
      dataGenerator: () => {
        const data = new Uint8Array(1024);
        data.fill(123);
        return data;
      },
      expectedMinRatio: 100,
      benchmark: false
    },
    {
      name: 'Incremental Steps with Noise',
      dataGenerator: () => {
        const data = new Uint8Array(1024);
        for (let i = 0; i < data.length; i++) {
          // Base value increases by 1 every 4 bytes
          const base = Math.floor(i / 4) % 256;
          // Add small noise
          data[i] = (base + (Math.random() < 0.5 ? 0 : 1)) % 256;
        }
        return data;
      },
      benchmark: false
    },
    {
      name: 'High Entropy with Small Patterns',
      dataGenerator: () => {
        const data = new Uint8Array(1024);
        const pattern = [13, 27, 42, 98];
        
        for (let i = 0; i < data.length; i++) {
          if (i % 16 < pattern.length) {
            // Insert pattern occasionally
            data[i] = pattern[i % pattern.length];
          } else {
            // Otherwise random
            data[i] = Math.floor(Math.random() * 256);
          }
        }
        return data;
      },
      benchmark: false
    }
  ],
  
  // Test forced strategies to validate each method works correctly
  forcedStrategyTests: [
    {
      name: 'Force Zeros Strategy on Random Data',
      dataGenerator: () => DataGenerator.generateTestData(1024, 'random'),
      strategy: 'zeros'
    },
    {
      name: 'Force Pattern Strategy on Random Data',
      dataGenerator: () => DataGenerator.generateTestData(1024, 'random'),
      strategy: 'pattern'
    },
    {
      name: 'Force Spectral Strategy on Random Data',
      dataGenerator: () => DataGenerator.generateTestData(1024, 'random'),
      strategy: 'spectral'
    },
    {
      name: 'Force Dictionary Strategy on Random Data',
      dataGenerator: () => DataGenerator.generateTestData(1024, 'random'),
      strategy: 'dictionary'
    },
    {
      name: 'Force Statistical Strategy on Zeros Data',
      dataGenerator: () => DataGenerator.generateTestData(1024, 'zeros'),
      strategy: 'statistical'
    }
  ],
  
  // Test the strategy selector's adaptability to mixed data
  strategyAdaptabilityTests: [
    {
      name: 'Mixed Data Strategy Selection (Equal Parts)',
      dataGenerator: () => DataGenerator.generateMixedData(5120),
      benchmark: true
    },
    {
      name: 'Dominant Sine Wave with Noise',
      dataGenerator: () => {
        // 90% sine wave, 10% random
        const data = DataGenerator.generateSineWave(5120, 0.03, 120, 0, 0);
        const noiseStart = Math.floor(data.length * 0.9);
        
        for (let i = noiseStart; i < data.length; i++) {
          data[i] = Math.floor(Math.random() * 256);
        }
        
        return data;
      },
      expectedStrategy: 'spectral',
      benchmark: true
    },
    {
      name: 'Text with Binary Data',
      dataGenerator: () => {
        // 80% text, 20% binary
        const textData = DataGenerator.generateText(4096, 'english');
        const binaryData = DataGenerator.generateTestData(1024, 'random');
        
        const combined = new Uint8Array(textData.length + binaryData.length);
        combined.set(textData, 0);
        combined.set(binaryData, textData.length);
        
        return combined;
      },
      expectedStrategy: 'dictionary',
      benchmark: true
    },
    {
      name: 'Patterns with Random Sections',
      dataGenerator: () => {
        const data = new Uint8Array(5120);
        
        // Fill with sequential pattern
        for (let i = 0; i < data.length; i++) {
          data[i] = i % 256;
        }
        
        // Add random sections
        for (let i = 0; i < 5; i++) {
          const sectionStart = Math.floor(Math.random() * (data.length - 200));
          for (let j = 0; j < 200; j++) {
            data[sectionStart + j] = Math.floor(Math.random() * 256);
          }
        }
        
        return data;
      },
      expectedStrategy: 'pattern',
      benchmark: true
    }
  ]
};

// Run all test suites
function runAllTests() {
  console.log("==================================================");
  console.log(" Prime Unified Compression Algorithm - Test Suite ");
  console.log("==================================================");
  
  const results = [];
  
  // Run each test suite
  for (const [suiteName, testCases] of Object.entries(testSuites)) {
    const suiteResult = TestRunner.runTestSuite(suiteName, testCases);
    results.push(suiteResult);
  }
  
  // Overall summary
  console.log("\n==================================================");
  console.log("                Overall Summary                   ");
  console.log("==================================================");
  
  let totalTests = 0;
  let totalPassed = 0;
  let totalFailed = 0;
  
  for (const result of results) {
    console.log(`${result.name}: ${result.passed}/${result.totalTests} passed, ${result.failed} failed`);
    totalTests += result.totalTests;
    totalPassed += result.passed;
    totalFailed += result.failed;
  }
  
  console.log("\nTotal tests: " + totalTests);
  console.log(`Passed: ${totalPassed} (${((totalPassed / totalTests) * 100).toFixed(2)}%)`);
  console.log(`Failed: ${totalFailed} (${((totalFailed / totalTests) * 100).toFixed(2)}%)`);
  
  console.log("\n==================================================");
  if (totalFailed === 0) {
    console.log("                 ALL TESTS PASSED                  ");
  } else {
    console.log(`                 ${totalFailed} TESTS FAILED                  `);
  }
  console.log("==================================================");
  
  return totalFailed === 0;
}

// Execute tests
runAllTests();