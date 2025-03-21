/**
 * Test suite specifically for text data compression
 * This focuses on real-world text scenarios which have different
 * properties than binary or numeric data
 */

// Import the compression module
const compression = require('../prime-compression.js');

// Test text compression
function testTextCompression() {
  console.log("\n=== Text Data Compression Test ===");
  
  // Test data with different text characteristics
  const testCases = [
    {
      name: "Short text",
      text: "This is a test."
    },
    {
      name: "Medium text",
      text: "This is a test of the prime compression algorithm with real text data that might contain patterns and repetitions."
    },
    {
      name: "Repeating text",
      text: "test test test test test test test test test test test test test test test test test test test test"
    },
    {
      name: "Structured text",
      text: "Name: John Smith\nAge: 30\nOccupation: Engineer\nEmail: john@example.com\nPhone: 555-1234\nAddress: 123 Main St.\nCity: Anytown\nState: CA\nZIP: 90210"
    },
    {
      name: "JSON data",
      text: JSON.stringify({
        users: [
          { id: 1, name: "Alice", role: "admin", active: true },
          { id: 2, name: "Bob", role: "user", active: true },
          { id: 3, name: "Charlie", role: "user", active: false }
        ],
        settings: {
          theme: "dark",
          notifications: true,
          language: "en-US"
        }
      })
    }
  ];
  
  console.log("Results:");
  console.log("------------------------------------------------");
  console.log("Text Type           | Original | Compressed | Ratio | Method      | Match");
  console.log("------------------------------------------------");
  
  // Test each case
  for (const testCase of testCases) {
    // Convert text to byte array for compression
    const data = Buffer.from(testCase.text);
    
    try {
      // Compress the data
      const compressed = compression.compress(data);
      
      // Decompress the data
      const decompressed = compression.decompress(compressed);
      
      // Convert back to text for comparison
      const decompressedText = Buffer.from(decompressed).toString();
      
      // Check if the decompressed text matches the original
      const isMatch = decompressedText === testCase.text;
      
      // Calculate compression ratio
      const ratio = compressed.compressionRatio.toFixed(2);
      
      // Log results in table format
      console.log(
        `${testCase.name.padEnd(20)} | ${data.length.toString().padEnd(8)} | ${compressed.compressedSize.toString().padEnd(10)} | ${ratio.padEnd(5)} | ${(compressed.specialCase || 'standard').padEnd(11)} | ${isMatch ? 'Yes' : 'NO - FAIL'}`
      );
      
      if (!isMatch) {
        console.log("Original:", testCase.text);
        console.log("Decompressed:", decompressedText);
        console.log("First difference at position:", findFirstDifference(testCase.text, decompressedText));
      }
    } catch (error) {
      console.error(`Error with ${testCase.name}: ${error.message}`);
    }
  }
  console.log("------------------------------------------------");
}

// Test text compression with modifications to improve success rate
function testOptimizedTextCompression() {
  console.log("\n=== Optimized Text Compression Test ===");
  
  // First test using the textToByteOptimized conversion that better preserves text data
  function textToByteOptimized(text) {
    // For text data, we want to ensure each character's full range is preserved
    return new Uint8Array(Buffer.from(text));
  }
  
  function byteToTextOptimized(bytes) {
    // Convert bytes back to text representation
    return Buffer.from(bytes).toString();
  }
  
  const testTexts = [
    "This is a simple test string.",
    "Repeated data: ABABABABABABABABABABABABABABAB",
    "Structured data with numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
    "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
  ];
  
  console.log("Using optimized text conversion:");
  
  for (const text of testTexts) {
    const byteData = textToByteOptimized(text);
    
    try {
      // Analyze first
      const analysis = compression.analyzeCompression(byteData);
      
      // Compress
      const startTime = performance.now();
      const compressed = compression.compress(byteData);
      const compressionTime = performance.now() - startTime;
      
      // Decompress
      const decompStartTime = performance.now();
      const decompressed = compression.decompress(compressed);
      const decompressionTime = performance.now() - decompStartTime;
      
      // Convert back to text
      const recoveredText = byteToTextOptimized(decompressed);
      
      // Check results
      const isMatch = text === recoveredText;
      
      console.log(`Text (${text.length} chars) => ${compressed.compressionRatio.toFixed(2)}x compression, Method: ${compressed.specialCase || 'standard'}`);
      console.log(`Entropy: ${analysis.entropy.toFixed(2)}, Coherence: ${analysis.coherenceScore.toFixed(2)}`);
      console.log(`Times: Compression ${compressionTime.toFixed(2)}ms, Decompression ${decompressionTime.toFixed(2)}ms`);
      console.log(`Recovery successful: ${isMatch ? 'YES' : 'NO - FAILED'}`);
      
      if (!isMatch) {
        const diffPos = findFirstDifference(text, recoveredText);
        console.log(`First difference at position ${diffPos}:`);
        console.log(`Original: ${text.substring(Math.max(0, diffPos-10), diffPos)}[${text.charAt(diffPos)}]${text.substring(diffPos+1, diffPos+11)}`);
        console.log(`Recovered: ${recoveredText.substring(Math.max(0, diffPos-10), diffPos)}[${recoveredText.charAt(diffPos)}]${recoveredText.substring(diffPos+1, diffPos+11)}`);
        
        // Show character codes
        console.log(`Character codes - Original: ${text.charCodeAt(diffPos)}, Recovered: ${recoveredText.charCodeAt(diffPos)}`);
      }
      
      console.log('-----');
    } catch (error) {
      console.error(`Error with text "${text.substring(0, 30)}...": ${error.message}`);
      console.log('-----');
    }
  }
}

// Helper function to find the first difference between two strings
function findFirstDifference(str1, str2) {
  const minLength = Math.min(str1.length, str2.length);
  
  for (let i = 0; i < minLength; i++) {
    if (str1[i] !== str2[i]) {
      return i;
    }
  }
  
  // If we get here, then one string might be longer than the other
  if (str1.length !== str2.length) {
    return minLength;
  }
  
  // Strings are identical
  return -1;
}

// Test Unicode handling
function testUnicodeHandling() {
  console.log("\n=== Unicode Text Compression Test ===");
  
  const unicodeTexts = [
    "English: Hello World",
    "Spanish: ¡Hola Mundo!",
    "French: Bonjour le Monde",
    "German: Hallo Welt mit Umlauten äöüß",
    "Japanese: こんにちは世界",
    "Chinese: 你好，世界",
    "Russian: Привет, мир",
    "Arabic: مرحبا بالعالم",
    "Emojis: 😀🌍🚀🎉🎈🎁🎂🎊🎋🎌",
    "Mixed: Hello 你好 Привет こんにちは 😀🌍"
  ];
  
  console.log("Testing Unicode text compression:");
  
  for (const text of unicodeTexts) {
    // Unicode text requires special handling - use Buffer for proper encoding
    const byteData = new Uint8Array(Buffer.from(text));
    
    try {
      // Compress
      const compressed = compression.compress(byteData);
      
      // Decompress
      const decompressed = compression.decompress(compressed);
      
      // Convert back to text
      const recoveredText = Buffer.from(decompressed).toString();
      
      // Check results
      const isMatch = text === recoveredText;
      
      console.log(`${text.substring(0, 20)}... (${byteData.length} bytes) => ${compressed.compressionRatio.toFixed(2)}x, Method: ${compressed.specialCase || 'standard'}`);
      console.log(`Recovery successful: ${isMatch ? 'YES' : 'NO'}`);
      
      if (!isMatch) {
        const diffPos = findFirstDifference(text, recoveredText);
        console.log(`First difference at position ${diffPos}`);
        
        // For Unicode, show codepoints
        console.log(`Original codepoint: U+${text.codePointAt(diffPos).toString(16).padStart(4, '0')}`);
        console.log(`Recovered codepoint: U+${recoveredText.codePointAt(diffPos).toString(16).padStart(4, '0')}`);
      }
      
      console.log('-----');
    } catch (error) {
      console.error(`Error with Unicode text "${text.substring(0, 20)}...": ${error.message}`);
      console.log('-----');
    }
  }
}

// Test binary data
function testBinaryData() {
  console.log("\n=== Binary Data Compression Test ===");
  
  // Create different types of binary data
  const binaryTests = [
    {
      name: "Random bytes",
      data: (() => {
        const bytes = new Uint8Array(100);
        for (let i = 0; i < bytes.length; i++) {
          bytes[i] = Math.floor(Math.random() * 256);
        }
        return bytes;
      })()
    },
    {
      name: "Structured binary",
      data: (() => {
        // Simple binary format: header (8 bytes) + repeating records (4 bytes each)
        const bytes = new Uint8Array(100);
        // Header
        bytes[0] = 0x89; // Magic number
        bytes[1] = 0x50; // P
        bytes[2] = 0x4E; // N
        bytes[3] = 0x47; // G
        bytes[4] = 0x0D; // CR
        bytes[5] = 0x0A; // LF
        bytes[6] = 0x1A; // EOF
        bytes[7] = 0x0A; // LF
        
        // Records - structured pattern
        for (let i = 8; i < bytes.length; i += 4) {
          if (i + 3 < bytes.length) {
            bytes[i] = i % 256;
            bytes[i+1] = (i * 2) % 256;
            bytes[i+2] = (i / 2) % 256;
            bytes[i+3] = 0xFF;
          }
        }
        return bytes;
      })()
    },
    {
      name: "Gradient pattern",
      data: (() => {
        const bytes = new Uint8Array(256);
        for (let i = 0; i < bytes.length; i++) {
          bytes[i] = i % 256;
        }
        return bytes;
      })()
    },
    {
      name: "Repeating pattern",
      data: (() => {
        const bytes = new Uint8Array(100);
        const pattern = [0xAA, 0xBB, 0xCC, 0xDD];
        for (let i = 0; i < bytes.length; i++) {
          bytes[i] = pattern[i % pattern.length];
        }
        return bytes;
      })()
    }
  ];
  
  console.log("Testing binary data compression:");
  
  for (const test of binaryTests) {
    try {
      // Analyze first
      const analysis = compression.analyzeCompression(test.data);
      
      // Compress
      const compressed = compression.compress(test.data);
      
      // Decompress
      const decompressed = compression.decompress(compressed);
      
      // Compare bytes
      let isMatch = decompressed.length === test.data.length;
      if (isMatch) {
        for (let i = 0; i < test.data.length; i++) {
          if (test.data[i] !== decompressed[i]) {
            isMatch = false;
            console.log(`Mismatch at byte ${i}: Original=${test.data[i]}, Decompressed=${decompressed[i]}`);
            break;
          }
        }
      }
      
      console.log(`${test.name} (${test.data.length} bytes) => ${compressed.compressionRatio.toFixed(2)}x, Method: ${compressed.specialCase || 'standard'}`);
      console.log(`Entropy: ${analysis.entropy.toFixed(2)}, Coherence: ${analysis.coherenceScore.toFixed(2)}`);
      console.log(`Recovery successful: ${isMatch ? 'YES' : 'NO'}`);
      console.log('-----');
    } catch (error) {
      console.error(`Error with ${test.name}: ${error.message}`);
      console.log('-----');
    }
  }
}

// Run all tests
function runAllTests() {
  console.log("======================================");
  console.log("  Prime Compression Text/Binary Tests");
  console.log("======================================");
  
  testTextCompression();
  testOptimizedTextCompression();
  testUnicodeHandling();
  testBinaryData();
  
  console.log("\n======================================");
  console.log("          Tests Complete");
  console.log("======================================");
}

// Execute tests
runAllTests();