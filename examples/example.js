/**
 * PrimeCompress Usage Examples
 * 
 * This file demonstrates basic usage of the PrimeCompress library.
 */

const compression = require('./unified-compression.js');

// Helper function to format byte sizes
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// Example 1: Compressing text data
function textCompressionExample() {
  console.log('\n==== Text Compression Example ====');
  
  // Create text sample
  const text = 'This is a sample text that contains repeated words. ' +
               'Compression works well when text contains repeated words ' +
               'and patterns that can be efficiently encoded using dictionary ' +
               'compression techniques like Huffman coding.';
  
  // Convert to Uint8Array
  const textData = new Uint8Array(Buffer.from(text));
  console.log(`Original size: ${formatBytes(textData.length)}`);
  
  // Analyze data
  const analysis = compression.analyzeCompression(textData);
  console.log(`Analysis: Entropy ${analysis.entropy.toFixed(2)}, recommended strategy: ${analysis.recommendedStrategy}`);
  
  // Compress
  console.log('Compressing...');
  const compressed = compression.compress(textData);
  console.log(`Compressed size: ${formatBytes(compressed.compressedSize)}`);
  console.log(`Compression ratio: ${compressed.compressionRatio.toFixed(2)}x`);
  console.log(`Strategy used: ${compressed.strategy}`);
  
  // Decompress
  console.log('Decompressing...');
  const decompressed = compression.decompress(compressed);
  
  // Verify
  const isMatch = textData.length === decompressed.length && 
                  textData.every((byte, i) => byte === decompressed[i]);
  console.log(`Decompression successful: ${isMatch ? 'Yes' : 'No'}`);
}

// Example 2: Compressing pattern data
function patternCompressionExample() {
  console.log('\n==== Pattern Compression Example ====');
  
  // Create a repeating pattern
  const pattern = [1, 2, 3, 4, 5, 4, 3, 2];
  const patternData = new Uint8Array(1024);
  
  for (let i = 0; i < patternData.length; i++) {
    patternData[i] = pattern[i % pattern.length];
  }
  
  console.log(`Original size: ${formatBytes(patternData.length)}`);
  
  // Analyze data
  const analysis = compression.analyzeCompression(patternData);
  console.log(`Analysis: Entropy ${analysis.entropy.toFixed(2)}, recommended strategy: ${analysis.recommendedStrategy}`);
  
  // Compress
  console.log('Compressing...');
  const compressed = compression.compress(patternData);
  console.log(`Compressed size: ${formatBytes(compressed.compressedSize)}`);
  console.log(`Compression ratio: ${compressed.compressionRatio.toFixed(2)}x`);
  console.log(`Strategy used: ${compressed.strategy}`);
  
  // Decompress
  console.log('Decompressing...');
  const decompressed = compression.decompress(compressed);
  
  // Verify
  const isMatch = patternData.length === decompressed.length && 
                  patternData.every((byte, i) => byte === decompressed[i]);
  console.log(`Decompression successful: ${isMatch ? 'Yes' : 'No'}`);
}

// Example 3: Compressing mixed data with block-based approach
function blockBasedCompressionExample() {
  console.log('\n==== Block-Based Compression Example ====');
  
  // Create mixed data with multiple patterns
  const mixedData = new Uint8Array(12288); // 12KB
  
  // Block 1: Zeros (constant data)
  for (let i = 0; i < 4096; i++) {
    mixedData[i] = 0; 
  }
  
  // Block 2: Sequential data
  for (let i = 0; i < 4096; i++) {
    mixedData[4096 + i] = i % 256;
  }
  
  // Block 3: Text-like data
  const text = 'This is block-based compression demonstration text. ';
  for (let i = 0; i < 4096; i++) {
    mixedData[8192 + i] = text.charCodeAt(i % text.length);
  }
  
  console.log(`Original size: ${formatBytes(mixedData.length)}`);
  
  // Compress with explicit block-based option
  console.log('Compressing with block-based strategy...');
  const compressed = compression.compress(mixedData, { useBlocks: true });
  console.log(`Compressed size: ${formatBytes(compressed.compressedSize)}`);
  console.log(`Compression ratio: ${compressed.compressionRatio.toFixed(2)}x`);
  console.log(`Strategy used: ${compressed.strategy}`);
  
  // Get more details about blocks
  if (compressed.metadata && compressed.metadata.blockStrategies) {
    console.log('Block strategies:', compressed.metadata.blockStrategies.join(', '));
  }
  
  // Decompress
  console.log('Decompressing...');
  const decompressed = compression.decompress(compressed);
  
  // Verify
  const isMatch = mixedData.length === decompressed.length && 
                  mixedData.every((byte, i) => byte === decompressed[i]);
  console.log(`Decompression successful: ${isMatch ? 'Yes' : 'No'}`);
}

// Run examples
textCompressionExample();
patternCompressionExample();
blockBasedCompressionExample();