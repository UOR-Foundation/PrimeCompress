/**
 * Enhanced pattern compression and decompression functions
 * 
 * This module provides improved implementations for pattern detection, 
 * compression and decompression to fix the identified issues in the test suite.
 */

/**
 * Detect if data contains a repeating pattern
 * 
 * @param {Uint8Array} data - The data to analyze
 * @param {Object} options - Options for pattern detection
 * @returns {Object|null} Pattern information or null if no pattern found
 */
function detectPattern(data, options = {}) {
  if (!data || data.length < 4) return null;
  
  const maxPatternSize = options.maxPatternSize || Math.min(64, Math.floor(data.length / 2));
  const minPatternSize = options.minPatternSize || 1;
  const minRepetitions = options.minRepetitions || 3;
  
  // Try different pattern sizes, preferring smaller patterns
  for (let patternSize = minPatternSize; patternSize <= maxPatternSize; patternSize++) {
    // Extract potential pattern
    const pattern = Array.from(data.slice(0, patternSize));
    let isPattern = true;
    let matchedLength = 0;
    
    // Check if pattern repeats throughout the data
    for (let i = 0; i < Math.min(data.length, 1024); i++) {
      if (data[i] !== pattern[i % patternSize]) {
        isPattern = false;
        break;
      }
      matchedLength = i + 1;
    }
    
    // If we've found a pattern that repeats enough times
    if (isPattern && matchedLength >= patternSize * minRepetitions) {
      return {
        pattern,
        patternSize,
        repetitions: Math.ceil(data.length / patternSize)
      };
    }
  }
  
  // Check for constant value (special case of pattern)
  const firstValue = data[0];
  let isConstant = true;
  
  for (let i = 1; i < Math.min(data.length, 1024); i++) {
    if (data[i] !== firstValue) {
      isConstant = false;
      break;
    }
  }
  
  if (isConstant) {
    return {
      pattern: [firstValue],
      patternSize: 1,
      repetitions: data.length,
      isConstant: true
    };
  }
  
  return null;
}

/**
 * Create enhanced pattern compression
 * 
 * @param {Uint8Array} data - The data to compress
 * @param {String} checksum - Data checksum for verification
 * @returns {Object} Compressed data object
 */
function createPatternCompression(data, checksum) {
  // Detect pattern in the data
  const patternInfo = detectPattern(data);
  
  if (!patternInfo) {
    // Fall back to storing the whole data if no pattern detected
    return {
      compressionType: 'standard',
      specialCase: 'pattern',
      compressedVector: Array.from(data),
      compressedSize: data.length,
      compressionRatio: 1,
      originalSize: data.length,
      checksum,
      patternSize: data.length,
      repetitions: 1
    };
  }
  
  // For constant value patterns, use zeros compression if value is 0
  if (patternInfo.isConstant && patternInfo.pattern[0] === 0) {
    return createZerosCompression(data, checksum);
  }
  
  // Create compressed representation
  return {
    compressionType: 'standard',
    specialCase: 'pattern',
    compressedVector: patternInfo.pattern,
    compressedSize: patternInfo.pattern.length,
    compressionRatio: data.length / patternInfo.pattern.length,
    originalSize: data.length,
    checksum,
    patternSize: patternInfo.patternSize,
    repetitions: patternInfo.repetitions
  };
}

/**
 * Create compression for zeros or constant values
 * 
 * @param {Uint8Array} data - The data to compress
 * @param {String} checksum - Data checksum for verification
 * @returns {Object} Compressed data object
 */
function createZerosCompression(data, checksum) {
  // Check if all values are the same
  const firstValue = data[0];
  let isConstant = true;
  
  for (let i = 1; i < data.length; i++) {
    if (data[i] !== firstValue) {
      isConstant = false;
      break;
    }
  }
  
  if (!isConstant) {
    // Fall back to pattern compression if not constant
    return createPatternCompression(data, checksum);
  }
  
  // Create compressed representation for constant value
  return {
    compressionType: 'standard',
    specialCase: 'zeros',
    compressedVector: [firstValue],
    compressedSize: 1,
    compressionRatio: data.length,
    originalSize: data.length,
    checksum,
    constantValue: firstValue
  };
}

/**
 * Decompress pattern-compressed data
 * 
 * @param {Object} compressedData - The compressed data object
 * @returns {Uint8Array} Decompressed data
 */
function decompressPattern(compressedData) {
  const pattern = compressedData.compressedVector;
  const result = new Uint8Array(compressedData.originalSize);
  
  // Handle empty pattern (shouldn't happen, but just in case)
  if (!pattern || pattern.length === 0) {
    return result;
  }
  
  // Fill result with repeating pattern
  for (let i = 0; i < result.length; i++) {
    result[i] = pattern[i % pattern.length];
  }
  
  return result;
}

/**
 * Decompress zeros or constant value compressed data
 * 
 * @param {Object} compressedData - The compressed data object
 * @returns {Uint8Array} Decompressed data
 */
function decompressZeros(compressedData) {
  const result = new Uint8Array(compressedData.originalSize);
  
  // For non-zero constant values, use the stored constant value
  const constantValue = compressedData.constantValue || 0;
  
  if (constantValue !== 0) {
    result.fill(constantValue);
  } // For zero values, the array is already filled with zeros by default
  
  return result;
}

// Export the enhanced functions
module.exports = {
  detectPattern,
  createPatternCompression,
  createZerosCompression,
  decompressPattern,
  decompressZeros
};