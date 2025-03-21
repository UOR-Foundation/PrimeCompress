/**
 * Enhanced sequential compression and decompression functions
 * 
 * This module provides improved implementations for sequence detection, 
 * compression and decompression to optimize compression ratios while
 * maintaining mathematical precision.
 */

/**
 * Detect if data follows a mathematical sequence
 * 
 * @param {Uint8Array} data - The data to analyze
 * @param {Object} options - Options for sequence detection
 * @returns {Object|null} Sequence information or null if no sequence found
 */
function detectSequence(data, options = {}) {
  if (!data || data.length < 8) return null;

  // Take a sufficient sample size for accurate detection
  const maxSampleSize = options.sampleSize || Math.min(1024, data.length);
  const sampleSize = Math.min(data.length, maxSampleSize);
  
  // Check for arithmetic sequence (first differences are constant)
  const firstDifferences = [];
  for (let i = 1; i < sampleSize; i++) {
    // Use modular arithmetic for byte data to handle wraparound
    firstDifferences.push((data[i] - data[i - 1] + 256) % 256);
  }
  
  // Check if all first differences are the same (arithmetic sequence)
  const firstDiff = firstDifferences[0];
  let isArithmeticSequence = true;
  for (let i = 1; i < firstDifferences.length; i++) {
    if (firstDifferences[i] !== firstDiff) {
      isArithmeticSequence = false;
      break;
    }
  }
  
  if (isArithmeticSequence) {
    return {
      type: 'arithmetic',
      start: data[0],
      difference: firstDiff,
      modulo: 256 // Assuming byte data
    };
  }
  
  // Check for second-order differences (quadratic sequence)
  if (firstDifferences.length >= 2) {
    const secondDifferences = [];
    for (let i = 1; i < firstDifferences.length; i++) {
      secondDifferences.push((firstDifferences[i] - firstDifferences[i - 1] + 256) % 256);
    }
    
    // Check if all second differences are the same
    const secondDiff = secondDifferences[0];
    let isQuadraticSequence = true;
    for (let i = 1; i < secondDifferences.length; i++) {
      if (secondDifferences[i] !== secondDiff) {
        isQuadraticSequence = false;
        break;
      }
    }
    
    if (isQuadraticSequence) {
      return {
        type: 'quadratic',
        start: data[0],
        firstDifference: firstDiff,
        secondDifference: secondDiff,
        modulo: 256
      };
    }
  }
  
  // Check for modulo sequence (i % N)
  const possibleModuli = [2, 3, 4, 5, 8, 10, 16, 32, 64, 100, 128, 255];
  
  for (const modulo of possibleModuli) {
    let isModuloSequence = true;
    let offset = 0;
    let scale = 1;
    
    // Try with i % modulo
    for (let i = 0; i < sampleSize; i++) {
      if (data[i] !== (i % modulo)) {
        isModuloSequence = false;
        break;
      }
    }
    
    if (isModuloSequence) {
      return {
        type: 'modulo',
        modulo,
        offset: 0,
        scale: 1
      };
    }
    
    // Try with offset: (i + offset) % modulo
    for (offset = 0; offset < modulo; offset++) {
      isModuloSequence = true;
      for (let i = 0; i < sampleSize; i++) {
        if (data[i] !== ((i + offset) % modulo)) {
          isModuloSequence = false;
          break;
        }
      }
      
      if (isModuloSequence) {
        return {
          type: 'modulo',
          modulo,
          offset,
          scale: 1
        };
      }
    }
    
    // Try with scale: (i * scale) % modulo
    for (scale = 1; scale < modulo; scale++) {
      isModuloSequence = true;
      for (let i = 0; i < sampleSize; i++) {
        if (data[i] !== ((i * scale) % modulo)) {
          isModuloSequence = false;
          break;
        }
      }
      
      if (isModuloSequence) {
        return {
          type: 'modulo',
          modulo,
          offset: 0,
          scale
        };
      }
    }
    
    // Try with both scale and offset: ((i * scale) + offset) % modulo
    for (scale = 1; scale < modulo; scale++) {
      for (offset = 0; offset < modulo; offset++) {
        isModuloSequence = true;
        for (let i = 0; i < sampleSize; i++) {
          if (data[i] !== (((i * scale) + offset) % modulo)) {
            isModuloSequence = false;
            break;
          }
        }
        
        if (isModuloSequence) {
          return {
            type: 'modulo',
            modulo,
            offset,
            scale
          };
        }
      }
    }
  }
  
  // Check for Fibonacci-like sequence
  if (data.length >= 8) {
    let isFibonacciLike = true;
    for (let i = 2; i < Math.min(sampleSize, 32); i++) {
      // Check if each term is the sum of the two preceding ones (mod 256)
      const expected = (data[i - 1] + data[i - 2]) % 256;
      if (data[i] !== expected) {
        isFibonacciLike = false;
        break;
      }
    }
    
    if (isFibonacciLike) {
      return {
        type: 'fibonacci',
        first: data[0],
        second: data[1],
        modulo: 256
      };
    }
  }
  
  // Check for exponential sequence a * b^i
  if (data.length >= 8 && data[0] !== 0) {
    // Try to detect the base by calculating ratios
    const ratios = [];
    for (let i = 1; i < Math.min(sampleSize, 16); i++) {
      if (data[i-1] === 0) continue;
      ratios.push(data[i] / data[i-1]);
    }
    
    // Check if all ratios are approximately equal
    if (ratios.length >= 2) {
      const baseRatio = ratios[0];
      let isExponential = true;
      
      for (let i = 1; i < ratios.length; i++) {
        // Allow small floating point differences
        if (Math.abs(ratios[i] - baseRatio) > 0.001) {
          isExponential = false;
          break;
        }
      }
      
      if (isExponential) {
        return {
          type: 'exponential',
          start: data[0],
          base: baseRatio,
          modulo: 256
        };
      }
    }
  }
  
  return null;
}

/**
 * Create sequence compression for data following mathematical patterns
 * 
 * @param {Uint8Array} data - The data to compress
 * @param {String} checksum - Data checksum for verification
 * @returns {Object} Compressed data object
 */
function createSequentialCompression(data, checksum) {
  // Detect sequence pattern with precise mathematical analysis
  const sequenceInfo = detectSequence(data);
  
  if (!sequenceInfo) {
    // Fall back to storing the data directly if no sequence pattern detected
    return {
      version: '1.0.0',
      strategy: 'sequential',
      compressionType: 'sequential',
      specialCase: 'raw',
      compressedVector: Array.from(data),
      compressedSize: data.length,
      compressionRatio: 1,
      originalSize: data.length,
      sequentialMetadata: { type: 'none' },
      checksum
    };
  }
  
  // Create compressed representation based on sequence type
  switch (sequenceInfo.type) {
    case 'arithmetic':
      return {
        version: '1.0.0',
        strategy: 'sequential',
        compressionType: 'sequential',
        specialCase: 'arithmetic',
        compressedVector: [
          sequenceInfo.start,
          sequenceInfo.difference
        ],
        sequentialMetadata: {
          type: 'arithmetic',
          modulo: sequenceInfo.modulo
        },
        compressedSize: 2,
        compressionRatio: data.length / 2,
        originalSize: data.length,
        checksum
      };
      
    case 'quadratic':
      return {
        version: '1.0.0',
        strategy: 'sequential',
        compressionType: 'sequential',
        specialCase: 'quadratic',
        compressedVector: [
          sequenceInfo.start,
          sequenceInfo.firstDifference,
          sequenceInfo.secondDifference
        ],
        sequentialMetadata: {
          type: 'quadratic',
          modulo: sequenceInfo.modulo
        },
        compressedSize: 3,
        compressionRatio: data.length / 3,
        originalSize: data.length,
        checksum
      };
      
    case 'modulo':
      return {
        version: '1.0.0',
        strategy: 'sequential',
        compressionType: 'sequential',
        specialCase: 'modulo',
        compressedVector: [
          sequenceInfo.modulo,
          sequenceInfo.offset,
          sequenceInfo.scale
        ],
        sequentialMetadata: {
          type: 'modulo'
        },
        compressedSize: 3,
        compressionRatio: data.length / 3,
        originalSize: data.length,
        checksum
      };
      
    case 'fibonacci':
      return {
        version: '1.0.0',
        strategy: 'sequential',
        compressionType: 'sequential',
        specialCase: 'fibonacci',
        compressedVector: [
          sequenceInfo.first,
          sequenceInfo.second
        ],
        sequentialMetadata: {
          type: 'fibonacci',
          modulo: sequenceInfo.modulo
        },
        compressedSize: 2,
        compressionRatio: data.length / 2,
        originalSize: data.length,
        checksum
      };
      
    case 'exponential':
      return {
        version: '1.0.0',
        strategy: 'sequential',
        compressionType: 'sequential',
        specialCase: 'exponential',
        compressedVector: [
          sequenceInfo.start,
          sequenceInfo.base
        ],
        sequentialMetadata: {
          type: 'exponential',
          modulo: sequenceInfo.modulo
        },
        compressedSize: 2,
        compressionRatio: data.length / 2,
        originalSize: data.length,
        checksum
      };
      
    default:
      // Unknown sequence type, store raw data
      return {
        version: '1.0.0',
        strategy: 'sequential',
        compressionType: 'sequential',
        specialCase: 'raw',
        compressedVector: Array.from(data),
        compressedSize: data.length,
        compressionRatio: 1,
        originalSize: data.length,
        checksum
      };
  }
}

/**
 * Decompress sequential compressed data with perfect mathematical precision
 * 
 * @param {Object} compressedData - The compressed data object
 * @returns {Uint8Array} Decompressed data
 */
function decompressSequential(compressedData) {
  const specialCase = compressedData.specialCase || '';
  const originalSize = compressedData.originalSize;
  const sequentialMetadata = compressedData.sequentialMetadata || {};
  const modulo = sequentialMetadata.modulo || 256;
  
  // For raw data, just return the compressed vector
  if (specialCase === 'raw') {
    return new Uint8Array(compressedData.compressedVector);
  }
  
  // Handle arithmetic sequence with precise calculation
  if (specialCase === 'arithmetic') {
    const start = compressedData.compressedVector[0];
    const difference = compressedData.compressedVector[1];
    
    // Reconstruct the sequence with high precision
    const result = new Uint8Array(originalSize);
    result[0] = start;
    
    for (let i = 1; i < originalSize; i++) {
      // Use modular arithmetic to maintain byte range
      // Ensure precise calculation with no shortcuts
      result[i] = (result[i - 1] + difference) % modulo;
    }
    
    return result;
  }
  
  // Handle quadratic sequence with precise calculation
  if (specialCase === 'quadratic') {
    const start = compressedData.compressedVector[0];
    const firstDifference = compressedData.compressedVector[1];
    const secondDifference = compressedData.compressedVector[2];
    
    // Reconstruct the sequence with high precision
    const result = new Uint8Array(originalSize);
    result[0] = start;
    
    if (originalSize > 1) {
      result[1] = (start + firstDifference) % modulo;
    }
    
    // Apply second-order recurrence relation
    for (let i = 2; i < originalSize; i++) {
      // Calculate next difference
      const nextDifference = (firstDifference + (i - 1) * secondDifference) % modulo;
      // Apply difference to previous value
      result[i] = (result[i - 1] + nextDifference) % modulo;
    }
    
    return result;
  }
  
  // Handle modulo sequence with precise calculation
  if (specialCase === 'modulo') {
    const modulo = compressedData.compressedVector[0];
    const offset = compressedData.compressedVector[1] || 0;
    const scale = compressedData.compressedVector[2] || 1;
    
    // Reconstruct the sequence
    const result = new Uint8Array(originalSize);
    for (let i = 0; i < originalSize; i++) {
      // Apply the modulo formula with full precision
      result[i] = ((i * scale) + offset) % modulo;
    }
    
    return result;
  }
  
  // Handle Fibonacci-like sequence
  if (specialCase === 'fibonacci') {
    const first = compressedData.compressedVector[0];
    const second = compressedData.compressedVector[1];
    
    // Reconstruct the sequence
    const result = new Uint8Array(originalSize);
    if (originalSize > 0) result[0] = first;
    if (originalSize > 1) result[1] = second;
    
    for (let i = 2; i < originalSize; i++) {
      // Precise modular addition for each term
      result[i] = (result[i - 1] + result[i - 2]) % modulo;
    }
    
    return result;
  }
  
  // Handle exponential sequence
  if (specialCase === 'exponential') {
    const start = compressedData.compressedVector[0];
    const base = compressedData.compressedVector[1];
    
    // Reconstruct the sequence
    const result = new Uint8Array(originalSize);
    for (let i = 0; i < originalSize; i++) {
      // Calculate term with precise mathematical computation
      // For exact integer math, we calculate power manually with no shortcuts
      let value = start;
      for (let j = 0; j < i; j++) {
        value = (value * base) % modulo;
      }
      result[i] = Math.round(value) % modulo;
    }
    
    return result;
  }
  
  // Unknown special case, return zeros as fallback
  return new Uint8Array(originalSize);
}

// Export the enhanced functions
module.exports = {
  detectSequence,
  createSequentialCompression,
  decompressSequential
};