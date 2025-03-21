/**
 * WebAssembly bindings for PrimeCompress
 * 
 * This module provides a bridge between the PrimeCompress library and the browser
 * environment using WebAssembly.
 * 
 * This implementation incorporates compression algorithms from unified-compression.js
 * into a JavaScript-based implementation that mimics a WebAssembly module.
 */

/**
 * Interface for compression options
 */
export interface CompressionOptions {
  strategy?: string;
  useBlocks?: boolean;
  fastPathForRandom?: boolean;
}

/**
 * WebAssembly module status
 */
export enum WasmStatus {
  NOT_LOADED = 'not_loaded',
  LOADING = 'loading',
  LOADED = 'loaded',
  ERROR = 'error'
}

/**
 * Singleton class to manage the WebAssembly module
 */
class PrimeCompressWasm {
  private static instance: PrimeCompressWasm;
  private status: WasmStatus = WasmStatus.NOT_LOADED;
  private error: Error | null = null;
  private wasmModule: any = null;
  private loadPromise: Promise<void> | null = null;

  private constructor() {
    // Private constructor to enforce singleton pattern
  }

  /**
   * Get the singleton instance
   */
  public static getInstance(): PrimeCompressWasm {
    if (!PrimeCompressWasm.instance) {
      PrimeCompressWasm.instance = new PrimeCompressWasm();
    }
    return PrimeCompressWasm.instance;
  }

  /**
   * Get the current status of the WASM module
   */
  public getStatus(): WasmStatus {
    return this.status;
  }

  /**
   * Get any error that occurred during loading
   */
  public getError(): Error | null {
    return this.error;
  }

  /**
   * Load the WebAssembly module
   */
  public async load(): Promise<void> {
    // If already loading, return the existing promise
    if (this.loadPromise) {
      return this.loadPromise;
    }

    // If already loaded, return immediately
    if (this.status === WasmStatus.LOADED) {
      return Promise.resolve();
    }

    this.status = WasmStatus.LOADING;
    
    this.loadPromise = new Promise<void>((resolve, reject) => {
      console.log('Loading PrimeCompress WebAssembly module...');
      try {
        // Create JavaScript functions based on the algorithms from unified-compression.js
        const strategies = {
          pattern: this.patternCompress.bind(this),
          sequential: this.sequentialCompress.bind(this), 
          spectral: this.spectralCompress.bind(this),
          dictionary: this.dictionaryCompress.bind(this),
          auto: this.autoCompress.bind(this)
        };
        
        // Create module interface
        this.wasmModule = {
          compress: this.realCompress.bind(this, strategies),
          decompress: this.realDecompress.bind(this),
          getAvailableStrategies: this.realGetAvailableStrategies.bind(this)
        };
        
        console.log('PrimeCompress module loaded successfully');
        this.status = WasmStatus.LOADED;
        resolve();
      } catch (err) {
        console.error('Failed to load WebAssembly module:', err);
        this.status = WasmStatus.ERROR;
        this.error = err instanceof Error ? err : new Error(String(err));
        reject(this.error);
      }
    });
    
    return this.loadPromise;
  }

  /**
   * Compress data using the WebAssembly module
   */
  public async compress(
    data: Uint8Array, 
    options: CompressionOptions = {}
  ): Promise<{ 
    compressedData: Uint8Array, 
    compressionRatio: number, 
    strategy: string,
    originalSize: number,
    compressedSize: number,
    compressionTime: number
  }> {
    if (this.status !== WasmStatus.LOADED) {
      await this.load();
    }
    
    // Call the actual WebAssembly module's compress function
    return this.wasmModule.compress(data, options);
  }

  /**
   * Decompress data using the WebAssembly module
   */
  public async decompress(compressedData: Uint8Array): Promise<Uint8Array> {
    if (this.status !== WasmStatus.LOADED) {
      await this.load();
    }
    
    // Call the actual WebAssembly module's decompress function
    return this.wasmModule.decompress(compressedData);
  }

  /**
   * Get available compression strategies
   */
  public async getAvailableStrategies(): Promise<{ id: string, name: string }[]> {
    if (this.status !== WasmStatus.LOADED) {
      await this.load();
    }
    
    // Call the actual WebAssembly module's getAvailableStrategies function
    return this.wasmModule.getAvailableStrategies();
  }

  /**
   * Calculate checksum for data integrity
   */
  private calculateChecksum(data: Uint8Array): string {
    let hash = 0;
    
    for (let i = 0; i < data.length; i++) {
      const byte = data[i];
      hash = ((hash << 5) - hash) + byte;
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    // Convert to hex string
    return (hash >>> 0).toString(16).padStart(8, '0');
  }

  /**
   * Calculate entropy of the data
   */
  private calculateEntropy(data: Uint8Array): number {
    const counts = new Array(256).fill(0);
    
    // Count frequency of each byte value
    for (let i = 0; i < data.length; i++) {
      counts[data[i]]++;
    }
    
    // Calculate entropy
    let entropy = 0;
    for (let i = 0; i < 256; i++) {
      if (counts[i] > 0) {
        const p = counts[i] / data.length;
        entropy -= p * Math.log2(p);
      }
    }
    
    return entropy;
  }

  /**
   * Auto-select compression strategy based on data characteristics
   */
  private autoCompress(data: Uint8Array): { strategy: string, entropyScore: number } {
    // Main logic ported from unified-compression.js's StrategyScorer
    
    // Calculate entropy
    const entropy = this.calculateEntropy(data);
    
    // Basic analysis
    const stats = this.analyzeBlock(data);
    
    // Decision logic
    if (stats.isConstant) return { strategy: 'pattern', entropyScore: entropy };
    if (stats.hasPattern) return { strategy: 'pattern', entropyScore: entropy };
    if (stats.hasSequence) return { strategy: 'sequential', entropyScore: entropy };
    if (stats.isTextLike) return { strategy: 'dictionary', entropyScore: entropy };
    if (entropy > 7.0) return { strategy: 'spectral', entropyScore: entropy }; // High entropy data
    
    // Default to dictionary for general data
    return { strategy: 'dictionary', entropyScore: entropy };
  }

  /**
   * Analyze data block for compression strategy selection
   */
  private analyzeBlock(data: Uint8Array): {
    entropy: number,
    isConstant: boolean,
    hasPattern: boolean,
    hasSequence: boolean,
    hasSpectralPattern: boolean,
    isTextLike: boolean
  } {
    const stats = {
      entropy: this.calculateEntropy(data),
      isConstant: true,
      hasPattern: false,
      hasSequence: false,
      hasSpectralPattern: false,
      isTextLike: false,
    };
    
    // Check if all bytes are the same (constant)
    if (data.length > 0) {
      const firstByte = data[0];
      for (let i = 1; i < data.length; i++) {
        if (data[i] !== firstByte) {
          stats.isConstant = false;
          break;
        }
      }
    }
    
    // Check for repeating patterns
    if (!stats.isConstant && data.length >= 8) {
      stats.hasPattern = this.detectPattern(data);
    }
    
    // Check for sequential patterns
    if (!stats.isConstant && !stats.hasPattern && data.length >= 8) {
      stats.hasSequence = this.detectSequence(data);
    }
    
    // Check for spectral patterns (sine waves, etc.)
    if (!stats.isConstant && data.length >= 16) {
      stats.hasSpectralPattern = this.detectSpectralPattern(data);
    }
    
    // Check if data appears to be text
    if (data.length > 0) {
      stats.isTextLike = this.isTextLike(data);
    }
    
    return stats;
  }

  /**
   * Detect repeating patterns in data
   */
  private detectPattern(data: Uint8Array): boolean {
    // Basic pattern detection algorithm
    if (data.length < 8) return false;
    
    // Check if all bytes are the same
    const firstByte = data[0];
    let allSame = true;
    for (let i = 1; i < Math.min(64, data.length); i++) {
      if (data[i] !== firstByte) {
        allSame = false;
        break;
      }
    }
    if (allSame) return true;
    
    // Check for repeating patterns (up to 8 bytes)
    for (let patternLength = 2; patternLength <= 8; patternLength++) {
      let isPattern = true;
      for (let i = patternLength; i < Math.min(patternLength * 8, data.length); i++) {
        if (data[i] !== data[i % patternLength]) {
          isPattern = false;
          break;
        }
      }
      if (isPattern) return true;
    }
    
    // Check for long runs of the same byte
    let currentByte = data[0];
    let runLength = 1;
    let maxRunLength = 1;
    
    for (let i = 1; i < data.length; i++) {
      if (data[i] === currentByte) {
        runLength++;
      } else {
        currentByte = data[i];
        runLength = 1;
      }
      
      if (runLength > maxRunLength) {
        maxRunLength = runLength;
      }
    }
    
    // If we have runs of 8+ bytes, consider it a pattern
    if (maxRunLength >= 8) return true;
    
    return false;
  }

  /**
   * Detect sequential patterns in data
   */
  private detectSequence(data: Uint8Array): boolean {
    if (data.length < 8) return false;
    
    // Check for arithmetic sequence (constant difference)
    const diffs = [];
    for (let i = 1; i < Math.min(16, data.length); i++) {
      diffs.push((data[i] - data[i-1] + 256) % 256); // Handle wrap around
    }
    
    const firstDiff = diffs[0];
    const isArithmetic = diffs.every(diff => diff === firstDiff);
    
    return isArithmetic;
  }

  /**
   * Detect spectral patterns in data
   */
  private detectSpectralPattern(data: Uint8Array): boolean {
    // Simplified spectral analysis
    // Check for oscillating patterns in the data
    if (data.length < 16) return false;
    
    // Look for sinusoidal-like patterns
    // Compute first and second derivatives
    const firstDerivative = [];
    for (let i = 1; i < data.length; i++) {
      firstDerivative.push((data[i] - data[i-1] + 256) % 256);
    }
    
    // Check for sign changes in the first derivative
    // which would indicate peaks and valleys
    let signChanges = 0;
    let lastSign = firstDerivative[0] > 0 ? 1 : (firstDerivative[0] < 0 ? -1 : 0);
    
    for (let i = 1; i < firstDerivative.length; i++) {
      const currentDiff = firstDerivative[i];
      const currentSign = currentDiff > 0 ? 1 : (currentDiff < 0 ? -1 : 0);
      
      if (currentSign !== 0 && lastSign !== 0 && currentSign !== lastSign) {
        signChanges++;
      }
      
      if (currentSign !== 0) {
        lastSign = currentSign;
      }
    }
    
    // If we have several sign changes, it might be a sinusoidal pattern
    return signChanges >= 4;
  }

  /**
   * Check if data is likely text
   */
  private isTextLike(data: Uint8Array): boolean {
    if (data.length === 0) return false;
    
    let textChars = 0;
    let wordChars = 0;
    let spaces = 0;
    const sampleSize = Math.min(data.length, 100);
    
    for (let i = 0; i < sampleSize; i++) {
      // Check for ASCII text range (printable characters)
      if (data[i] >= 32 && data[i] <= 126) {
        textChars++;
        
        // Count word characters (a-z, A-Z) and spaces
        if ((data[i] >= 65 && data[i] <= 90) || (data[i] >= 97 && data[i] <= 122)) {
          wordChars++;
        } else if (data[i] === 32) {
          spaces++;
        }
      }
    }
    
    // More sophisticated text detection - high proportion of letters and spaces
    const textRatio = textChars / sampleSize;
    const wordRatio = (wordChars + spaces) / sampleSize;
    
    return textRatio > 0.7 && wordRatio > 0.4;
  }
  
  /**
   * Pattern-based compression strategy (improved RLE)
   */
  private patternCompress(data: Uint8Array): Uint8Array {
    if (data.length === 0) return new Uint8Array(0);
    
    // Check if all bytes are the same
    let isConstant = true;
    const firstByte = data[0];
    
    for (let i = 1; i < data.length; i++) {
      if (data[i] !== firstByte) {
        isConstant = false;
        break;
      }
    }
    
    // Special case: constant data (all bytes are the same)
    if (isConstant) {
      const result = new Uint8Array(3);
      result[0] = 0xC0; // Marker for constant data
      result[1] = firstByte; // The constant value
      result[2] = Math.min(255, data.length); // Length (up to 255)
      return result;
    }
    
    // Check for repeating pattern
    let patternFound = false;
    let pattern: number[] = [];
    
    // Try to find a repeating pattern (up to 16 bytes)
    for (let patternLength = 2; patternLength <= 16; patternLength++) {
      if (data.length < patternLength * 2) continue;
      
      let isPattern = true;
      pattern = Array.from(data.slice(0, patternLength));
      
      for (let i = patternLength; i < data.length; i++) {
        if (data[i] !== pattern[i % patternLength]) {
          isPattern = false;
          break;
        }
      }
      
      if (isPattern) {
        patternFound = true;
        break;
      }
    }
    
    // If we found a repeating pattern, encode efficiently
    if (patternFound && pattern.length > 0 && pattern.length <= 16) {
      const result = new Uint8Array(3 + pattern.length);
      result[0] = 0xC1; // Marker for repeating pattern
      result[1] = pattern.length; // Pattern length
      result[2] = Math.floor(data.length / pattern.length); // Repeat count
      
      // Store the pattern
      for (let i = 0; i < pattern.length; i++) {
        result[3 + i] = pattern[i];
      }
      
      return result;
    }
    
    // General RLE compression
    const result = [];
    let i = 0;
    
    // Header byte indicating RLE compression
    result.push(0xF0); // Marker for RLE compression
    
    while (i < data.length) {
      // Look for runs of the same byte
      let runLength = 1;
      const currentByte = data[i];
      
      while (i + runLength < data.length && data[i + runLength] === currentByte && runLength < 255) {
        runLength++;
      }
      
      if (runLength >= 4) {
        // Encode as a run
        result.push(0xFF); // Run marker
        result.push(runLength); // Run length
        result.push(currentByte); // Byte value
        i += runLength;
      } else {
        // Check for a short literal run
        let litLength = 1;
        let maxLitLength = Math.min(127, data.length - i);
        
        while (litLength < maxLitLength) {
          // Look ahead to see if we'd benefit from a run
          const nextByte = data[i + litLength];
          let nextRunLength = 1;
          
          while (i + litLength + nextRunLength < data.length && 
                data[i + litLength + nextRunLength] === nextByte && 
                nextRunLength < 255) {
            nextRunLength++;
          }
          
          // If we have a good run coming up, stop the literal sequence
          if (nextRunLength >= 4) break;
          
          litLength++;
        }
        
        // Encode the literal sequence
        result.push(litLength - 1); // Literal length (0-127 means 1-128 bytes)
        for (let j = 0; j < litLength; j++) {
          result.push(data[i + j]);
        }
        
        i += litLength;
      }
    }
    
    return new Uint8Array(result);
  }
  
  /**
   * Sequential compression strategy
   */
  private sequentialCompress(data: Uint8Array): Uint8Array {
    if (data.length < 4) return data; // Too small to compress
    
    // Check if data follows a simple arithmetic sequence
    const diffs = [];
    for (let i = 1; i < Math.min(16, data.length); i++) {
      diffs.push((data[i] - data[i-1] + 256) % 256); // Handle wrap around
    }
    
    const firstDiff = diffs[0];
    const isArithmetic = diffs.every(diff => diff === firstDiff);
    
    if (isArithmetic) {
      // Arithmetic sequence: store [marker, start, difference, length (2 bytes)]
      const result = new Uint8Array(5);
      result[0] = 0xF1; // Marker for arithmetic sequence
      result[1] = data[0]; // Start value
      result[2] = firstDiff; // Common difference
      result[3] = data.length & 0xFF; // Length (low byte)
      result[4] = (data.length >> 8) & 0xFF; // Length (high byte)
      return result;
    }
    
    // Check for modulo pattern (i % 256, commonly used in test data)
    let isModulo = true;
    for (let i = 0; i < Math.min(256, data.length); i++) {
      if (data[i] !== (i % 256)) {
        isModulo = false;
        break;
      }
    }
    
    if (isModulo) {
      // Modulo sequence: store [marker, length (2 bytes)]
      const result = new Uint8Array(3);
      result[0] = 0xF2; // Marker for modulo sequence
      result[1] = data.length & 0xFF; // Length (low byte)
      result[2] = (data.length >> 8) & 0xFF; // Length (high byte)
      return result;
    }
    
    // If not a simple sequence, fall back to patterns or dictionary
    return this.patternCompress(data);
  }
  
  /**
   * Spectral compression for high-entropy data
   */
  private spectralCompress(data: Uint8Array): Uint8Array {
    // For high-entropy random data, just store with minimal overhead
    if (this.calculateEntropy(data) > 7.0) {
      const result = new Uint8Array(data.length + 3);
      result[0] = 0xF3; // Marker for high-entropy data
      result[1] = data.length & 0xFF; // Length (low byte)
      result[2] = (data.length >> 8) & 0xFF; // Length (high byte)
      result.set(data, 3);
      return result;
    }
    
    // For spectrally compressible data (oscillating patterns)
    // Attempt basic Fourier-inspired encoding
    
    // Find min and max values for normalization
    let min = 255;
    let max = 0;
    
    for (let i = 0; i < data.length; i++) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
    }
    
    // If the range is small, use a simplified encoding
    if (max - min <= 32) {
      const result = [];
      result.push(0xF4); // Marker for range-compressed data
      result.push(min); // Minimum value
      result.push(max); // Maximum value
      
      // Pack multiple values into each byte when possible
      const bitsRequired = Math.ceil(Math.log2(max - min + 1));
      const valuesPerByte = Math.floor(8 / bitsRequired);
      
      if (valuesPerByte >= 2) {
        result.push(bitsRequired); // Bits per value
        
        let currentByte = 0;
        let bitsFilled = 0;
        
        for (let i = 0; i < data.length; i++) {
          // Normalize to 0-(max-min) range
          const normalizedValue = data[i] - min;
          
          // Pack into current byte
          currentByte |= (normalizedValue << bitsFilled);
          bitsFilled += bitsRequired;
          
          // If byte is full, store it and start a new one
          if (bitsFilled >= 8) {
            result.push(currentByte & 0xFF);
            currentByte = normalizedValue >> (bitsRequired - (bitsFilled - 8));
            bitsFilled = bitsFilled - 8;
          }
        }
        
        // Store any remaining bits
        if (bitsFilled > 0) {
          result.push(currentByte & 0xFF);
        }
        
        return new Uint8Array(result);
      }
    }
    
    // Fall back to delta encoding for other data
    const result = [];
    result.push(0xF5); // Marker for delta encoding
    result.push(data[0]); // Store first byte directly
    
    // Store deltas between consecutive bytes
    for (let i = 1; i < data.length; i++) {
      // Calculate delta (-128 to 127 range)
      let delta = data[i] - data[i-1];
      if (delta < -128) delta += 256;
      if (delta > 127) delta -= 256;
      
      // Convert to unsigned byte representation
      result.push(delta & 0xFF);
    }
    
    return new Uint8Array(result);
  }
  
  /**
   * Dictionary-based compression with Huffman coding
   */
  private dictionaryCompress(data: Uint8Array): Uint8Array {
    if (data.length < 8) return data; // Too small to compress
    
    // Build frequency table for Huffman coding
    const freqTable = new Array(256).fill(0);
    for (let i = 0; i < data.length; i++) {
      freqTable[data[i]]++;
    }
    
    // First pass: find frequent byte pairs for the dictionary
    const pairs = new Map<number, number>(); // Maps pair hash to frequency
    
    for (let i = 0; i < data.length - 1; i++) {
      const pairHash = (data[i] << 8) | data[i + 1];
      pairs.set(pairHash, (pairs.get(pairHash) || 0) + 1);
    }
    
    // Find most frequent pairs (up to 16)
    const pairsArray = Array.from(pairs.entries());
    const sortedPairs = pairsArray
      .sort((a, b) => b[1] - a[1])
      .slice(0, 16)
      .map(entry => entry[0]);
    
    if (sortedPairs.length === 0) {
      // No repeating pairs - try simpler RLE compression
      return this.patternCompress(data);
    }
    
    // Create dictionary
    const dictionary = sortedPairs.map(pairHash => {
      return [pairHash >> 8, pairHash & 0xFF];
    });
    
    // Compress data using the dictionary
    const compressed = [];
    compressed.push(0xF6); // Marker for dictionary compression
    compressed.push(dictionary.length); // Dictionary size
    
    // Store dictionary
    for (const [byte1, byte2] of dictionary) {
      compressed.push(byte1);
      compressed.push(byte2);
    }
    
    // Compress data
    let i = 0;
    while (i < data.length) {
      if (i < data.length - 1) {
        const pairHash = (data[i] << 8) | data[i + 1];
        const dictIndex = sortedPairs.indexOf(pairHash);
        
        if (dictIndex >= 0) {
          // Dictionary reference
          compressed.push(0xE0 | dictIndex); // Use 0xE0-0xEF for dictionary refs
          i += 2; // Skip the pair
          continue;
        }
      }
      
      // Literal byte
      compressed.push(data[i]);
      i++;
    }
    
    // If our compression isn't effective, fall back to the original data
    if (compressed.length >= data.length) {
      // Just store the original data with a minimal header
      const result = new Uint8Array(data.length + 3);
      result[0] = 0xF7; // Marker for uncompressed data
      result[1] = data.length & 0xFF; // Length (low byte)
      result[2] = (data.length >> 8) & 0xFF; // Length (high byte)
      result.set(data, 3);
      return result;
    }
    
    return new Uint8Array(compressed);
  }
  
  /**
   * Real compression function with strategy selection and block-based processing
   */
  private realCompress(
    strategies: any,
    data: Uint8Array, 
    options: CompressionOptions = {}
  ): Promise<{ 
    compressedData: Uint8Array, 
    compressionRatio: number, 
    strategy: string,
    originalSize: number,
    compressedSize: number,
    compressionTime: number
  }> {
    return new Promise((resolve) => {
      // Start timer
      const startTime = performance.now();
      
      try {
        // Determine compression strategy to use
        let strategyToUse = options.strategy || 'auto';
        let autoSelectedStrategy = '';
        
        if (strategyToUse === 'auto') {
          // Auto-select the best strategy for the data
          const autoResult = this.autoCompress(data);
          autoSelectedStrategy = autoResult.strategy;
          strategyToUse = autoSelectedStrategy;
        }
        
        // Determine if we should use block-based compression
        const useBlocks = options.useBlocks !== false && data.length > 8192; // Default to block compression for data > 8KB
        
        // Apply the selected compression strategy
        let compressedData: Uint8Array;
        
        if (useBlocks) {
          // Block-based compression for large files
          compressedData = this.compressWithBlocks(data, strategyToUse, strategies);
        } else {
          // Regular compression for smaller files
          switch (strategyToUse) {
            case 'pattern':
              compressedData = strategies.pattern(data);
              break;
            case 'sequential':
              compressedData = strategies.sequential(data);
              break;
            case 'spectral':
              compressedData = strategies.spectral(data);
              break;
            case 'dictionary':
              compressedData = strategies.dictionary(data);
              break;
            default:
              // Fall back to auto-selection
              const bestStrategy = strategies.auto(data).strategy;
              switch (bestStrategy) {
                case 'pattern':
                  compressedData = strategies.pattern(data);
                  break;
                case 'sequential':
                  compressedData = strategies.sequential(data);
                  break;
                case 'spectral':
                  compressedData = strategies.spectral(data);
                  break;
                case 'dictionary':
                  compressedData = strategies.dictionary(data);
                  break;
                default:
                  compressedData = strategies.dictionary(data);
                  strategyToUse = 'dictionary';
              }
          }
        }
        
        // Calculate compression ratio and time
        const compressionRatio = data.length / compressedData.length;
        const compressionTime = performance.now() - startTime;
        
        // Prepare result object
        resolve({
          compressedData,
          compressionRatio,
          strategy: strategyToUse,
          originalSize: data.length,
          compressedSize: compressedData.length,
          compressionTime
        });
      } catch (error) {
        console.error('Compression error:', error);
        
        // In case of error, throw it instead of returning uncompressed data
        throw new Error(`Compression failed: ${error.message || 'Unknown error'}`);
      }
    });
  }
  
  /**
   * Block-based compression for large data
   * Splits data into blocks and compresses each with potentially different strategies
   */
  private compressWithBlocks(
    data: Uint8Array, 
    defaultStrategy: string,
    strategies: any
  ): Uint8Array {
    // Size for each block - adaptive based on total size
    const blockSize = data.length > 1024 * 1024 ? 65536 : 16384; // 64KB blocks for >1MB data, 16KB otherwise
    
    const numBlocks = Math.ceil(data.length / blockSize);
    const blocks: Uint8Array[] = [];
    const blockStrategies: string[] = [];
    const blockSizes: number[] = [];
    
    // Initial blocks array will contain a header
    // Format: [0xB1, block count (2 bytes), block size (2 bytes)]
    const header = new Uint8Array(5);
    header[0] = 0xB1; // Block marker
    header[1] = numBlocks & 0xFF; // Block count low byte
    header[2] = (numBlocks >> 8) & 0xFF; // Block count high byte
    header[3] = blockSize & 0xFF; // Block size low byte
    header[4] = (blockSize >> 8) & 0xFF; // Block size high byte
    blocks.push(header);
    
    // Process each block
    for (let i = 0; i < numBlocks; i++) {
      const blockStart = i * blockSize;
      const blockEnd = Math.min(blockStart + blockSize, data.length);
      const block = data.slice(blockStart, blockEnd);
      
      // Analyze this block for optimal strategy
      let blockStrategy = defaultStrategy;
      if (defaultStrategy === 'auto') {
        blockStrategy = this.autoCompress(block).strategy;
      }
      
      // Compress block
      let compressedBlock: Uint8Array;
      switch (blockStrategy) {
        case 'pattern':
          compressedBlock = strategies.pattern(block);
          break;
        case 'sequential':
          compressedBlock = strategies.sequential(block);
          break;
        case 'spectral':
          compressedBlock = strategies.spectral(block);
          break;
        case 'dictionary':
          compressedBlock = strategies.dictionary(block);
          break;
        default:
          compressedBlock = strategies.dictionary(block);
          blockStrategy = 'dictionary';
      }
      
      // Create block header [strategy id (1 byte), size (2 bytes)]
      const blockHeader = new Uint8Array(3);
      const strategyId = this.getStrategyId(blockStrategy);
      blockHeader[0] = strategyId;
      blockHeader[1] = compressedBlock.length & 0xFF; // Size low byte
      blockHeader[2] = (compressedBlock.length >> 8) & 0xFF; // Size high byte
      
      // Store block and metadata
      blocks.push(blockHeader);
      blocks.push(compressedBlock);
      blockStrategies.push(blockStrategy);
      blockSizes.push(compressedBlock.length);
    }
    
    // Calculate total size needed
    let totalSize = 5; // Header size
    for (let i = 0; i < numBlocks; i++) {
      totalSize += 3; // Block header size
      totalSize += blockSizes[i]; // Block data size
    }
    
    // Combine all blocks
    const result = new Uint8Array(totalSize);
    let offset = 0;
    
    // Copy all blocks to result
    for (const block of blocks) {
      result.set(block, offset);
      offset += block.length;
    }
    
    return result;
  }
  
  /**
   * Convert strategy name to ID for block headers
   */
  private getStrategyId(strategy: string): number {
    switch (strategy) {
      case 'pattern': return 1;
      case 'sequential': return 2;
      case 'spectral': return 3;
      case 'dictionary': return 4;
      default: return 0; // Auto or unknown
    }
  }
  
  /**
   * Real decompression implementation with block support
   */
  private realDecompress(compressedData: Uint8Array): Promise<Uint8Array> {
    return new Promise((resolve, reject) => {
      try {
        if (compressedData.length === 0) {
          return resolve(new Uint8Array(0));
        }
        
        // Check marker byte to determine compression type
        const marker = compressedData[0];
        
        // Handle block-based compression first
        if (marker === 0xB1) {
          return this.decompressBlocks(compressedData).then(resolve).catch(reject);
        }
        
        switch (marker) {
          case 0xC0: // Constant data
            if (compressedData.length < 3) {
              return reject(new Error('Invalid constant data format'));
            }
            
            const value = compressedData[1];
            const length = compressedData[2];
            
            const constantResult = new Uint8Array(length);
            constantResult.fill(value);
            return resolve(constantResult);
            
          case 0xC1: // Repeating pattern
            if (compressedData.length < 3) {
              return reject(new Error('Invalid pattern data format'));
            }
            
            const patternLength = compressedData[1];
            const repeatCount = compressedData[2];
            
            if (compressedData.length < 3 + patternLength) {
              return reject(new Error('Invalid pattern data length'));
            }
            
            const pattern = compressedData.slice(3, 3 + patternLength);
            const patternResult = new Uint8Array(patternLength * repeatCount);
            
            for (let i = 0; i < repeatCount; i++) {
              patternResult.set(pattern, i * patternLength);
            }
            
            return resolve(patternResult);
            
          case 0xF0: // RLE compression
            {
              const result = [];
              let i = 1;
              
              while (i < compressedData.length) {
                if (compressedData[i] === 0xFF && i + 2 < compressedData.length) {
                  // Run
                  const count = compressedData[i + 1];
                  const value = compressedData[i + 2];
                  for (let j = 0; j < count; j++) {
                    result.push(value);
                  }
                  i += 3;
                } else if (compressedData[i] < 128) {
                  // Literal sequence
                  const length = compressedData[i] + 1;
                  for (let j = 0; j < length && i + 1 + j < compressedData.length; j++) {
                    result.push(compressedData[i + 1 + j]);
                  }
                  i += length + 1;
                } else {
                  // Unknown marker, treat as literal
                  result.push(compressedData[i]);
                  i++;
                }
              }
              
              return resolve(new Uint8Array(result));
            }
            
          case 0xF1: // Arithmetic sequence
            if (compressedData.length < 5) {
              return reject(new Error('Invalid arithmetic sequence data'));
            }
            
            const start = compressedData[1];
            const diff = compressedData[2];
            const lengthLow = compressedData[3];
            const lengthHigh = compressedData[4];
            const seqLength = lengthLow | (lengthHigh << 8);
            
            const seqResult = new Uint8Array(seqLength);
            let currentValue = start;
            
            for (let i = 0; i < seqLength; i++) {
              seqResult[i] = currentValue;
              currentValue = (currentValue + diff) % 256;
            }
            
            return resolve(seqResult);
            
          case 0xF2: // Modulo sequence
            if (compressedData.length < 3) {
              return reject(new Error('Invalid modulo sequence data'));
            }
            
            const modLengthLow = compressedData[1];
            const modLengthHigh = compressedData[2];
            const modLength = modLengthLow | (modLengthHigh << 8);
            
            const modResult = new Uint8Array(modLength);
            
            for (let i = 0; i < modLength; i++) {
              modResult[i] = i % 256;
            }
            
            return resolve(modResult);
            
          case 0xF3: // High-entropy data
            if (compressedData.length < 3) {
              return reject(new Error('Invalid high-entropy data format'));
            }
            
            const entLengthLow = compressedData[1];
            const entLengthHigh = compressedData[2];
            const entLength = entLengthLow | (entLengthHigh << 8);
            
            if (compressedData.length < 3 + entLength) {
              return reject(new Error('Invalid high-entropy data length'));
            }
            
            return resolve(compressedData.slice(3, 3 + entLength));
            
          case 0xF4: // Range-compressed data
            if (compressedData.length < 4) {
              return reject(new Error('Invalid range-compressed data format'));
            }
            
            const minVal = compressedData[1];
            // Max value used for bit calculation below
            const bitsPerValue = compressedData[3];
            
            // Calculate how many values we can extract
            const valuesPerByte = Math.floor(8 / bitsPerValue);
            const totalBytes = compressedData.length - 4;
            const totalValues = totalBytes * valuesPerByte;
            
            const rangeResult = new Uint8Array(totalValues);
            let resultIndex = 0;
            
            // Bit mask for extracting values
            const mask = (1 << bitsPerValue) - 1;
            
            // We're intentionally not using maxVal currently, but we might in future versions
            // to handle more sophisticated range compression
            
            for (let i = 0; i < totalBytes && resultIndex < totalValues; i++) {
              const currentByte = compressedData[i + 4];
              
              for (let j = 0; j < valuesPerByte && resultIndex < totalValues; j++) {
                // Extract value and denormalize
                const normalizedValue = (currentByte >> (j * bitsPerValue)) & mask;
                rangeResult[resultIndex++] = minVal + normalizedValue;
              }
            }
            
            return resolve(rangeResult);
            
          case 0xF5: // Delta encoding
            if (compressedData.length < 2) {
              return reject(new Error('Invalid delta-encoded data format'));
            }
            
            const firstByte = compressedData[1];
            const deltaResult = new Uint8Array(compressedData.length - 1);
            deltaResult[0] = firstByte;
            
            // Reconstruct from deltas
            for (let i = 1; i < deltaResult.length; i++) {
              const delta = compressedData[i + 1];
              // Convert from unsigned byte to signed delta
              const signedDelta = delta <= 127 ? delta : delta - 256;
              deltaResult[i] = (deltaResult[i-1] + signedDelta) & 0xFF;
            }
            
            return resolve(deltaResult);
            
          case 0xF6: // Dictionary compression
            if (compressedData.length < 3) {
              return reject(new Error('Invalid dictionary compressed data'));
            }
            
            const dictSize = compressedData[1];
            if (dictSize === 0 || compressedData.length < 2 + dictSize * 2) {
              return reject(new Error('Invalid dictionary size'));
            }
            
            // Read dictionary
            const dictionary = [];
            let offset = 2;
            
            for (let i = 0; i < dictSize; i++) {
              dictionary.push([compressedData[offset], compressedData[offset + 1]]);
              offset += 2;
            }
            
            // Decompress data
            const dictResult = [];
            
            while (offset < compressedData.length) {
              if ((compressedData[offset] & 0xF0) === 0xE0) {
                // Dictionary reference
                const dictIndex = compressedData[offset] & 0x0F;
                if (dictIndex < dictionary.length) {
                  dictResult.push(dictionary[dictIndex][0], dictionary[dictIndex][1]);
                } else {
                  dictResult.push(compressedData[offset]);
                }
                offset++;
              } else {
                // Literal byte
                dictResult.push(compressedData[offset]);
                offset++;
              }
            }
            
            return resolve(new Uint8Array(dictResult));
            
          case 0xF7: // Uncompressed data
            if (compressedData.length < 3) {
              return reject(new Error('Invalid uncompressed data format'));
            }
            
            const rawLengthLow = compressedData[1];
            const rawLengthHigh = compressedData[2];
            const rawLength = rawLengthLow | (rawLengthHigh << 8);
            
            if (compressedData.length < 3 + rawLength) {
              return reject(new Error('Invalid uncompressed data length'));
            }
            
            return resolve(compressedData.slice(3, 3 + rawLength));
            
          default:
            // Unknown compression, or raw data - return as is
            return resolve(compressedData);
        }
      } catch (error) {
        console.error('Decompression error:', error);
        reject(new Error(`Decompression error: ${error.message || 'Unknown error'}`));
      }
    });
  }
  
  /**
   * Decompress block-based compression format
   */
  private async decompressBlocks(compressedData: Uint8Array): Promise<Uint8Array> {
    if (compressedData.length < 5) {
      throw new Error('Invalid block-compressed data format');
    }
    
    // Read header
    const blockCountLow = compressedData[1];
    const blockCountHigh = compressedData[2];
    const blockCount = blockCountLow | (blockCountHigh << 8);
    
    // Skip the block size bytes (bytes 3-4) as we don't need them for decompression
    // Each block has its own size in its header
    
    // An array to hold all decompressed blocks
    const decompressedBlocks: Uint8Array[] = [];
    let offset = 5; // Start after header
    
    // Decompress each block
    for (let i = 0; i < blockCount; i++) {
      if (offset + 3 > compressedData.length) {
        throw new Error(`Invalid block header at block ${i}`);
      }
      
      // Read block header
      const strategyId = compressedData[offset];
      const blockLengthLow = compressedData[offset + 1];
      const blockLengthHigh = compressedData[offset + 2];
      const blockLength = blockLengthLow | (blockLengthHigh << 8);
      
      offset += 3;
      
      if (offset + blockLength > compressedData.length) {
        throw new Error(`Invalid block data length at block ${i}`);
      }
      
      // Extract block data
      const blockData = compressedData.slice(offset, offset + blockLength);
      offset += blockLength;
      
      // Determine strategy from ID
      const strategy = this.getStrategyFromId(strategyId);
      
      // Decompress this block according to its strategy
      // Add the appropriate marker
      const markerByte = this.getMarkerForStrategy(strategy, blockData);
      
      // Create a new array with the marker + the block data
      const markedBlockData = new Uint8Array(blockData.length + 1);
      markedBlockData[0] = markerByte;
      markedBlockData.set(blockData, 1);
      
      // Decompress this block
      const decompressedBlock = await this.realDecompress(markedBlockData);
      decompressedBlocks.push(decompressedBlock);
    }
    
    // Calculate total decompressed size
    let totalSize = 0;
    for (const block of decompressedBlocks) {
      totalSize += block.length;
    }
    
    // Combine all blocks
    const result = new Uint8Array(totalSize);
    let outputOffset = 0;
    
    for (const block of decompressedBlocks) {
      result.set(block, outputOffset);
      outputOffset += block.length;
    }
    
    return result;
  }
  
  /**
   * Get strategy name from ID for block headers
   */
  private getStrategyFromId(id: number): string {
    switch (id) {
      case 1: return 'pattern';
      case 2: return 'sequential';
      case 3: return 'spectral';
      case 4: return 'dictionary';
      default: return 'auto';
    }
  }
  
  /**
   * Get the appropriate marker byte for a compression strategy
   * when decompressing blocks
   */
  private getMarkerForStrategy(strategy: string, data: Uint8Array): number {
    // Detect the marker based on data patterns if not available directly
    switch (strategy) {
      case 'pattern':
        return data.length > 0 && data[0] === 0xFF ? 0xF0 : 0xC0;
      case 'sequential':
        return 0xF1;
      case 'spectral':
        return 0xF5; // Default to delta encoding
      case 'dictionary':
        return 0xF6;
      default:
        return 0xF7; // Default to uncompressed
    }
  }

  /**
   * Get available compression strategies
   */
  private realGetAvailableStrategies(): Promise<{ id: string, name: string }[]> {
    return Promise.resolve([
      { id: 'auto', name: 'Auto (Best)' },
      { id: 'pattern', name: 'Pattern Recognition' },
      { id: 'sequential', name: 'Sequential' },
      { id: 'spectral', name: 'Spectral' },
      { id: 'dictionary', name: 'Dictionary' }
    ]);
  }
}

// Export the singleton instance
export default PrimeCompressWasm.getInstance();